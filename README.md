# CharacterStrokeGenerator

Turns a CJK glyph outline SVG (Kai style, as exported by `training_data/scripts/export_glyphs.py`) into
[hanzi-writer](https://hanziwriter.org) data: `{"strokes": [...], "medians": [...]}`.

## Pipeline

1. **Cut (learned, Stage A).** A transformer over outline segments predicts the vertices where one stroke
   ends and the next begins, plus a same-stroke affinity for every pair of segments. Cuts split the
   contours into chains, and chains are clustered into strokes. Each stroke is closed with bridge
   segments: straight lines, or tangent-continuous cubic curves like the ones in the training data.
   Outline segments are never changed, so the strokes together render the original glyph.
2. **Medians (geometry, Stage B).** Voronoi skeleton of each stroke → longest path → corner spurs
   trimmed → Ramer-Douglas-Peucker simplification.
3. **Order and direction (learned, Stage C).** A transformer over strokes, using per-stroke shape and
   pairwise relation features, predicts which way each median runs and the writing order.

Coordinates use the hanzi-writer frame: `x_hw = x_svg`, `y_hw = 900 − y_svg` for the 1024-unit Arphic exports.

## Usage

```sh
uv sync                                   # Python 3.12 venv with CUDA torch
uv run python -m strokegen.build_dataset  # labels -> data/processed/{train,val,test}.pkl
uv run python -m strokegen.train_cut      # Stage A -> checkpoints/cut.pt
uv run python -m strokegen.train_order    # Stage C -> checkpoints/order.pt
uv run python -m strokegen.evaluate       # test-split metrics + outputs/preview.html
uv run python -m strokegen.evaluate --chars 永林  # specific characters -> outputs/preview_chars.html
uv run python -m strokegen.predict training_data/font_glyphs/ukai-cn/永.svg -o 永.json
uv run python -m strokegen.predict some_glyph_dir/ -o out_dir/
uv run --group dev pytest tests
uv run python tools/dashboard.py          # review dashboard on http://127.0.0.1:8417
```

### Review dashboard

`tools/dashboard.py` serves a local page for turning glyphs into training data by hand:

1. Pick glyphs from any `training_data/font_glyphs/<font>` directory (search by character or
   `U+XXXX`, Shift-click for ranges), or upload your own SVG files.
2. Press **Process**: each glyph runs through `strokegen.predict` on the server.
3. Check the result in hanzi-writer (animate, loop, quiz) next to the current training data file.
   The **Writing test** card lets you write the character yourself, against either the prediction
   or the training file, and shows per stroke whether it was recognized and how many misses it
   took. **Allow reversed** accepts strokes drawn backwards and marks them, which points at a
   median with the wrong direction. **R** restarts it.
4. Reorder strokes by dragging the tiles, with ◀ ▶, or by typing an order; ⇄ reverses a stroke.
   ⊕ (or **M** on a focused tile) merges a stroke with the next one: the server rebuilds one
   outline from the font segments of both and runs the median through it, so the result looks
   like a predicted stroke. If the outline's skeleton can't run from the start of the first
   stroke to the end of the second (for example, the strokes cross or don't touch), the two
   medians are joined end to end instead. ✂ splits a merged stroke again.
5. **Save** writes `training_data/hanzi-writer-data-youyin/data/{char}.json` (or a `-jp`/`-ko`
   variant) and adds plain files to the font glyph manifest so `build_dataset` uses them.
   Overwriting an existing file needs an explicit **Overwrite** click. Existing `radStrokes`
   are dropped because they refer to the old strokes.

## Training data

Only plain `{char}.json` files are used; `-jp`/`-ko` variants are ignored. About 9,500 characters
match a font exactly: their hw strokes are cut from the gkai00mp, bkai00mp or ukai-cn outline. The
~1,660 that don't match come from AnimCJK outlines (kana, Japan-only kanji) and are skipped.
Splits are by character, 90/5/5 with a fixed seed.

## Results (held-out test split, 475 characters)

| Metric | Value |
|---|---|
| Strokes cut exactly right | 99.2% |
| Glyphs where every stroke is cut exactly right | 94.7% |
| Union of strokes vs. original render (IoU, mean / worst) | 0.9993 / 0.991 |
| Median distance to GT (1024-unit em) | 4.8 |
| Stroke direction accuracy | 99.9% |
| Exact stroke order (GT strokes / end to end) | 93.1% / 88.6% |
| Glyphs fully correct (cut + order + direction) | 88.6% |

`outputs/preview.html` shows predictions next to the GT, with hanzi-writer animation.
