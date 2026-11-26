# CharacterStrokeGenerator
A machine learning model and toolchain for turning regular font SVGs of characters into SVGs
split into strokes with medians in [hanzi-writer](https://github.com/chanind/hanzi-writer) format.

## Usage
We use this ML model to automatically generate and split characters to feed 
[hanzi-writer-data-youyin](https://github.com/MadLadSquad/hanzi-writer-data-youyin) with additional character data.

Our goal is to completely automate away splitting any character from any font from any writing system into strokes.
Such a model can produce valuable data with applications in:

- Live handwriting recognition
- Language-learning apps
- Automated generation of learning sheets for calligraphers and pre-school children

The model is fed on about 15K characters from [hanzi-writer-data-youyin](https://github.com/MadLadSquad/hanzi-writer-data-youyin),
of which 11K were in the original dataset and the rest are data generated from the model that has passed human verification.

## Pipeline

1. **Cut (learned, Stage A).** A transformer over outline segments predicts the vertices where one stroke
   ends and the next begins, plus a same-stroke affinity for every pair of segments. Cuts split the
   contours into chains, and chains are clustered into strokes. Each stroke is closed with bridge
   segments: straight lines, or tangent-continuous cubic curves like the ones in the training data
   (a curve that would bulge out of the glyph, as at the square stroke ends of sans fonts, is
   straightened). Outline segments are never changed, so the strokes together render the original glyph.
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
uv run --group dev ruff check .
uv run python tools/dashboard.py          # review dashboard on http://127.0.0.1:8417
```

Training needs a CUDA GPU. Checkpoints record the feature version they were trained on, and
`StrokeGenerator` refuses one that does not match the code; retrain after a feature change.

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
   Overwriting an existing file needs an explicit **Overwrite** click (also when saving with
   **S**). Existing `radStrokes` are dropped because they refer to the old strokes.

The dashboard has no login. It only answers requests addressed to `127.0.0.1`/`localhost` and
refuses cross-site writes; with `--host 0.0.0.0` anyone on the network can change the training
data. The first time a font without exported glyphs is opened, its glyphs are unpacked from the
font file in the background (this needs FontForge's Python bindings).

`tools/` also has command-line correction tools (`fix_hole_strokes.py`, `median_repair.py`,
`median_reroute.py`, `family_transfer.py`, `verify_glyph.py`, `render_compare.py`, ...).
`tools/scratch/` keeps one-off scripts from earlier manual passes; see its README before running one.

## How we fix issues with characters
This model is still early in its training phase and as a result cannot be trusted. Many characters,
especially ones generated from non-CJK scripts are often completely broken once processed by the model.

To automate away the fixing of broken results we use general-purpose multimodal LLM models for fixing these characters in-place.
Currently, we use Claude Opus 5.5 to analyze general patterns in broken characters, which then turn into automated scripts for some
issues, while everything else that's more complex is analyzed and fixed by Sonnet 5.5 subagents manually. The subagents prompt from
Opus is also inputted into Antigravity, where we use Gemini 4 Hydrogen High to fix characters, though results often require a lot of corrections,
since Gemini often does not produce any scripts to fix common issues.

### Attempt history and staging

Every version of every glyph is kept in `training_data/attempts/` (append-only JSONL per glyph, tracked in git):
the model's prediction, each fix-up script's and LLM's repair, each human edit, with the checker's result and the
reviewer's verdict. The loop:

1. **Process** glyphs in the dashboard. Every prediction is recorded and checked, and glyphs without an approved
   version appear in the **Staging** tab (`training_data/attempts/staging/`, rebuilt with
   `uv run python -m strokegen.attempts stage`).
2. **Save** the good ones: the training file is written as before, and the version is marked approved. **Reject…**
   the wrong ones with a reason (required when the checker passed them); they stay staged.
3. LLM agents run repair passes on what is left: `attempts export` → scripts in `tools/fixups/<run_id>/` or
   per-glyph fixes → `attempts import`. The workflow agents follow is in `CLAUDE.md` ("Repair passes").
4. Review the repaired versions in the Staging tab (filter "Repaired") and Save or reject them again.

`uv run python -m strokegen.attempts stats --by source` shows how many glyphs passed on the first try, after
fix-up scripts, after LLM repair, or are still pending/unresolved; `--by font`, `--by script` and `--by fixup`
break it down. Start the dashboard with `--no-attempts` to leave the store alone.

### Iterative refinement (experimental, off by default)

The store's history trains a Stage A refiner: given a decomposition and the checker's feedback on it, it predicts a
corrected one. It starts as an exact copy of the Stage A model (its new inputs are zero-initialised), so it can only
learn to do better than the single pass. On a CUDA machine:

```sh
uv run python -m strokegen.refine_dataset [--synthetic 2]      # pairs from the attempt store
uv run python -m strokegen.train_refine                        # about 1 h on an RTX 2070 Super -> checkpoints/refine.pt
uv run python -m strokegen.eval_refine --k 3 [--holdout-font F]  # report: outputs/refine_eval_*.md
```

`--refine-iters K` on `strokegen.predict` and `tools/dashboard.py` turns on the loop: a glyph the checker fails is
refined, decoded and checked again, up to K times. Every round is stored in the attempt store (source `refiner`).
