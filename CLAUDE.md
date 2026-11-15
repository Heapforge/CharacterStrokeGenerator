# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`strokegen` turns a CJK glyph outline SVG (Arphic Kai fonts, exported by FontForge) into
[hanzi-writer](https://hanziwriter.org) data: `{"strokes": [svg path per stroke], "medians": [[x, y], ...] per stroke}`.
README.md has the current metrics and the review dashboard's user guide.

## Commands

Python 3.12 only, managed with `uv`. Training scripts hard-require CUDA; `predict`/`evaluate`/dashboard fall back to CPU.

```sh
uv sync
uv run python -m strokegen.build_dataset [--limit N]   # -> data/processed/{train,val,test}.pkl
uv run python -m strokegen.train_cut [--epochs 60]     # Stage A -> checkpoints/cut.pt
uv run python -m strokegen.train_order [--epochs 80]   # Stage C -> checkpoints/order.pt
uv run python -m strokegen.evaluate [--chars 永林]      # metrics + outputs/preview*.html
uv run python -m strokegen.predict glyph.svg -o out.json   # also accepts a directory
uv run python tools/dashboard.py [--device cpu]        # http://127.0.0.1:8417
uv run --group dev pytest tests
uv run --group dev pytest tests/test_pipeline.py::test_cross_split_into_two_strokes   # single test
uv run --group dev ruff check .                        # lint (config in pyproject.toml)
```

The data-dependent tests are skipped unless `data/processed/val.pkl` exists; the end-to-end prediction tests
need checkpoints trained on the current feature version; the dashboard unpack test needs FontForge.
`data/`, `checkpoints/`, `outputs/` and all `*.svg` files are gitignored, so the glyph SVGs under
`training_data/font_glyphs/` are local only.

**Feature versions.** `features.FEATURE_VERSION` (Stage A) and `order.FEATURE_VERSION` (Stage C) are saved in
checkpoints as `args["feat_version"]`; `StrokeGenerator` refuses a checkpoint with a different version. Bump the
constant whenever a change alters model inputs, then retrain. `train_order` caches Stage B output in
`data/processed/order_{split}.pkl` keyed on that version and the split file's mtime, so it rebuilds itself after
`build_dataset` or a feature change (other changes to `medians.py` still need the cache deleted by hand).

## Architecture

Three-stage pipeline, orchestrated by `StrokeGenerator` in `strokegen/predict.py` (also used by `evaluate` and the dashboard):

1. **Stage A: cut (learned).** `features.glyph_features` → `model.CutModel` (transformer over outline
   segments + small raster CNN) predicts per-vertex cut probabilities and pairwise same-stroke affinities.
   `decode.partition_from_predictions` splits contours into chains at cuts, clusters chains into strokes,
   and validates bridges against the glyph mask (`bridge_validator`: open chains are joined by straight
   test bridges; a chain that is a whole contour gets no bridge but must touch the rest of the stroke
   through ink, so a counter can join its outline but separate parts cannot merge across the background).
   `absorb_counters` then hands any stroke made only of whole counters (contours whose inside the glyph
   leaves empty) to a stroke owning the enclosing contour. Result: `seg_stroke`, a stroke label per outline segment.
2. **Assembly (geometry).** `decode.assemble_strokes` closes each stroke by joining its chains with bridge
   segments (straight, or tangent-continuous cubics with `C_RATIO`). A cubic that bulges out of the glyph
   is straightened: for a cut shared by two strokes both take the chord, otherwise only when the bulge lies
   mostly outside. **Invariant: outline segments are never modified**, so the union of strokes renders
   exactly the original glyph (checked by IoU in eval/tests).
3. **Stage B: medians (no learning).** `medians.stroke_median`: Voronoi skeleton → longest path → tip
   extension/spur trimming → RDP. Output is unoriented (`order.canonical`). Stroke shapes come from
   `medians.stroke_geometry`, which fills a stroke's loops with nonzero winding like hanzi-writer, so counters
   stay holes.
4. **Stage C: order + direction (learned).** `order.stroke_features`/`pair_features` →
   `model.OrderNet` (wraps `model.OrderModel`) predicts per-stroke flip and pairwise precedence;
   `order.decode_order` turns precedence into a sequence.
   It is trained on medians computed from GT stroke outlines so training input matches inference.

Single-glyph prediction goes through the same batched path as bulk prediction (`_predict_batch`); the CPU work
lives in `strokegen/_batch.py` (spawned workers, which report per-glyph errors as `_error` instead of raising).

**Coordinate frame.** All internal geometry is in hanzi-writer space ("hw"): 1024-unit em, y up,
`y_hw = 900 − y_svg` (`svg_io.HW_BASE`). `svg_io.Seg` (kind `L`/`Q`/`C` + control points) is the core
geometry type; contours are `list[Seg]`. Model normalization constants are `svg_io.NORM_CX/NORM_CY/NORM_SC`.
Repository paths are in `strokegen/paths.py`; the exported font list (and its priority) is `strokegen/fonts.py`.

**Training data (`build_dataset.py`).** Labels come from the `training_data/hanzi-writer-data-youyin`
git submodule (`data/{char}.json`; only plain files, `-jp`/`-ko` variants are ignored). For each character,
`font_glyphs/manifest.json` lists which exported fonts have the glyph; fonts are tried in `FONT_PRIORITY`
order and the best match is kept. `decompose` then assigns each font segment to the hw stroke that uses
it (same endpoints ±1 and same shape) and records the bridges; unmatched segments stay -1 and are masked
out of both training losses. A record is kept only if ≥`MIN_VERTEX_MATCH` (0.7) of hw vertices are font vertices,
≥`MIN_COVERAGE` (0.6) of font segments are matched, and every bridge end lies on the font outline. The
gates are loose so hand-corrected strokes that cut partway along a font edge pass; `rec["exact"]` marks
records whose GT is an exact segment partition. The split is a function of the outline hash alone
(`split_of`, 90/5/5 buckets): identical outlines always share a split, and adding characters never moves
other glyphs between train and test.
`training_data/scripts/export_glyphs.py` (FontForge, `python -I`) and `manifest.py` regenerate the glyph
SVGs and manifest.

**Dashboard (`tools/dashboard.py` + `dashboard.html`).** Stdlib HTTP server that runs `StrokeGenerator`,
supports stroke merge/split via `predict.merge_strokes`, and on Save writes into the submodule's `data/`
and registers the char in `manifest.json`. That is why those two paths show up as modified in `git status`.
It has no authentication: it only answers Host headers of its loopback address and refuses cross-origin or
non-JSON POSTs. Fonts without exported SVGs are unpacked in a background thread (`/api/glyphs` answers 202).

`tools/` holds the maintained correction tools; `tools/scratch/` holds one-off scripts from the manual
correction passes (several overwrite training data when run; see its README).

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- graphify-out/ is gitignored here, so the graph is local to each checkout; rebuild it rather than expecting it in a pull request.
