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
```

`test_gt_partition_roundtrip` is skipped unless `data/processed/val.pkl` exists. `data/`, `checkpoints/`,
`outputs/` and all `*.svg` files are gitignored, so the glyph SVGs under `training_data/font_glyphs/` are local only.

**Stale cache:** `train_order` caches Stage B output as `data/processed/order_{split}.pkl` and reuses it if
present. Delete those files after re-running `build_dataset` or changing `medians.py`/`order.py` features.

## Architecture

Three-stage pipeline, orchestrated by `StrokeGenerator` in `strokegen/predict.py` (also used by `evaluate` and the dashboard):

1. **Stage A: cut (learned).** `features.glyph_features` → `model.CutModel` (transformer over outline
   segments + small raster CNN) predicts per-vertex cut probabilities and pairwise same-stroke affinities.
   `decode.partition_from_predictions` splits contours into chains at cuts, clusters chains into strokes,
   and validates bridges against the glyph mask. Result: `seg_stroke`, a stroke label per outline segment.
2. **Assembly (geometry).** `decode.assemble_strokes` closes each stroke by joining its chains with bridge
   segments (straight, or tangent-continuous cubics with `C_RATIO`). **Invariant: outline segments are never
   modified**, so the union of strokes renders exactly the original glyph (checked by IoU in eval/tests).
3. **Stage B: medians (no learning).** `medians.stroke_median`: Voronoi skeleton → longest path → tip
   extension/spur trimming → RDP. Output is unoriented (`order.canonical`).
4. **Stage C: order + direction (learned).** `order.stroke_features`/`pair_features` →
   `train_order.OrderNet` (wraps `model.OrderModel`; note it lives in `train_order.py`, not `model.py`)
   predicts per-stroke flip and pairwise precedence; `order.decode_order` turns precedence into a sequence.
   It is trained on medians computed from GT stroke outlines so training input matches inference.

**Coordinate frame.** All internal geometry is in hanzi-writer space ("hw"): 1024-unit em, y up,
`y_hw = 900 − y_svg` (`svg_io.HW_BASE`). `svg_io.Seg` (kind `L`/`Q`/`C` + control points) is the core
geometry type; contours are `list[Seg]`. Normalization constants for models are in `features.py`/`order.py`.

**Training data (`build_dataset.py`).** Labels come from the `training_data/hanzi-writer-data-youyin`
git submodule (`data/{char}.json`; only plain files, `-jp`/`-ko` variants are ignored). For each character,
`font_glyphs/manifest.json` lists which exported fonts have the glyph; fonts are tried in `FONT_PRIORITY`
order and the best match is kept. `decompose` then assigns each font segment to the hw stroke that uses
it (same endpoints ±1 and same shape) and records the bridges; unmatched segments stay -1 and are masked
out of the cut loss. A record is kept only if ≥`MIN_VERTEX_MATCH` (0.7) of hw vertices are font vertices,
≥`MIN_COVERAGE` (0.6) of font segments are matched, and every bridge end lies on the font outline. The
gates are loose so hand-corrected strokes that cut partway along a font edge pass; `rec["exact"]` marks
records whose GT is an exact segment partition. Records with identical outlines (hash) are all kept but
always land in the same split; split is 90/5/5 by outline group with seed 0.
`training_data/scripts/export_glyphs.py` (FontForge, `python -I`) and `manifest.py` regenerate the glyph
SVGs and manifest.

**Dashboard (`tools/dashboard.py` + `dashboard.html`).** Stdlib HTTP server that runs `StrokeGenerator`,
supports stroke merge/split via `predict.merge_strokes`, and on Save writes into the submodule's `data/`
and registers the char in `manifest.json`. That is why those two paths show up as modified in `git status`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
