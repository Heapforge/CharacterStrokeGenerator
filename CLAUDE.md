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
uv run python tools/dashboard.py [--device cpu]        # http://127.0.0.1:8417  (--no-attempts: no store)
uv run python -m strokegen.attempts --help             # attempt store: stage/export/import/history/stats
uv run python -m strokegen.refine_dataset              # refiner pairs from the store -> data/processed/refine_*
uv run python -m strokegen.train_refine [--budget-min 55]   # Stage A refiner -> checkpoints/refine.pt (CUDA)
uv run python -m strokegen.eval_refine [--k 3] [--holdout-font F]   # single pass vs refinement -> outputs/refine_eval_*
uv run python -m strokegen.predict glyph.svg --refine-iters 3        # refinement loop (also tools/dashboard.py)
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
Unless started with `--no-attempts` it also records every prediction in the attempt store (a background
`Recorder` thread maps it to segment labels and runs the checker), lists not-yet-approved glyphs in its
**Staging** tab, records **Reject…** verdicts, and on Save approves the saved version (as a `human_edit` when no
stored version has that content). The training file is written first and exactly as without the store; a store
error only adds to the Save note.
It has no authentication: it only answers Host headers of its loopback address and refuses cross-origin or
non-JSON POSTs. Fonts without exported SVGs are unpacked in a background thread (`/api/glyphs` answers 202).

**Checker.** `tools/verify_glyph.check_glyph(font_contours, data)` is the single pass/fail source: union
IoU ≥ 0.99, every median point inside its stroke, median coverage ≥ 0.9, and (`reveal=True`) no stroke area
hanzi-writer leaves undrawn (`tools/reveal_check.py`). Don't reimplement any of it; call it
(`strokegen.attempts.check.run_check` wraps it and adds the segments next to the IoU deficit).

**Attempt store (`strokegen/attempts/`, data in `training_data/attempts/`).** Every version of every glyph
(model, fix-up script, LLM repair, human edit, refiner, synthetic, imported) as append-only JSONL events,
one file per glyph: `log/<font>/<U+XXXX>.jsonl`, tracked in git (the folder is self-contained so it can become
a submodule). Versions are stored in segment-label space (`labels.labels_from_hw` reuses `build_dataset.decompose`)
next to the hanzi-writer JSON, the glyph's exact outline, checker results and verdicts. `index.sqlite` and
`staging/` are derived and gitignored (`attempts reindex` / `attempts stage` rebuild them, e.g. after a pull).
A glyph is **staged** while it has no approved version; `staging/<font>/<U+XXXX>.json` holds the version to work
from plus its history. The format is in `training_data/attempts/README.md`.
`strokegen/refine_dataset.py` turns the store into refiner training pairs (every earlier version of an approved
glyph → its approved version), split by the same outline hash as `build_dataset`.

**Refiner (`model.RefineCutModel`, `strokegen/refine.py`, `train_refine.py`).** A CutModel that also gets a
current decomposition: its labels as a per-layer same-stroke attention bias plus output residuals, and per-segment
feedback (`refine.REFINE_FEATURES`: `check.SEG_FLAGS`, i.e. the checker's failing stroke / IoU deficit / median /
hidden flags, unassigned, `in_rejected_merge` from the decoder's bridge validator (`partition_from_predictions(...,
report=)`, stored as `labels.vetoed`), and where the labels cut). The new parameters start at zero, so an untrained
refiner is exactly its base checkpoint (tested). Training warm-starts from `cut.pt` with Stage A's losses, weighted
by the pairs' recency weights. Bump `refine.REFINE_VERSION` when the refiner's inputs change.
`StrokeGenerator(refine_iters=K)` (default 0 = single pass; `--refine-iters` on `predict` and the dashboard) runs
predict → decode → Stage B/C → check → refine until the checker passes, the partition stops changing, it returns
to an earlier partition, or K rounds; the result is the last round, all rounds are in `_rounds`, and the dashboard
stores round 0 as `model` and later rounds as `refiner`. `eval_refine` scores K = 0..K from one such run.

`tools/` holds the maintained correction tools; `tools/fixups/<run_id>/` holds the scripts of recorded repair
passes; `tools/scratch/` holds one-off scripts from the manual correction passes before the store existed
(several overwrite training data when run; see its README).

## Repair passes (LLM agents fixing staged glyphs)

Glyphs the model gets wrong are fixed in passes by LLM agents, then reviewed by a human in the dashboard. Every
attempt, good or bad, is training data for the refiner, so **record all of them and keep them reproducible**:

1. **Start a run.** Pick a run id `YYYYMMDD-<model>-<topic>` (e.g. `20261126-sonnet-latin-counters`) and create
   `tools/fixups/<run_id>/NOTES.md`: the goal, the glyphs targeted, and as you go what you tried, what failed and
   why, and the result of each import (the counts `attempts import` prints). Commit the notes with the scripts.
2. **Get the work.** `uv run python -m strokegen.attempts stage` (after a pull), then
   `uv run python -m strokegen.attempts export --font F [--failing] [--format list] -o <scratch>/queue.json`.
   The queue has each glyph's staged version (`attempt_id`, checker result, `segStroke`); the glyph's history,
   including rejection reasons from the reviewer, is in `training_data/attempts/staging/F/<U+XXXX>.json`.
   `attempts history CHAR --font F` prints it too. Read rejection reasons before you start: a version the checker
   passed but the reviewer rejected is wrong in a way the checker can't see.
3. **Fix with scripts where a pattern repeats.** Put each script in `tools/fixups/<run_id>/`, reusing
   `tools/*.py` helpers (`split_vertical`, `median_reroute.hw_safe`, `fix_hole_strokes`, ...). Outline segments
   are never modified: strokes must be cut from the font outline (see the assembly invariant). **Never edit a
   script after it produced imported attempts**: copy it to `<name>_v2.py` and change the copy, so every
   attempt's `source_ref` (script path + git blob sha) still points at the code that made it. Commit a script
   before importing its output (the import warns when it is dirty).
4. **Record every output.** Script output:
   `attempts import out.json --font F --source fixup_script --script tools/fixups/<run_id>/x.py [--args "..."] --tags <error tags>`.
   Glyph-by-glyph repairs: `attempts import out.json --font F --source llm_repair --llm-model <model id> --run <run_id>`.
   `import` takes a JSON map `char -> {strokes, medians}`, a list of `{char, strokes, medians}`, or a directory
   of `{char}.json`; it parents each version on the staged one, runs the checker and prints how many passed.
   Import failed attempts as well, and retry with a new version instead of overwriting anything.
5. **Check your work** with `attempts history CHAR --font F` and `attempts stats --by fixup` (or `--by source`).
   Use the checker (`tools/verify_glyph.py`, `tools/render_compare.py`) before importing; a reviewer still
   decides.
6. **Give up explicitly.** When a glyph can't be fixed in this pass:
   `attempts unresolved CHAR --font F --reason "..."`, so later passes see why.

Never write to `training_data/hanzi-writer-data-youyin/data/` (only the dashboard's Save does, after a human looked),
never approve from a script, and never edit or delete lines in `training_data/attempts/log/`: the log is
append-only, and git keeps both sides of a merge conflict in it.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- graphify-out/ is gitignored here, so the graph is local to each checkout; rebuild it rather than expecting it in a pull request.
