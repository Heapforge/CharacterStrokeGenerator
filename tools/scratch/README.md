# One-off scripts

Scripts from the manual correction passes over the dashboard's first ~100 Latin/Cyrillic glyphs
(noto-sans). They are kept for reference and are not maintained:

- Many read exports that no longer exist (`/tmp/queue100.json`, `/tmp/queue_done.json`) or post to
  a dashboard on port 9000.
- Several **overwrite files in `training_data/hanzi-writer-data-youyin/data/`** when run
  (`fix_svgs.py`, `apply_*.py`, `build_100_corrections.py`). Read one before running it.
- `test_*.py` here are scratch checks, not pytest tests (`pyproject.toml` limits pytest to `tests/`).

Run them from the repository root (`uv run python tools/scratch/<name>.py`); most of them put `.` on
`sys.path` to import `strokegen`. The maintained tools are in `tools/`.
