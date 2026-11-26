"""Attempt history: every version of every glyph, who produced it, what the checker said, and its verdict.

See training_data/attempts/README.md for the on-disk format and CLAUDE.md ("Repair passes") for the
workflow. CLI: python -m strokegen.attempts --help
"""
from .store import SOURCE_TYPES, VERDICTS, AttemptStore  # noqa: F401
