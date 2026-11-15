"""Repository paths shared by the pipeline, the training scripts and the tools."""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
CKPT = os.path.join(ROOT, "checkpoints")
TRAINING_DATA = os.path.join(ROOT, "training_data")
GLYPHS = os.path.join(TRAINING_DATA, "font_glyphs")
FONTS_DIR = os.path.join(TRAINING_DATA, "fonts")
HWDATA = os.path.join(TRAINING_DATA, "hanzi-writer-data-youyin", "data")
OUTPUTS = os.path.join(ROOT, "outputs")
