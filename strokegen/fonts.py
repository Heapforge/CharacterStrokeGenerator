"""The exported fonts: one list shared by build_dataset, the manifest script and the dashboard.

The order is the priority build_dataset uses when several fonts have a glyph for a character.
Each entry is (name of the training_data/font_glyphs/<name> directory, font file relative to
training_data/fonts, face name inside a .ttc collection or None).
"""
from __future__ import annotations

import os

from .paths import FONTS_DIR

FONTS: list[tuple[str, str, str | None]] = [
    ("gkai00mp", "arphic/gkai00mp.ttf", None),
    ("bkai00mp", "arphic/bkai00mp.ttf", None),
    ("ukai-cn", "arphic/ukai.ttc", "AR PL UKai CN"),
    ("ukai-hk", "arphic/ukai.ttc", "AR PL UKai HK"),
    ("ukai-tw", "arphic/ukai.ttc", "AR PL UKai TW"),
    ("ukai-tw-mbe", "arphic/ukai.ttc", "AR PL UKai TW MBE"),
    ("noto-sans", "noto/NotoSans-Regular.ttf", None),
    ("ubuntu", "ubuntu/Ubuntu-R.ttf", None),
    ("liberation-serif", "liberation/LiberationSerif-Regular.ttf", None),
    ("merriweather", "merriweather/Merriweather-Regular.ttf", None),
    ("noto-sans-cjk-sc", "noto-cjk/NotoSansCJK-Regular.ttc", "Noto Sans CJK SC"),
    ("noto-sans-cjk-hk", "noto-cjk/NotoSansCJK-Regular.ttc", "Noto Sans CJK HK"),
    ("noto-sans-cjk-tc", "noto-cjk/NotoSansCJK-Regular.ttc", "Noto Sans CJK TC"),
    ("noto-sans-cjk-jp", "noto-cjk/NotoSansCJK-Regular.ttc", "Noto Sans CJK JP"),
    ("noto-sans-cjk-kr", "noto-cjk/NotoSansCJK-Regular.ttc", "Noto Sans CJK KR"),
    ("noto-serif-cjk-sc", "noto-cjk/NotoSerifCJK-Regular.ttc", "Noto Serif CJK SC"),
    ("noto-serif-cjk-hk", "noto-cjk/NotoSerifCJK-Regular.ttc", "Noto Serif CJK HK"),
    ("noto-serif-cjk-tc", "noto-cjk/NotoSerifCJK-Regular.ttc", "Noto Serif CJK TC"),
    ("noto-serif-cjk-jp", "noto-cjk/NotoSerifCJK-Regular.ttc", "Noto Serif CJK JP"),
    ("noto-serif-cjk-kr", "noto-cjk/NotoSerifCJK-Regular.ttc", "Noto Serif CJK KR"),
]

FONT_PRIORITY: list[str] = [name for name, _, _ in FONTS]
_RANK = {name: i for i, name in enumerate(FONT_PRIORITY)}


def font_rank(name: str) -> int:
    """Priority of a font directory; fonts missing from FONTS sort last instead of failing."""
    return _RANK.get(name, len(FONT_PRIORITY))


def font_spec(name: str) -> str | None:
    """FontForge open() spec for a font: 'path' or 'path(face name)' for collections."""
    for n, rel, face in FONTS:
        if n == name:
            path = os.path.join(FONTS_DIR, rel)
            return f"{path}({face})" if face else path
    return None
