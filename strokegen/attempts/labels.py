"""Versions of a glyph in segment-label space, so any two of them can be compared directly.

Every version (model output, fix-up script, LLM repair, human edit) is stored as labels over the
font's own outline segments: which stroke each segment belongs to (strokes numbered in writing
order), the bridges that close each stroke, the oriented medians, and whether each median runs
against the canonical direction (order.canonical). Versions that are only hanzi-writer JSON are
mapped onto the font with build_dataset.decompose, the same matcher that turns the training data
into training records.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata

import numpy as np

from ..build_dataset import decompose, outline_hash
from ..order import canonical


def glyph_key(contours) -> str:
    """The outline hash build_dataset splits on: every version of a glyph shares it."""
    return outline_hash(contours)


def script_of(char: str) -> str:
    """cjk | latin | greek | cyrillic | other, from the first letter-like code point's Unicode name."""
    for ch in char:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        for prefix, script in (("CJK", "cjk"), ("KANGXI", "cjk"), ("HIRAGANA", "cjk"), ("KATAKANA", "cjk"),
                               ("HANGUL", "cjk"), ("BOPOMOFO", "cjk"), ("IDEOGRAPHIC", "cjk"),
                               ("LATIN", "latin"), ("GREEK", "greek"), ("COPTIC", "greek"),
                               ("CYRILLIC", "cyrillic")):
            if name.startswith(prefix) or f" {prefix} " in f" {name} ":
                return script
    return "other"


def lowercase_a_variant(char: str) -> bool:
    """a and its accented / variant forms (á, ä, ɑ, а ...), reported separately in evaluation."""
    try:
        name = unicodedata.name(char[0]) if char else ""
    except ValueError:
        return False
    base = unicodedata.normalize("NFD", char)[:1]
    return base in ("a", "а", "ɑ", "α") or name.startswith(("LATIN SMALL LETTER A ", "LATIN SMALL LETTER ALPHA"))


def _flip(medians) -> list[bool]:
    out = []
    for m in medians:
        m = np.asarray(m, float)
        out.append(bool(len(m) >= 2 and canonical(m) is not m))
    return out


def labels_from_hw(contours, hw: dict, seg_stroke=None, vetoed=None) -> dict:
    """Labels for hanzi-writer data cut from `contours`.

    seg_stroke: the producer's own segment labels when it has them (model output: exact); otherwise
    they come from decompose, and unmatched segments are -1. Strokes are in writing order, so the
    order is the identity over stroke indices. vetoed: segments of merges the decoder's bridge
    validator refused (model / refiner output only; None when no decoder was involved).
    """
    dec = decompose(contours, hw)
    lab = dec["seg_stroke"] if seg_stroke is None else np.asarray(seg_stroke, np.int64)
    bridges = [[[b[1].kind, np.round(b[1].pts, 2).tolist(), int(b[2]), int(b[3])] for b in loop if b[0] == "B"]
               for loop in dec["loops"]]
    medians = [[[int(round(x)), int(round(y))] for x, y in m] for m in hw["medians"]]
    return dict(
        seg_stroke=[int(v) for v in lab],
        n_strokes=len(hw["strokes"]),
        order=list(range(len(hw["strokes"]))),
        flip=_flip(medians),
        medians=medians,
        bridges=bridges,
        vetoed=None if vetoed is None else sorted(int(v) for v in vetoed),
        match=dict(coverage=round(float(dec["coverage"]), 4), n_bridge_bad=int(dec["n_bridge_bad"]),
                   n_bridge_on_edge=int(dec["n_bridge_on_edge"]),
                   unassigned=int((np.asarray(lab) < 0).sum())),
    )


def content_hash(hw: dict) -> str:
    """Identity of a version's content: its hanzi-writer strokes and medians."""
    s = json.dumps({"strokes": hw["strokes"], "medians": hw["medians"]}, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(s.encode()).hexdigest()


def same_partition(a, b) -> bool:
    """Whether two segment labellings group the segments the same way (stroke ids may differ)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))
