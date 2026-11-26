"""Backfill the store from the training-data submodule's git history (opt-in, `attempts backfill-hwdata`).

Every committed version of data/{char}.json becomes an `imported` attempt (source_ref: repo, commit,
commit date), parented on the previous one. The newest version (the working copy if it differs from
the last commit) is approved, the older ones superseded. These are approved versions only: the model
outputs and repairs that led to them were never kept, so none are invented here.

The font is chosen as build_dataset does (manifest fonts, best vertex match of the newest version), so
it needs the glyph SVGs (training_data/scripts/export_glyphs.py). Characters without one are skipped.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections import Counter

import numpy as np

from ..build_dataset import MIN_VERTEX_MATCH, _vertex_match_frac
from ..fonts import font_rank
from ..paths import GLYPHS, HWDATA
from ..svg_io import load_svg_glyph, parse_hw_path
from .labels import content_hash

REPO_NAME = "hanzi-writer-data-youyin"


def _versions(repo: str, chars: set[str] | None) -> dict[str, list[tuple[str, str]]]:
    """char -> [(commit, committer date)] oldest first, for data/{char}.json (plain files only)."""
    out = subprocess.run(["git", "-c", "core.quotepath=off", "log", "--reverse", "--format=>%H %cI", "--name-only",
                          "--diff-filter=AMR", "--", "data"], cwd=repo, capture_output=True, text=True, check=True)
    res: dict[str, list[tuple[str, str]]] = {}
    commit = date = None
    for line in out.stdout.splitlines():
        if line.startswith(">"):
            commit, date = line[1:].split(" ", 1)
        elif line.startswith("data/") and line.endswith(".json"):
            c = line[5:-5]
            if c.endswith(("-jp", "-ko")) or (chars and c not in chars):
                continue
            res.setdefault(c, []).append((commit, date))
    return res


class _Blobs:
    """git cat-file --batch reader."""

    def __init__(self, repo: str):
        self.p = subprocess.Popen(["git", "cat-file", "--batch"], cwd=repo, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE)

    def read(self, rev: str) -> bytes | None:
        self.p.stdin.write(rev.encode() + b"\n")
        self.p.stdin.flush()
        head = self.p.stdout.readline().split()
        if len(head) < 3 or head[1] == b"missing":
            return None
        data = self.p.stdout.read(int(head[2]))
        self.p.stdout.read(1)
        return data

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def _best_font(hw: dict, fonts: list[str]):
    pts = np.array([s.p0 for sd in hw["strokes"] for c in parse_hw_path(sd) for s in c])
    best = None
    for f in sorted(fonts, key=font_rank):
        p = os.path.join(GLYPHS, f, f"{hw['_char']}.svg")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            contours = load_svg_glyph(fh.read())
        if not contours or not len(pts):
            continue
        frac = _vertex_match_frac(pts, np.array([s.p0 for c in contours for s in c]))
        if best is None or frac > best[0] + 1e-9:
            best = (frac, f, contours)
    return best


def backfill_hwdata(store, chars: str = "", dry_run: bool = False, hwdata: str = HWDATA) -> Counter:
    repo = os.path.dirname(hwdata)
    with open(os.path.join(GLYPHS, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)["characters"]
    want = set(chars) if chars else None
    versions = _versions(repo, want)
    blobs = _Blobs(repo)
    counts: Counter = Counter()
    try:
        for char in sorted(versions):
            entry = manifest.get(char)
            if not entry or not entry.get("fonts"):
                counts["not_in_manifest"] += 1
                continue
            datas = []
            for commit, date in versions[char]:
                raw = blobs.read(f"{commit}:data/{char}.json")
                try:
                    d = json.loads(raw) if raw else None
                except ValueError:
                    d = None
                if d and d.get("strokes") and len(d["strokes"]) == len(d.get("medians", [])):
                    datas.append((dict(repo=REPO_NAME, commit=commit, committed_at=date), d, date))
            p = os.path.join(hwdata, f"{char}.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    cur = json.load(f)
                if not datas or content_hash(cur) != content_hash(datas[-1][1]):
                    datas.append((dict(repo=REPO_NAME, commit="worktree"), cur, None))
            # identical consecutive versions (a commit that only touched radStrokes, say) are one version
            uniq = [x for i, x in enumerate(datas) if i == 0 or content_hash(x[1]) != content_hash(datas[i - 1][1])]
            if not uniq:
                counts["no_valid_version"] += 1
                continue
            best = _best_font(dict(uniq[-1][1], _char=char), entry["fonts"])
            if best is None:
                counts["no_font_svg"] += 1
                continue
            frac, font, contours = best
            if frac < MIN_VERTEX_MATCH:
                counts["no_exact_font"] += 1
                continue
            if dry_run:
                counts["would_import"] += 1
                counts["versions"] += len(uniq)
                continue
            parent = None
            for ref, d, date in uniq:
                parent, created = store.record(font, char, contours, d, "imported", ref, parent=parent,
                                               created_at=date)
                counts["versions" if created else "already_imported"] += 1
            if store.get(parent)["verdict"] != "approved":
                store.approve(parent, by="backfill", reason="current training file")
            counts["imported"] += 1
    finally:
        blobs.close()
    print(", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return counts
