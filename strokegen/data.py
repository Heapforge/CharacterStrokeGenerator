"""Datasets and batching for the cutting model."""
from __future__ import annotations

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from .decode import fill_unassigned
from .features import augment, glyph_features, targets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")


def load_split(name: str) -> list[dict]:
    with open(os.path.join(PROCESSED, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


class CutDataset(Dataset):
    def __init__(self, recs: list[dict], train: bool, max_segs: int = 400):
        self.recs = [r for r in recs if sum(len(c) for c in r["contours"]) <= max_segs]
        self.train = train

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        contours, lab = r["contours"], fill_unassigned(r["contours"], r["seg_stroke"])
        if self.train:
            rng = np.random.default_rng()
            contours, lab = augment(contours, lab, rng)
        f = glyph_features(contours)
        f.update(targets(contours, lab, f["nxt"]))
        f["index"] = i
        return f


def collate(items: list[dict]) -> dict:
    B = len(items)
    N = max(len(it["feat"]) for it in items)
    F = items[0]["feat"].shape[1]
    P = items[0]["probe"].shape[1]
    out = dict(
        feat=torch.zeros(B, N, F),
        pad=torch.ones(B, N, dtype=torch.bool),
        bucket=torch.full((B, N, N), 10, dtype=torch.int8),
        probe=torch.zeros(B, N, P, 2),
        nxt=torch.zeros(B, N, dtype=torch.long),
        img=torch.from_numpy(np.stack([it["img"] for it in items])),
        index=torch.tensor([it["index"] for it in items]),
    )
    has_t = "lab" in items[0]
    if has_t:
        out.update(
            lab=torch.full((B, N), -1, dtype=torch.long),
            cut=torch.zeros(B, N),
            cut_mask=torch.zeros(B, N),
        )
    for b, it in enumerate(items):
        n = len(it["feat"])
        out["feat"][b, :n] = torch.from_numpy(it["feat"])
        out["pad"][b, :n] = False
        out["bucket"][b, :n, :n] = torch.from_numpy(it["bucket"])
        out["probe"][b, :n] = torch.from_numpy(it["probe"])
        out["nxt"][b, :n] = torch.from_numpy(it["nxt"])
        out["nxt"][b, n:] = torch.arange(n, N)
        if has_t:
            out["lab"][b, :n] = torch.from_numpy(it["lab"])
            out["cut"][b, :n] = torch.from_numpy(it["cut"])
            out["cut_mask"][b, :n] = torch.from_numpy(it["cut_mask"])
    return out


def to_device(batch: dict, dev) -> dict:
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
