"""Stage A refiner inputs: a glyph's current decomposition plus the checker's feedback on it.

Training (train_refine, on the refine_dataset pairs) and inference (StrokeGenerator(refine_iters=K)) both
build the refiner's per-segment input with input_features, so they always agree.

Per segment the refiner sees (REFINE_FEATURES):
  - strokegen.attempts.check.SEG_FLAGS: in a failing stroke, near the IoU deficit, median not covering the
    stroke, stroke left partly undrawn, unassigned, in a merge the decoder's bridge validator refused;
  - "cut_after": the current labels cut between this segment and the next one on its contour.
The current assignment itself (batch["in_lab"]) becomes a same-stroke attention bias in model.RefineCutModel.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .attempts.check import SEG_FLAGS, segment_flags
from .data import collate, worker_rng
from .features import augment, glyph_features, targets

# Bump when the meaning or number of the refiner's inputs changes; refine checkpoints record it.
REFINE_VERSION = 1
REFINE_FEATURES = (*SEG_FLAGS, "cut_after")
N_RFEAT = len(REFINE_FEATURES)


def input_features(in_lab: np.ndarray, flags: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """(N, N_RFEAT) float32: checker/decoder flags (N, len(SEG_FLAGS)) plus where the current labels cut."""
    lab = np.asarray(in_lab, np.int64)
    cut = (lab != lab[nxt]) | (lab < 0)
    return np.concatenate([np.asarray(flags, np.float32), cut[:, None].astype(np.float32)], 1)


def refine_item(feat: dict, in_lab, checker: dict | None, vetoed=None) -> dict:
    """glyph_features output + the refiner inputs for one current decomposition."""
    lab = np.asarray(in_lab, np.int64)
    flags = segment_flags(lab, checker, vetoed)
    return dict(feat, in_lab=lab, rfeat=input_features(lab, flags, feat["nxt"]))


def collate_refine(items: list[dict]) -> dict:
    """data.collate plus in_lab (B,N; -1 padding), rfeat (B,N,N_RFEAT) and weight (B,)."""
    out = collate(items)
    B, N = out["pad"].shape
    out["in_lab"] = torch.full((B, N), -1, dtype=torch.long)
    out["rfeat"] = torch.zeros(B, N, N_RFEAT)
    for b, it in enumerate(items):
        n = len(it["feat"])
        out["in_lab"][b, :n] = torch.from_numpy(np.asarray(it["in_lab"], np.int64))
        out["rfeat"][b, :n] = torch.from_numpy(np.asarray(it["rfeat"], np.float32))
    out["weight"] = torch.tensor([float(it.get("weight", 1.0)) for it in items])
    return out


class RefineDataset(Dataset):
    """refine_dataset records -> refiner training items (targets as in Stage A, cut BCE + affinity).

    Augmentation is Stage A's (features.augment): it is run on segment indices, so the target labels, the
    input labels and the flags all follow the same subdivision and contour rotation.
    """

    def __init__(self, recs: list[dict], train: bool, max_segs: int = 400, use_weights: bool = True):
        self.recs = [r for r in recs if sum(len(c) for c in r["contours"]) <= max_segs]
        self.train = train
        self.use_weights = use_weights
        self._rng: dict = {}

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        contours = r["contours"]
        ids = np.arange(sum(len(c) for c in contours))
        if self.train:
            contours, ids = augment(contours, ids, worker_rng(self._rng))
        lab = np.asarray(r["seg_stroke"], np.int64)[ids]
        in_lab = np.asarray(r["input_seg_stroke"], np.int64)[ids]
        flags = np.asarray(r["seg_flags"], np.float32)[ids]
        f = glyph_features(contours)
        f.update(targets(contours, lab, f["nxt"]))
        f.update(in_lab=in_lab, rfeat=input_features(in_lab, flags, f["nxt"]), index=i,
                 weight=float(r.get("weight", 1.0)) if self.use_weights else 1.0)
        return f
