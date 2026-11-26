"""Neural models.

CutModel (Stage A): transformer over outline segments, predicts cut points and pairwise
same-stroke affinities.
OrderModel (Stage C): transformer over strokes, predicts direction and pairwise precedence.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import N_BUCKETS, N_FEAT
from .order import N_PFEAT, N_SFEAT

N_FOURIER = 16
FOURIER_COORDS = 6  # segment midpoint, start and end (x, y each)


def fourier(x: torch.Tensor, n: int = 16) -> torch.Tensor:
    freqs = (2.0 ** torch.arange(n, device=x.device, dtype=x.dtype)) * math.pi * 0.25
    a = x[..., None] * freqs
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class BiasedLayer(nn.Module):
    """Pre-norm transformer layer with an additive per-head attention bias."""

    def __init__(self, d: int, heads: int, drop: float = 0.1):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, bias):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))


class RasterCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.GELU())

        self.s1 = nn.Sequential(block(1, 32, 2), block(32, 32, 1))  # 128
        self.s2 = nn.Sequential(block(32, 64, 2), block(64, 64, 1))  # 64
        self.s3 = nn.Sequential(block(64, 128, 2), block(128, 128, 1))  # 32
        self.s4 = nn.Sequential(block(128, 128, 2), block(128, 128, 1))  # 16
        self.out_dim = 64 + 128 + 128

    def forward(self, img, probe):
        """img (B,1,H,W); probe (B,N,P,2) in [-1,1] -> (B,N,P*out_dim)"""
        f2 = self.s2(self.s1(img))
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        B, N, P, _ = probe.shape
        g = probe.view(B, N * P, 1, 2)
        feats = [F.grid_sample(f, g, align_corners=False).squeeze(-1) for f in (f2, f3, f4)]
        s = torch.cat(feats, 1)  # (B, C, N*P)
        return s.transpose(1, 2).reshape(B, N, P * self.out_dim)


class CutModel(nn.Module):
    def __init__(self, d: int = 256, layers: int = 8, heads: int = 8, aff_dim: int = 128, n_probe: int = 4):
        super().__init__()
        self.heads = heads
        self.cnn = RasterCNN()
        self.inp = nn.Sequential(
            nn.Linear(N_FEAT + FOURIER_COORDS * N_FOURIER * 2 + n_probe * self.cnn.out_dim, d), nn.GELU(),
            nn.Linear(d, d)
        )
        self.bias = nn.Embedding(N_BUCKETS, heads)
        self.layers = nn.ModuleList([BiasedLayer(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.cut = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.aff = nn.Linear(d, aff_dim)
        self.aff_b = nn.Parameter(torch.zeros(1))
        self.aff_dim = aff_dim

    def embed(self, batch):
        """Per-segment input tokens (B,N,d)."""
        feat = batch["feat"]
        xy = feat[..., 8:10]  # segment midpoint
        se = torch.cat([feat[..., 0:2], feat[..., 6:8]], -1)  # start, end
        cnnf = self.cnn(batch["img"], batch["probe"])
        return self.inp(torch.cat([feat, fourier(torch.cat([xy, se], -1), N_FOURIER), cnnf], -1))

    def attn_bias(self, batch):
        """Relative-position attention bias (B,H,N,N), -inf on padding keys."""
        bias = self.bias(batch["bucket"].long()).permute(0, 3, 1, 2)
        return bias.masked_fill(batch["pad"][:, None, None, :], float("-inf"))

    def outputs(self, x, batch):
        h = self.norm(x)
        hn = torch.gather(h, 1, batch["nxt"][..., None].expand(-1, -1, h.shape[-1]))
        cut = self.cut(torch.cat([h, hn], -1)).squeeze(-1)
        u = self.aff(h)
        aff = u @ u.transpose(1, 2) / math.sqrt(self.aff_dim) + self.aff_b
        return dict(cut=cut, aff=aff, h=h)

    def forward(self, batch):
        x = self.embed(batch)
        bias = self.attn_bias(batch)
        for layer in self.layers:
            x = layer(x, bias)
        return self.outputs(x, batch)


class RefineCutModel(CutModel):
    """Stage A refiner: CutModel that also sees a current decomposition and the checker's feedback on it.

    batch["in_lab"] (B,N): current stroke per segment (-1 unassigned or padding). It enters as a per-layer,
    per-head attention bias between segments of the same current stroke (stroke ids are arbitrary, so no
    id embeddings) and as residuals on the outputs: +r_cut where the current labels cut, +r_aff between
    same / -r_aff between different current strokes.
    batch["rfeat"] (B,N,n_rfeat): per-segment feedback (strokegen.refine.input_features), added to the tokens
    through a linear map.
    All new parameters start at zero, so an untrained refiner computes exactly what its base CutModel does.
    """

    NEW = ("r_inp.weight", "r_inp.bias", "r_same", "r_cut", "r_aff")

    def __init__(self, *args, n_rfeat: int, **kw):
        super().__init__(*args, **kw)
        d = self.inp[-1].out_features
        self.r_inp = nn.Linear(n_rfeat, d)
        nn.init.zeros_(self.r_inp.weight)
        nn.init.zeros_(self.r_inp.bias)
        self.r_same = nn.Parameter(torch.zeros(len(self.layers), self.heads))
        self.r_cut = nn.Parameter(torch.zeros(1))
        self.r_aff = nn.Parameter(torch.zeros(1))

    @classmethod
    def from_cut(cls, state: dict, **kw) -> RefineCutModel:
        """A refiner warm-started from a CutModel state dict (its new parameters zero)."""
        m = cls(**kw)
        missing, unexpected = m.load_state_dict(state, strict=False)
        if unexpected or set(missing) - set(cls.NEW):
            raise ValueError(f"not a CutModel state for this size: missing {missing}, unexpected {unexpected}")
        return m

    def new_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if n in self.NEW]

    def forward(self, batch):
        lab = batch["in_lab"]
        known = lab >= 0
        same = ((lab[:, :, None] == lab[:, None, :]) & known[:, :, None] & known[:, None, :]).float()
        x = self.embed(batch) + self.r_inp(batch["rfeat"])
        bias = self.attn_bias(batch)
        for li, layer in enumerate(self.layers):
            x = layer(x, bias + self.r_same[li][None, :, None, None] * same[:, None])
        out = self.outputs(x, batch)
        nxt_lab = torch.gather(lab, 1, batch["nxt"])
        cut_in = ((lab != nxt_lab) | ~known).float()
        out["cut"] = out["cut"] + self.r_cut * cut_in
        pair_known = (known[:, :, None] & known[:, None, :]).float()
        out["aff"] = out["aff"] + self.r_aff * (2 * same - pair_known)
        return out


def cut_losses(out, batch, pos_weight: float = 3.0, sample_weight: torch.Tensor | None = None):
    """(cut BCE, affinity BCE). sample_weight (B,): weight per glyph (refiner pairs); None = all 1."""
    m = batch["cut_mask"]
    if sample_weight is not None:
        return _weighted_cut_losses(out, batch, pos_weight, sample_weight)
    l_cut = F.binary_cross_entropy_with_logits(
        out["cut"].float(), batch["cut"], weight=m, pos_weight=torch.tensor(pos_weight, device=m.device),
        reduction="sum",
    ) / m.sum().clamp(min=1)
    lab = batch["lab"]
    valid = (lab >= 0) & ~batch["pad"]
    pm = valid[:, :, None] & valid[:, None, :]
    eye = torch.eye(lab.shape[1], dtype=torch.bool, device=lab.device)[None]
    pm = pm & ~eye
    same = (lab[:, :, None] == lab[:, None, :]) & pm
    diff = pm & ~same
    la = F.binary_cross_entropy_with_logits(out["aff"].float(), same.float(), reduction="none")
    l_aff = 0.5 * (la[same].mean() if same.any() else 0.0) + 0.5 * (la[diff].mean() if diff.any() else 0.0)
    return l_cut, l_aff


def _affinity_masks(batch):
    lab = batch["lab"]
    valid = (lab >= 0) & ~batch["pad"]
    pm = valid[:, :, None] & valid[:, None, :]
    pm = pm & ~torch.eye(lab.shape[1], dtype=torch.bool, device=lab.device)[None]
    same = (lab[:, :, None] == lab[:, None, :]) & pm
    return same, pm & ~same


def _weighted_cut_losses(out, batch, pos_weight, w):
    """cut_losses with each glyph's terms weighted by w (weighted means instead of plain means)."""
    m = batch["cut_mask"] * w[:, None]
    l_cut = F.binary_cross_entropy_with_logits(
        out["cut"].float(), batch["cut"], weight=m, pos_weight=torch.tensor(pos_weight, device=m.device),
        reduction="sum",
    ) / m.sum().clamp(min=1e-6)
    same, diff = _affinity_masks(batch)
    la = F.binary_cross_entropy_with_logits(out["aff"].float(), same.float(), reduction="none")
    wp = w[:, None, None].expand_as(la)

    def mean(sel):
        return (la * wp)[sel].sum() / wp[sel].sum().clamp(min=1e-6) if sel.any() else 0.0

    return l_cut, 0.5 * mean(same) + 0.5 * mean(diff)


# ---------------------------------------------------------------- Stage C
class OrderModel(nn.Module):
    def __init__(self, in_dim: int, d: int = 256, layers: int = 4, heads: int = 8, pair_dim: int = 0):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))
        self.pair_dim = pair_dim
        if pair_dim:
            self.pemb = nn.Sequential(nn.Linear(pair_dim, 64), nn.GELU(), nn.Linear(64, 64))
            self.pbias = nn.ModuleList([nn.Linear(64, heads) for _ in range(layers)])
        self.layers = nn.ModuleList([BiasedLayer(d, heads) for _ in range(layers)])
        self.heads = heads
        self.norm = nn.LayerNorm(d)
        self.dir = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.pq = nn.Linear(d, d)
        self.pk = nn.Linear(d, d)
        self.pair = nn.Sequential(nn.Linear(2 * d + (64 if pair_dim else 0), d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x, pad, pair=None):
        B, N, _ = x.shape
        h = self.inp(x)
        mask = torch.zeros(B, self.heads, N, N, device=x.device, dtype=h.dtype)
        mask = mask.masked_fill(pad[:, None, None, :], float("-inf"))
        pe = self.pemb(pair) if self.pair_dim else None
        for li, layer in enumerate(self.layers):
            bias = mask + self.pbias[li](pe).permute(0, 3, 1, 2) if self.pair_dim else mask
            h = layer(h, bias)
        h = self.norm(h)
        d = self.dir(h).squeeze(-1)
        # antisymmetric precedence logits: s_ij = f(i,j) - f(j,i)
        a, b = self.pq(h), self.pk(h)
        cat = [a[:, :, None].expand(-1, -1, N, -1), b[:, None].expand(-1, N, -1, -1)]
        if self.pair_dim:
            cat.append(pe)
        f = self.pair(torch.cat(cat, -1)).squeeze(-1)
        prec = f - f.transpose(1, 2)
        return dict(dir=d, prec=prec)


class OrderNet(nn.Module):
    """OrderModel with Fourier features on the resampled median points."""

    def __init__(self, d: int = 256, layers: int = 4):
        super().__init__()
        self.core = OrderModel(N_SFEAT + 32 * 12, d=d, layers=layers, pair_dim=N_PFEAT)

    def forward(self, x, pad, pair):
        return self.core(torch.cat([x, fourier(x[..., :32], 6)], -1), pad, pair)
