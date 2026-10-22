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
            nn.Linear(N_FEAT + 4 * 16 * 2 + n_probe * self.cnn.out_dim, d), nn.GELU(), nn.Linear(d, d)
        )
        self.bias = nn.Embedding(N_BUCKETS, heads)
        self.layers = nn.ModuleList([BiasedLayer(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.cut = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.aff = nn.Linear(d, aff_dim)
        self.aff_b = nn.Parameter(torch.zeros(1))
        self.aff_dim = aff_dim

    def forward(self, batch):
        feat, pad = batch["feat"], batch["pad"]  # pad: (B,N) True for padding
        B, N, _ = feat.shape
        xy = feat[..., 8:10]  # segment midpoint
        se = torch.cat([feat[..., 0:2], feat[..., 6:8]], -1)  # start, end
        cnnf = self.cnn(batch["img"], batch["probe"])
        x = self.inp(torch.cat([feat, fourier(torch.cat([xy, se], -1)[..., :4]), cnnf], -1))
        bias = self.bias(batch["bucket"].long()).permute(0, 3, 1, 2)  # (B,H,N,N)
        bias = bias.masked_fill(pad[:, None, None, :], float("-inf"))
        for layer in self.layers:
            x = layer(x, bias)
        h = self.norm(x)
        hn = torch.gather(h, 1, batch["nxt"][..., None].expand(-1, -1, h.shape[-1]))
        cut = self.cut(torch.cat([h, hn], -1)).squeeze(-1)
        u = self.aff(h)
        aff = u @ u.transpose(1, 2) / math.sqrt(self.aff_dim) + self.aff_b
        return dict(cut=cut, aff=aff, h=h)


def cut_losses(out, batch, pos_weight: float = 3.0):
    m = batch["cut_mask"]
    l_cut = F.binary_cross_entropy_with_logits(
        out["cut"].float(), batch["cut"], weight=m, pos_weight=torch.tensor(pos_weight, device=m.device), reduction="sum"
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
