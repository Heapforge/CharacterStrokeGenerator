"""Train the Stage A cutting model.

usage: python -m strokegen.train_cut [--epochs 60] [--bs 24]
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ROOT, CutDataset, collate, load_split, to_device
from .decode import match_strokes, partition_from_predictions
from .model import CutModel, cut_losses

CKPT = os.path.join(ROOT, "checkpoints")


@torch.no_grad()
def evaluate(model, loader, dev, limit=None):
    model.eval()
    ds = loader.dataset
    agg = dict(n_gt=0, exact=0, glyphs=0, glyph_exact=0, miou=0.0, cut_tp=0, cut_fp=0, cut_fn=0)
    for batch in loader:
        batch = to_device(batch, dev)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(batch)
        cut_p = torch.sigmoid(out["cut"].float()).cpu().numpy()
        aff_p = torch.sigmoid(out["aff"].float()).cpu().numpy()
        for b, i in enumerate(batch["index"].tolist()):
            r = ds.recs[i]
            n = sum(len(c) for c in r["contours"])
            pred = partition_from_predictions(r["contours"], cut_p[b, :n], aff_p[b, :n, :n])
            w = np.array([s.length() for c in r["contours"] for s in c])
            m = match_strokes(pred, r["seg_stroke"], w)
            agg["n_gt"] += m["n_gt"]
            agg["exact"] += m["exact"]
            agg["glyphs"] += 1
            agg["glyph_exact"] += m["glyph_exact"]
            agg["miou"] += m["miou"]
            cm = batch["cut_mask"][b, :n].cpu().numpy() > 0
            ct = batch["cut"][b, :n].cpu().numpy() > 0.5
            cp = cut_p[b, :n] > 0.5
            agg["cut_tp"] += int((cp & ct & cm).sum())
            agg["cut_fp"] += int((cp & ~ct & cm).sum())
            agg["cut_fn"] += int((~cp & ct & cm).sum())
        if limit and agg["glyphs"] >= limit:
            break
    p = agg["cut_tp"] / max(1, agg["cut_tp"] + agg["cut_fp"])
    r = agg["cut_tp"] / max(1, agg["cut_tp"] + agg["cut_fn"])
    return dict(
        stroke_exact=agg["exact"] / max(1, agg["n_gt"]),
        glyph_exact=agg["glyph_exact"] / max(1, agg["glyphs"]),
        miou=agg["miou"] / max(1, agg["glyphs"]),
        cut_f1=2 * p * r / max(1e-9, p + r),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--out", default="cut.pt")
    ap.add_argument("--limit", type=int, default=0, help="train on a subset (debug)")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    train = load_split("train")
    if args.limit:
        train = train[: args.limit]
    val = load_split("val")
    tl = DataLoader(CutDataset(train, True), args.bs, shuffle=True, num_workers=args.workers, collate_fn=collate,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)
    vl = DataLoader(CutDataset(val, False), 32, shuffle=False, num_workers=8, collate_fn=collate)
    model = CutModel(d=args.d, layers=args.layers).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * len(tl)
    warm = min(1000, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    )
    scaler = torch.amp.GradScaler()
    os.makedirs(CKPT, exist_ok=True)
    best = -1.0
    step = 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        sums = np.zeros(2)
        for batch in tl:
            batch = to_device(batch, dev)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(batch)
            l_cut, l_aff = cut_losses(out, batch)
            loss = l_cut + l_aff
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            sums += [l_cut.item(), l_aff.item()]
        sums /= len(tl)
        m = evaluate(model, vl, dev)
        print(
            f"ep {ep:3d} | loss cut {sums[0]:.4f} aff {sums[1]:.4f} | val cutF1 {m['cut_f1']:.4f} "
            f"stroke {m['stroke_exact']:.4f} glyph {m['glyph_exact']:.4f} mIoU {m['miou']:.4f} | {time.time() - t0:.0f}s",
            flush=True,
        )
        score = m["stroke_exact"] + m["glyph_exact"]
        if score > best:
            best = score
            torch.save(dict(model=model.state_dict(), args=vars(args), epoch=ep, val=m), os.path.join(CKPT, args.out))
    print("best val score", best)


if __name__ == "__main__":
    main()
