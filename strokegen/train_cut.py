"""Train the Stage A cutting model.

usage: python -m strokegen.train_cut [--epochs 60] [--bs 24]
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader

from ._batch import decode_partition
from .data import CutDataset, collate, load_split, to_device
from .decode import match_strokes
from .features import FEATURE_VERSION
from .model import CutModel, cut_losses
from .paths import CKPT


def lr_schedule(total: int, warm_max: int):
    """Linear warm-up then cosine decay, as a LambdaLR factor; safe for very short runs."""
    total = max(1, total)
    warm = max(1, min(warm_max, total // 10))
    return lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))


def default_workers(cap: int) -> int:
    return max(1, min(cap, (os.cpu_count() or 2) - 2))


@torch.no_grad()
def evaluate(model, loader, dev, limit=None, pool: ProcessPoolExecutor | None = None):
    """Validation metrics; the partition decoding of all glyphs runs on `pool` when given."""
    model.eval()
    ds = loader.dataset
    agg = dict(n_gt=0, exact=0, glyphs=0, glyph_exact=0, miou=0.0, cut_tp=0, cut_fp=0, cut_fn=0)
    jobs, recs = [], []
    for batch in loader:
        batch = to_device(batch, dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(batch)
        cut_p = torch.sigmoid(out["cut"].float()).cpu().numpy()
        aff_p = torch.sigmoid(out["aff"].float()).cpu().numpy()
        cut_mask, cut_t = batch["cut_mask"].cpu().numpy() > 0, batch["cut"].cpu().numpy() > 0.5
        for b, i in enumerate(batch["index"].tolist()):
            r = ds.recs[i]
            n = sum(len(c) for c in r["contours"])
            jobs.append((r["contours"], cut_p[b, :n].copy(), aff_p[b, :n, :n].copy()))
            recs.append(r)
            cm, ct, cp = cut_mask[b, :n], cut_t[b, :n], cut_p[b, :n] > 0.5
            agg["cut_tp"] += int((cp & ct & cm).sum())
            agg["cut_fp"] += int((cp & ~ct & cm).sum())
            agg["cut_fn"] += int((~cp & ct & cm).sum())
        if limit and len(jobs) >= limit:
            break
    preds = pool.map(decode_partition, jobs, chunksize=4) if pool else map(decode_partition, jobs)
    for r, pred in zip(recs, preds):
        w = np.array([s.length() for c in r["contours"] for s in c])
        m = match_strokes(pred, r["seg_stroke"], w)
        agg["n_gt"] += m["n_gt"]
        agg["exact"] += m["exact"]
        agg["glyphs"] += 1
        agg["glyph_exact"] += m["glyph_exact"]
        agg["miou"] += m["miou"]
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
    ap.add_argument("--workers", type=int, default=default_workers(18), help="DataLoader workers")
    ap.add_argument("--decode-workers", type=int, default=default_workers(16),
                    help="processes that decode validation partitions")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--out", default="cut.pt")
    ap.add_argument("--limit", type=int, default=0, help="train on a subset (debug)")
    args = ap.parse_args()

    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise SystemExit("train_cut needs a CUDA device")
    dev = torch.device("cuda")
    train = load_split("train")
    if args.limit:
        train = train[: args.limit]
    val = load_split("val")
    tl = DataLoader(CutDataset(train, True), args.bs, shuffle=True, num_workers=args.workers, collate_fn=collate,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)
    vl = DataLoader(CutDataset(val, False), 32, shuffle=False, num_workers=min(8, args.workers), collate_fn=collate)
    model = CutModel(d=args.d, layers=args.layers).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    if not len(tl):
        raise SystemExit(f"not enough training glyphs for one batch of {args.bs}")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule(args.epochs * len(tl), 1000))
    scaler = torch.amp.GradScaler()
    os.makedirs(CKPT, exist_ok=True)
    best = -1.0
    step = 0
    # decode in fresh processes: forking a process that holds CUDA state is unsafe
    pool = None
    if args.decode_workers > 1:
        pool = ProcessPoolExecutor(args.decode_workers, mp_context=mp.get_context("spawn"))
    ckpt_args = dict(vars(args), feat_version=FEATURE_VERSION)
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
        m = evaluate(model, vl, dev, pool=pool)
        print(
            f"ep {ep:3d} | loss cut {sums[0]:.4f} aff {sums[1]:.4f} | val cutF1 {m['cut_f1']:.4f} "
            f"stroke {m['stroke_exact']:.4f} glyph {m['glyph_exact']:.4f} mIoU {m['miou']:.4f} "
            f"| {time.time() - t0:.0f}s",
            flush=True,
        )
        score = m["stroke_exact"] + m["glyph_exact"]
        if score > best:
            best = score
            torch.save(dict(model=model.state_dict(), args=ckpt_args, epoch=ep, val=m), os.path.join(CKPT, args.out))
    if pool:
        pool.shutdown()
    print("best val score", best)


if __name__ == "__main__":
    main()
