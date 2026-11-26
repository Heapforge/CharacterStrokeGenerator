"""Train the Stage A refiner on the attempt-store pairs (run python -m strokegen.refine_dataset first).

usage: python -m strokegen.train_refine [--base checkpoints/cut.pt] [--epochs 12] [--bs 12] [--budget-min 55]

Warm-starts model.RefineCutModel from the Stage A checkpoint (its new inputs start at zero, so epoch "-1",
evaluated before any step, is exactly the base model) and trains it with Stage A's losses on
data/processed/refine_{train,val}.pkl, each pair weighted by its recency weight. Sized for one RTX 2070
Super (8 GB) in about an hour: batch 12, mixed precision, and training stops at the first epoch boundary
past --budget-min. Writes checkpoints/refine.pt (the epoch with the best validation score).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader

from .attempts.refs import file_sha256
from .data import load_split, to_device
from .decode import match_strokes
from .features import FEATURE_VERSION
from .model import RefineCutModel, cut_losses
from .paths import CKPT
from .predict import load_checkpoint
from .refine import N_RFEAT, REFINE_FEATURES, REFINE_VERSION, RefineDataset, collate_refine
from .train_cut import default_workers, evaluate, lr_schedule


def input_baseline(recs: list[dict]) -> dict:
    """How good the refiner's inputs already are (input labels scored against the targets)."""
    agg = dict(n_gt=0, exact=0, glyph_exact=0, miou=0.0)
    for r in recs:
        w = np.array([s.length() for c in r["contours"] for s in c])
        m = match_strokes(np.asarray(r["input_seg_stroke"]), r["seg_stroke"], w)
        for k in agg:
            agg[k] += m[k]
    n = max(1, len(recs))
    return dict(stroke_exact=agg["exact"] / max(1, agg["n_gt"]), glyph_exact=agg["glyph_exact"] / n,
                miou=agg["miou"] / n)


def train_epoch(model, loader, opt, sched, scaler, dev, deadline: float | None = None,
                use_weights: bool = True) -> tuple[np.ndarray, int]:
    """One pass over loader. Returns (mean [cut, aff] losses, steps); stops early at `deadline` (time.time())."""
    model.train()
    sums, steps = np.zeros(2), 0
    for batch in loader:
        batch = to_device(batch, dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(batch)
        l_cut, l_aff = cut_losses(out, batch, sample_weight=batch["weight"] if use_weights else None)
        loss = l_cut + l_aff
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        sums += [l_cut.item(), float(l_aff.detach()) if torch.is_tensor(l_aff) else l_aff]
        steps += 1
        if deadline is not None and time.time() > deadline:
            break
    return sums / max(1, steps), steps


def make_optimizer(model: RefineCutModel, lr: float, new_lr: float) -> torch.optim.Optimizer:
    """AdamW; the zero-initialised refiner inputs learn at new_lr, the warm-started rest at lr."""
    new = {id(p) for p in model.new_parameters()}
    old = [p for p in model.parameters() if id(p) not in new]
    return torch.optim.AdamW([dict(params=old, lr=lr), dict(params=model.new_parameters(), lr=new_lr,
                                                             weight_decay=0.0)], lr=lr, weight_decay=0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(CKPT, "cut.pt"), help="Stage A checkpoint to start from")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-5, help="learning rate of the warm-started Stage A weights")
    ap.add_argument("--new-lr", type=float, default=1e-3, help="learning rate of the new (zero-init) inputs")
    ap.add_argument("--budget-min", type=float, default=55.0, help="stop training after this many minutes")
    ap.add_argument("--no-weights", action="store_true", help="ignore the pairs' recency weights")
    ap.add_argument("--workers", type=int, default=default_workers(8), help="DataLoader workers")
    ap.add_argument("--decode-workers", type=int, default=default_workers(8),
                    help="processes that decode validation partitions")
    ap.add_argument("--out", default="refine.pt")
    ap.add_argument("--limit", type=int, default=0, help="train on a subset (debug)")
    args = ap.parse_args()

    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise SystemExit("train_refine needs a CUDA device")
    dev = torch.device("cuda")
    t_start = time.time()
    deadline = t_start + args.budget_min * 60
    base = load_checkpoint(args.base, FEATURE_VERSION, "strokegen.train_cut")
    d, layers = base["args"].get("d", 256), base["args"].get("layers", 8)
    model = RefineCutModel.from_cut(base["model"], d=d, layers=layers, n_rfeat=N_RFEAT).to(dev)

    train = load_split("refine_train")
    if args.limit:
        train = train[: args.limit]
    val = load_split("refine_val")
    tds, vds = RefineDataset(train, True, use_weights=not args.no_weights), RefineDataset(val, False)
    print(f"pairs: train {len(tds)} val {len(vds)} · refiner inputs {', '.join(REFINE_FEATURES)}")
    if len(tds) < args.bs:
        raise SystemExit(f"not enough refiner pairs for one batch of {args.bs}")
    tl = DataLoader(tds, args.bs, shuffle=True, num_workers=args.workers, collate_fn=collate_refine,
                    drop_last=True, persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    vl = DataLoader(vds, 16, shuffle=False, num_workers=min(4, args.workers), collate_fn=collate_refine)
    print("val inputs as they are:", {k: round(v, 4) for k, v in input_baseline(vds.recs).items()})

    opt = make_optimizer(model, args.lr, args.new_lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule(args.epochs * len(tl), 200))
    scaler = torch.amp.GradScaler()
    pool = None
    if args.decode_workers > 1:
        pool = ProcessPoolExecutor(args.decode_workers, mp_context=mp.get_context("spawn"))
    ckpt_args = dict(vars(args), d=d, layers=layers, feat_version=FEATURE_VERSION, refine_version=REFINE_VERSION,
                     refine_features=list(REFINE_FEATURES), base_sha=file_sha256(args.base),
                     base_epoch=base.get("epoch"))
    os.makedirs(CKPT, exist_ok=True)
    m0 = evaluate(model, vl, dev, pool=pool)  # == the base Stage A model (zero-init)
    print(f"ep  -1 | base model on the refiner inputs: stroke {m0['stroke_exact']:.4f} glyph {m0['glyph_exact']:.4f} "
          f"mIoU {m0['miou']:.4f} cutF1 {m0['cut_f1']:.4f}", flush=True)
    base_score = m0["stroke_exact"] + m0["glyph_exact"]
    best = -1.0
    for ep in range(args.epochs):
        t0 = time.time()
        losses, steps = train_epoch(model, tl, opt, sched, scaler, dev, deadline, not args.no_weights)
        m = evaluate(model, vl, dev, pool=pool)
        print(f"ep {ep:3d} | loss cut {losses[0]:.4f} aff {losses[1]:.4f} ({steps} steps) | val cutF1 "
              f"{m['cut_f1']:.4f} stroke {m['stroke_exact']:.4f} glyph {m['glyph_exact']:.4f} mIoU {m['miou']:.4f} "
              f"| {time.time() - t0:.0f}s", flush=True)
        score = m["stroke_exact"] + m["glyph_exact"]
        if score > best:
            best = score
            torch.save(dict(model=model.state_dict(), args=ckpt_args, epoch=ep, val=m, base_val=m0),
                       os.path.join(CKPT, args.out))
        if time.time() > deadline:
            print(f"time budget of {args.budget_min:g} min reached after epoch {ep}")
            break
    if pool:
        pool.shutdown()
    verdict = "better than" if best > base_score else "NOT better than"
    print(f"best val score {best:.4f}, {verdict} the base model ({base_score:.4f}) · "
          f"{(time.time() - t_start) / 60:.0f} min")


if __name__ == "__main__":
    main()
