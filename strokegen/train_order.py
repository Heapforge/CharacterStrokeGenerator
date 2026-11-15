"""Train the Stage C stroke order + direction model.

usage: python -m strokegen.train_order [--epochs 80]
Stroke medians come from Stage B run on the GT stroke outlines, so training sees the same
kind of input as inference. That Stage B output is cached in data/processed/order_{split}.pkl
and rebuilt automatically when the split or the order features change.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import load_split, to_device, worker_rng
from .medians import median_distance
from .model import OrderNet
from .order import (
    FEATURE_VERSION,
    N_PFEAT,
    N_SFEAT,
    augment_strokes,
    decode_order,
    kendall_tau,
    pair_features,
    stroke_features,
    stroke_record,
)
from .paths import CKPT, PROCESSED
from .svg_io import parse_hw_path
from .train_cut import default_workers, lr_schedule

__all__ = ["OrderNet", "order_split"]  # OrderNet lives in model.py; re-exported for older imports


def _prep(r):
    strokes, dirs = [], []
    for sd, gm in zip(r["strokes_gt"], r["medians"]):
        st = stroke_record(parse_hw_path(sd))
        m = st["median"]
        flip = median_distance(m[::-1], gm) < median_distance(m, gm)
        strokes.append(st)
        dirs.append(float(flip))
    return dict(char=r["char"], strokes=strokes, dir=np.array(dirs, np.float32))


def order_split(name: str, workers: int = 8) -> list[dict]:
    """Stage B output for a split, cached; the cache is rebuilt when the split file or the order
    features changed since it was written."""
    path = os.path.join(PROCESSED, f"order_{name}.pkl")
    key = dict(feat_version=FEATURE_VERSION, source_mtime=os.path.getmtime(os.path.join(PROCESSED, f"{name}.pkl")))
    if os.path.exists(path):
        with open(path, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, dict) and cached.get("key") == key:
            return cached["recs"]
        print(f"{path} is stale, rebuilding", flush=True)
    recs = load_split(name)
    with mp.get_context().Pool(workers) as pool:
        out = pool.map(_prep, recs, chunksize=16)
    with open(path, "wb") as f:
        pickle.dump(dict(key=key, recs=out), f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


class OrderDataset(Dataset):
    def __init__(self, recs, train: bool):
        self.recs, self.train = recs, train
        self._rng: dict = {}

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        S = len(r["strokes"])
        strokes = r["strokes"]
        perm = np.arange(S)
        if self.train:
            rng = worker_rng(self._rng)
            strokes = augment_strokes(strokes, rng)
            perm = rng.permutation(S)
        ordered = [strokes[p] for p in perm]
        return dict(x=stroke_features(ordered), pair=pair_features(ordered), rank=perm.astype(np.int64),
                    dir=r["dir"][perm], index=i)


def collate(items):
    B = len(items)
    S = max(len(it["x"]) for it in items)
    x = torch.zeros(B, S, N_SFEAT)
    pair = torch.zeros(B, S, S, N_PFEAT)
    pad = torch.ones(B, S, dtype=torch.bool)
    rank = torch.full((B, S), -1, dtype=torch.long)
    d = torch.zeros(B, S)
    for b, it in enumerate(items):
        n = len(it["x"])
        x[b, :n] = torch.from_numpy(it["x"])
        pair[b, :n, :n] = torch.from_numpy(it["pair"])
        pad[b, :n] = False
        rank[b, :n] = torch.from_numpy(it["rank"])
        d[b, :n] = torch.from_numpy(it["dir"])
    return dict(x=x, pair=pair, pad=pad, rank=rank, dir=d, index=torch.tensor([it["index"] for it in items]))


def order_losses(out, batch):
    valid = ~batch["pad"]
    l_dir = F.binary_cross_entropy_with_logits(out["dir"].float()[valid], batch["dir"][valid])
    r = batch["rank"]
    pm = valid[:, :, None] & valid[:, None, :]
    pm &= ~torch.eye(r.shape[1], dtype=torch.bool, device=r.device)[None]
    tgt = (r[:, :, None] < r[:, None, :]).float()
    l_prec = F.binary_cross_entropy_with_logits(out["prec"].float()[pm], tgt[pm])
    return l_dir, l_prec


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    n = dir_ok = n_glyph = exact = 0
    taus = []
    for batch in loader:
        batch = to_device(batch, dev)
        out = model(batch["x"], batch["pad"], batch["pair"])
        dl = out["dir"].float().cpu().numpy()
        pl = out["prec"].float().cpu().numpy()
        for b in range(len(dl)):
            S = int((~batch["pad"][b]).sum())
            gd = batch["dir"][b, :S].cpu().numpy()
            dir_ok += int(((dl[b, :S] > 0) == (gd > 0.5)).sum())
            n += S
            order = decode_order(pl[b, :S, :S])
            rank = batch["rank"][b, :S].cpu().numpy()
            taus.append(kendall_tau(order, rank))
            exact += int((rank[order] == np.arange(S)).all())
            n_glyph += 1
    return dict(dir_acc=dir_ok / max(1, n), order_exact=exact / max(1, n_glyph), tau=float(np.mean(taus)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--workers", type=int, default=default_workers(6), help="DataLoader workers")
    ap.add_argument("--prep-workers", type=int, default=default_workers(8), help="processes for the Stage B cache")
    ap.add_argument("--out", default="order.pt")
    args = ap.parse_args()
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise SystemExit("train_order needs a CUDA device")
    dev = torch.device("cuda")
    t0 = time.time()
    train, val = order_split("train", args.prep_workers), order_split("val", args.prep_workers)
    order_split("test", args.prep_workers)
    print(f"prepared order data in {time.time() - t0:.0f}s ({len(train)} train glyphs)", flush=True)
    tl = DataLoader(OrderDataset(train, True), args.bs, shuffle=True, num_workers=args.workers, collate_fn=collate,
                    drop_last=True, persistent_workers=True)
    vl = DataLoader(OrderDataset(val, False), 128, num_workers=min(2, args.workers), collate_fn=collate)
    model = OrderNet(layers=args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    if not len(tl):
        raise SystemExit(f"not enough training glyphs for one batch of {args.bs}")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule(args.epochs * len(tl), 500))
    os.makedirs(CKPT, exist_ok=True)
    ckpt_args = dict(vars(args), feat_version=FEATURE_VERSION)
    best = -1
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        sums = np.zeros(2)
        for batch in tl:
            batch = to_device(batch, dev)
            out = model(batch["x"], batch["pad"], batch["pair"])
            l_dir, l_prec = order_losses(out, batch)
            loss = l_dir + l_prec
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            sums += [l_dir.item(), l_prec.item()]
        sums /= len(tl)
        m = evaluate(model, vl, dev)
        print(f"ep {ep:3d} | loss dir {sums[0]:.4f} prec {sums[1]:.4f} | val dir {m['dir_acc']:.4f} "
              f"order_exact {m['order_exact']:.4f} tau {m['tau']:.4f} | {time.time() - t0:.0f}s", flush=True)
        score = m["dir_acc"] + m["order_exact"]
        if score > best:
            best = score
            torch.save(dict(model=model.state_dict(), args=ckpt_args, epoch=ep, val=m), os.path.join(CKPT, args.out))
    print("best", best)


if __name__ == "__main__":
    main()
