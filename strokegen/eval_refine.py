"""Single pass vs the refinement loop at K = 1..K, as a markdown report.

usage: python -m strokegen.eval_refine [--split test] [--holdout-font F] [--k 3] [--limit N] [--out outputs/]

Runs StrokeGenerator(refine_iters=K) once. The loop is deterministic and only the round cap depends on K,
so the result at K = k is round min(k, last round) of that run, and round 0 is the single pass.
Per K it reports strokes cut exactly right, characters with all strokes right, stroke IoU mean / worst,
checker pass rate (tools/verify_glyph.check_glyph, as run by the loop), exact end-to-end order and fully
correct glyphs; the regression rate (correct at K = 0, wrong at K) and the oscillation rate (A -> B -> A);
broken down by script, with lowercase a variants separately. Writes refine_eval_<tag>.md and .json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

from .attempts.labels import lowercase_a_variant, script_of
from .data import load_split
from .evaluate import score_glyph
from .paths import CKPT, OUTPUTS
from .predict import StrokeGenerator

GROUPS = ("all", "cjk", "latin", "greek", "cyrillic", "other", "lowercase_a")


def round_result(res: dict, k: int) -> tuple[dict, dict | None]:
    """(result in score_glyph's form, checker result) of the loop capped at k rounds."""
    rounds = res.get("_rounds")
    if not rounds:  # failed or empty glyph: never entered the loop
        return res, None
    r = rounds[min(k, len(rounds) - 1)]
    return dict(_partition=np.asarray(r["partition"]), _order=np.asarray(r["order"]), medians=r["medians"],
                strokes=r["strokes"]), r["checker"]


def collect(gen: StrokeGenerator, recs: list[dict], k_max: int, workers: int | None = None) -> list[dict]:
    """Per glyph: its groups, and per K = 0..k_max its scores, checker verdict and loop state."""
    preds = gen.predict_contours_batch([r["contours"] for r in recs], workers=workers)
    rows = []
    for r, res in zip(recs, preds):
        n_rounds = len(res.get("_rounds") or [None]) - 1
        stop = (res.get("_refine") or {}).get("stop")
        per_k = []
        for k in range(k_max + 1):
            rk, chk = round_result(res, k)
            sc = score_glyph(r, rk) if "_error" not in res else None
            m = sc["m"] if sc else dict(n_gt=int(np.asarray(r["seg_stroke"]).max()) + 1, exact=0, glyph_exact=False,
                                        miou=0.0)
            per_k.append(dict(
                n_gt=int(m["n_gt"]), exact=int(m["exact"]), glyph_exact=bool(m["glyph_exact"]),
                miou=float(m["miou"]), order_ok=bool(sc and sc["order_ok"]), full=bool(sc and sc["full"]),
                passed=bool(chk and chk.get("passed")), rounds=min(k, n_rounds)))
        groups = ["all", script_of(r["char"])] + (["lowercase_a"] if lowercase_a_variant(r["char"]) else [])
        rows.append(dict(char=r["char"], font=r.get("font"), groups=groups, per_k=per_k, stop=stop,
                         n_rounds=n_rounds, error=res.get("_error")))
    return rows


def summarize(rows: list[dict], k_max: int) -> dict:
    """{group: [metrics at K = 0..k_max]} for every group with glyphs."""
    out: dict[str, list[dict]] = {}
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for g in row["groups"]:
            by_group[g].append(row)
    for g in GROUPS:
        rs = by_group.get(g)
        if not rs:
            continue
        n = len(rs)
        table = []
        for k in range(k_max + 1):
            cur = [r["per_k"][k] for r in rs]
            base = [r["per_k"][0] for r in rs]
            mious = np.array([c["miou"] for c in cur])
            right0 = sum(b["glyph_exact"] for b in base)
            regress = sum(b["glyph_exact"] and not c["glyph_exact"] for b, c in zip(base, cur))
            fixed = sum(c["glyph_exact"] and not b["glyph_exact"] for b, c in zip(base, cur))
            osc = sum(r["stop"] == "oscillation" and r["n_rounds"] <= k for r in rs)
            table.append(dict(
                K=k, glyphs=n,
                stroke_exact=sum(c["exact"] for c in cur) / max(1, sum(c["n_gt"] for c in cur)),
                glyph_exact=sum(c["glyph_exact"] for c in cur) / n,
                miou_mean=float(mious.mean()), miou_worst=float(mious.min()),
                checker_pass=sum(c["passed"] for c in cur) / n,
                order_exact=sum(c["order_ok"] for c in cur) / n,
                full_correct=sum(c["full"] for c in cur) / n,
                regressed=regress, regression_rate=regress / max(1, right0), fixed=fixed,
                oscillated=osc, oscillation_rate=osc / n,
                mean_rounds=float(np.mean([c["rounds"] for c in cur])),
            ))
        out[g] = table
    return out


COLUMNS = (("K", "K", "{}"), ("stroke_exact", "strokes exact", "{:.2%}"), ("glyph_exact", "glyphs all right",
           "{:.2%}"), ("miou_mean", "IoU mean", "{:.4f}"), ("miou_worst", "IoU worst", "{:.4f}"),
           ("checker_pass", "checker pass", "{:.2%}"), ("order_exact", "order exact e2e", "{:.2%}"),
           ("full_correct", "fully correct", "{:.2%}"), ("regression_rate", "regressions", "{:.2%}"),
           ("fixed", "fixed", "{}"), ("oscillation_rate", "oscillation", "{:.2%}"), ("mean_rounds", "rounds", "{:.2f}"))


def markdown(sets: dict[str, dict], meta: dict) -> str:
    lines = ["# Refinement loop evaluation", "",
             f"Single pass (K = 0) vs the refinement loop at K = 1..{meta['k']}. Refiner `{meta['refine_ckpt']}` "
             f"(sha `{meta['refine_sha']}`, epoch {meta['refine_epoch']}), base `{meta['cut_sha']}`.", "",
             "- *strokes exact*: GT strokes whose segments were predicted exactly; *glyphs all right*: every stroke "
             "exact; *IoU*: length-weighted stroke IoU per glyph (mean, and the worst glyph).",
             "- *checker pass*: `tools/verify_glyph.check_glyph`; *order exact e2e*: right cut and order; "
             "*fully correct*: also every direction.",
             "- *regressions*: share of the glyphs all right at K = 0 that are not at K; *fixed*: glyphs wrong at "
             "K = 0 and right at K; *oscillation*: glyphs whose loop returned to an earlier decomposition (A → B → A).",
             ""]
    for name, info in sets.items():
        lines += [f"## {name}", "", info["note"], ""]
        for g, table in info["summary"].items():
            lines += [f"### {g} ({table[0]['glyphs']} glyphs)", "",
                      "| " + " | ".join(h for _, h, _ in COLUMNS) + " |",
                      "|" + "---|" * len(COLUMNS)]
            for row in table:
                lines.append("| " + " | ".join(fmt.format(row[key]) for key, _, fmt in COLUMNS) + " |")
            lines.append("")
        stops = info["stops"]
        lines += ["Loop stops: " + ", ".join(f"{k} {v}" for k, v in sorted(stops.items())), ""]
    lines += ["## Reproduce", "", "```sh", *meta["commands"], "```", ""]
    return "\n".join(lines)


def holdout_records(font: str, limit: int = 0) -> tuple[list[dict], str]:
    """Labelled records cut from one font's outlines (build_dataset.process restricted to that font)."""
    from .build_dataset import GLYPHS, process

    with open(os.path.join(GLYPHS, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)["characters"]
    items = [(c, dict(e, fonts=[font])) for c, e in sorted(man.items()) if font in e["fonts"]]
    recs = []
    for item in items:
        _, rec, _ = process(item)
        if rec is not None:
            recs.append(rec)
            if limit and len(recs) >= limit:
                break
    try:
        trained = sum(r["font"] == font for r in load_split("train"))
    except FileNotFoundError:
        trained = -1
    note = (f"Glyphs of `{font}` with ground truth ({len(recs)}). " +
            ("The training split has no records from this font, so it is held out." if trained == 0 else
             f"Warning: the training split has {trained} records from this font, so it is NOT held out."
             if trained > 0 else "No training split to check whether it is held out."))
    return recs, note


def run_set(gen, recs, k, workers) -> dict:
    rows = collect(gen, recs, k, workers)
    stops: dict[str, int] = defaultdict(int)
    for r in rows:
        stops[r["stop"] or "not refined"] += 1
    return dict(rows=rows, summary=summarize(rows, k), stops=dict(stops))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="test", help="build_dataset split to evaluate on ('' to skip)")
    ap.add_argument("--holdout-font", help="also evaluate on this font's glyphs")
    ap.add_argument("--k", type=int, default=3, help="largest K to compare")
    ap.add_argument("--refine-ckpt", default=os.path.join(CKPT, "refine.pt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out", default=OUTPUTS)
    a = ap.parse_args()
    gen = StrokeGenerator(device=a.device, refine_iters=a.k, refine_ckpt=a.refine_ckpt)
    sets: dict[str, dict] = {}
    try:
        if a.split:
            recs = load_split(a.split)[: a.limit or None]
            sets[f"split {a.split}"] = dict(run_set(gen, recs, a.k, a.workers),
                                            note=f"build_dataset `{a.split}` split ({len(recs)} glyphs).")
        if a.holdout_font:
            recs, note = holdout_records(a.holdout_font, a.limit)
            sets[f"font {a.holdout_font}"] = dict(run_set(gen, recs, a.k, a.workers), note=note)
        ref = gen.refine_ref(0)
    finally:
        gen.close()
    if not sets:
        raise SystemExit("nothing to evaluate: give --split and/or --holdout-font")
    cmd = " ".join(["uv run python -m strokegen.eval_refine", *sys.argv[1:]])
    meta = dict(k=a.k, refine_ckpt=os.path.relpath(a.refine_ckpt), refine_sha=ref["refine_sha"],
                refine_epoch=ref["refine_epoch"], cut_sha=ref["cut_sha"], commands=[
                    "uv run python -m strokegen.build_dataset", "uv run python -m strokegen.refine_dataset",
                    "uv run python -m strokegen.train_refine", cmd])
    tag = "_".join(filter(None, [a.split, a.holdout_font]))
    os.makedirs(a.out, exist_ok=True)
    md = markdown(sets, meta)
    with open(os.path.join(a.out, f"refine_eval_{tag}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(a.out, f"refine_eval_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, sets={name: dict(summary=v["summary"], stops=v["stops"], note=v["note"],
                                                    glyphs=v["rows"]) for name, v in sets.items()}),
                  f, ensure_ascii=False, indent=1)
    print(md)
    print(f"wrote {a.out}/refine_eval_{tag}.md and .json")


if __name__ == "__main__":
    main()
