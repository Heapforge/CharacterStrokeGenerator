"""Attempt store CLI: staging, export/import for repair passes, verdicts, history and stats.

    uv run python -m strokegen.attempts stage [--font F]
    uv run python -m strokegen.attempts export --font F [--status pending] [--failing] [--format list] -o queue.json
    uv run python -m strokegen.attempts import out.json --font F --source fixup_script --script tools/fixups/RUN/x.py
    uv run python -m strokegen.attempts import out_dir/ --font F --source llm_repair --llm-model M --run RUN_ID
    uv run python -m strokegen.attempts reject CHAR --font F --reason "..." [--tags a,b] [--attempt ID]
    uv run python -m strokegen.attempts unresolved CHAR --font F --reason "..."
    uv run python -m strokegen.attempts history CHAR [--font F] [--json]
    uv run python -m strokegen.attempts stats [--by font|script|source|fixup]
    uv run python -m strokegen.attempts check [--font F] [--stale]
    uv run python -m strokegen.attempts backfill-hwdata [--chars ...] [--dry-run]
    uv run python -m strokegen.attempts reindex

Approving is the dashboard's Save button, never a CLI command: a human looks at every approved glyph.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from collections import Counter, defaultdict

from ..paths import ATTEMPTS, GLYPHS
from .check import CHECK_VERSION
from .refs import script_ref
from .store import AttemptStore


def _store(a) -> AttemptStore:
    return AttemptStore(a.root)


def _font_contours(font: str, char: str):
    from ..svg_io import load_svg_glyph
    p = os.path.join(GLYPHS, font, f"{char}.svg")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return load_svg_glyph(f.read())


def read_results(path: str) -> dict[str, dict]:
    """char -> {strokes, medians} from a tool's output: a JSON map, a JSON list of {char, ...}, or a
    directory of {char}.json files (names starting with _ are reports and are skipped)."""
    out: dict[str, dict] = {}
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.endswith(".json") and not name.startswith("_"):
                with open(os.path.join(path, name), encoding="utf-8") as f:
                    out[name[:-5]] = json.load(f)
        return out
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.items() if isinstance(data, dict) else ((it.get("char"), it) for it in data)
    for c, d in items:
        if isinstance(c, str) and not c.startswith("_") and isinstance(d, dict) and "strokes" in d:
            out[c] = d
    return out


# ---------------------------------------------------------------- commands
def cmd_stage(a):
    n = _store(a).stage(a.font)
    print(f"{n} glyphs staged under {os.path.relpath(os.path.join(a.root, 'staging'))}")


def cmd_export(a):
    s = _store(a)
    s.stage(a.font)
    rows = []
    for r in s.staged_list(a.font):
        if a.status != "any" and r["status"] != a.status:
            continue
        if a.failing and r["passed"] is not False:
            continue
        if r["has_training_file"] and not a.include_existing:
            continue
        if a.chars and r["char"] not in a.chars:
            continue
        with open(s.staging_path(a.font, r["char"]), encoding="utf-8") as f:
            rows.append(json.load(f))
    if a.format == "list":
        out = [{k: r[k] for k in ("char", "strokes", "medians", "segStroke", "attempt_id", "checker")} for r in rows]
    else:
        out = {r["char"]: {k: r[k] for k in ("strokes", "medians", "segStroke", "attempt_id", "checker")}
               for r in rows}
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"exported {len(rows)} glyphs of {a.font} to {a.output}")


def cmd_import(a):
    s = _store(a)
    if a.source == "fixup_script":
        if not a.script:
            sys.exit("--source fixup_script needs --script (the script that produced the output)")
        ref = script_ref(a.script, a.args)
        if ref["dirty"]:
            print(f"warning: {ref['script']} differs from the committed version; commit it so this run "
                  "can be reproduced (the blob sha is recorded either way)", file=sys.stderr)
    else:
        if not a.llm_model or not a.run:
            sys.exit("--source llm_repair needs --llm-model and --run")
        ref = dict(llm_model=a.llm_model, run_id=a.run)
        if a.notes:
            ref["notes"] = a.notes
    tags = [t for t in (a.tags or "").split(",") if t]
    results = read_results(a.input)
    counts = Counter()
    for char, d in results.items():
        if a.chars and char not in a.chars:
            continue
        staged = s.staged(a.font, char)
        hist = s.history(a.font, char, sync=False)
        if staged is not None:
            contours = s.contours(a.font, char, staged["glyph_key"])
        elif hist:
            contours = s.contours(a.font, char, hist[-1]["glyph_key"])
        else:
            contours = _font_contours(a.font, char)
        if contours is None:
            print(f"{char}: no outline for {a.font} (not in the store, no glyph SVG): skipped", file=sys.stderr)
            counts["no_outline"] += 1
            continue
        parent = a.parent or (staged or (hist[-1] if hist else {})).get("attempt_id")
        hw = {"strokes": d["strokes"], "medians": d["medians"]}
        from .labels import content_hash
        if parent and content_hash(hw) == s.get(parent)["content_hash"]:
            counts["unchanged"] += 1
            continue
        try:
            aid, created = s.record(a.font, char, contours, hw, a.source, ref, parent=parent, tags=tags)
        except (ValueError, KeyError) as e:
            print(f"{char}: {e}", file=sys.stderr)
            counts["invalid"] += 1
            continue
        if not created:
            counts["duplicate"] += 1
            continue
        counts["recorded"] += 1
        if not a.no_check:
            res = s.check(aid)
            counts["passed" if res["passed"] else "failed"] += 1
    print(", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "nothing to import")


def _pick(s: AttemptStore, a):
    if a.attempt:
        return s.get(a.attempt)
    st = s.staged(a.font, a.char)
    if st is None:
        sys.exit(f"{a.char} ({a.font}) has no staged version; pass --attempt")
    return st


def cmd_reject(a):
    s = _store(a)
    att = _pick(s, a)
    s.set_verdict(att["attempt_id"], "rejected", a.reason, [t for t in (a.tags or "").split(",") if t],
                  by=a.by or getpass.getuser())
    print(f"rejected {att['attempt_id']}")


def cmd_unresolved(a):
    s = _store(a)
    att = _pick(s, a)
    s.set_verdict(att["attempt_id"], "unresolved", a.reason, [t for t in (a.tags or "").split(",") if t],
                  by=a.by or getpass.getuser())
    print(f"marked {a.char} ({a.font}) unresolved at {att['attempt_id']}")


def _ref_label(att) -> str:
    r = att["source_ref"] or {}
    if att["source_type"] == "model":
        return f"cut {r.get('cut_sha', '?')[:8]} order {r.get('order_sha', '?')[:8]}"
    if att["source_type"] == "fixup_script":
        return f"{r.get('script')}@{(r.get('blob_sha') or '?')[:8]}{' dirty' if r.get('dirty') else ''}"
    if att["source_type"] == "llm_repair":
        return f"{r.get('llm_model')} run {r.get('run_id')}"
    if att["source_type"] == "human_edit":
        return f"{r.get('reviewer')} {r.get('edits') or ''}"
    return json.dumps(r, ensure_ascii=False)[:60]


def cmd_history(a):
    s = _store(a)
    fonts = [a.font] if a.font else sorted({f for f, c in s.glyphs() if c == a.char})
    rows = {f: s.history(f, a.char) for f in fonts}
    if a.json:
        json.dump(rows, sys.stdout, ensure_ascii=False, indent=1)
        print()
        return
    for f, hist in rows.items():
        st, _ = s.status(hist)
        print(f"{a.char} {f}: {st}, {len(hist)} versions")
        for h in hist:
            c = h["checker"]
            chk = "unchecked" if c is None else ("PASS" if c["passed"] else "FAIL " + ",".join(
                sorted(set(c.get("failures", []) + [x for r in c.get("strokes", []) for x in r.get("failures", [])]))))
            par = f" <- {h['parent_attempt_id']}" if h["parent_attempt_id"] else ""
            why = f" ({h['reason']})" if h["reason"] else ""
            print(f"  {h['attempt_id']}{par}  {h['created_at'][:19]}  {h['source_type']:<12} {_ref_label(h)}")
            print(f"      {len(h['hw']['strokes'])} strokes  {chk}  -> {h['verdict']}{why}")


LINEAGE_RANK = ("llm_repair", "fixup_script", "refiner", "human_edit", "synthetic", "imported", "model")
OUTCOME = {"model": "first_try", "human_edit": "after_human_edit", "fixup_script": "after_fixup",
           "llm_repair": "after_llm", "refiner": "after_refiner", "imported": "imported", "synthetic": "synthetic"}


def lineage(s: AttemptStore, att: dict, by_id: dict) -> list[dict]:
    out, seen = [att], {att["attempt_id"]}
    while out[-1]["parent_attempt_id"] and out[-1]["parent_attempt_id"] not in seen:
        p = by_id.get(out[-1]["parent_attempt_id"]) or s.get(out[-1]["parent_attempt_id"])
        out.append(p)
        seen.add(p["attempt_id"])
    return out


def outcome(s: AttemptStore, hist: list[dict]) -> tuple[str, list[dict]]:
    """How a glyph ended up: first_try / after_fixup / after_llm / ... / pending / unresolved."""
    st, att = s.status(hist)
    if st != "approved":
        return st, []
    lin = lineage(s, att, {h["attempt_id"]: h for h in hist})
    kinds = {x["source_type"] for x in lin}
    for k in LINEAGE_RANK:
        if k in kinds:
            return OUTCOME[k], lin
    return "other", lin


def glyph_stats(s: AttemptStore) -> list[dict]:
    rows = []
    for font, char in s.glyphs():
        hist = s.history(font, char, sync=False)
        oc, lin = outcome(s, hist)
        scripts = sorted({x["source_ref"].get("script") for x in (lin or hist)
                          if x["source_type"] == "fixup_script" and x["source_ref"].get("script")})
        rows.append(dict(font=font, char=char, script=hist[0]["script"], outcome=oc, versions=len(hist),
                         fixups=scripts))
    return rows


def cmd_stats(a):
    s = _store(a)
    rows = glyph_stats(s)
    cols = ["first_try", "after_human_edit", "after_fixup", "after_llm", "after_refiner", "imported", "pending",
            "unresolved", "replaced"]
    groups: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        keys = {"font": [r["font"]], "script": [r["script"]], "source": [r["outcome"]],
                "fixup": r["fixups"] or ["(none)"]}[a.by]
        for k in keys:
            groups[k][r["outcome"]] += 1
        groups["ALL"][r["outcome"]] += 1
    used = [c for c in cols if any(g[c] for g in groups.values())] or cols[:1]
    w = max([len(k) for k in groups] + [8])
    print(f"{a.by:<{w}} " + " ".join(f"{c:>16}" for c in used) + f" {'glyphs':>8}")
    for k in sorted(groups, key=lambda k: (k == "ALL", k)):
        g = groups[k]
        print(f"{k:<{w}} " + " ".join(f"{g[c]:>16}" for c in used) + f" {sum(g.values()):>8}")


def cmd_check(a):
    s = _store(a)
    n = 0
    for font, char in s.glyphs(a.font):
        for h in s.history(font, char, sync=False):
            if h["checker"] is None or (a.stale and h["checker"].get("version") != CHECK_VERSION):
                s.check(h["attempt_id"])
                n += 1
    print(f"checked {n} versions")


def cmd_reindex(a):
    s = _store(a)
    s.reindex()
    print(f"indexed {len(s.glyphs())} glyphs; {s.stage()} staged")


def cmd_backfill(a):
    from .backfill import backfill_hwdata
    backfill_hwdata(_store(a), chars=a.chars, dry_run=a.dry_run)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strokegen.attempts", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ATTEMPTS, help="store directory (default: training_data/attempts)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stage", help="rebuild the staging view")
    p.add_argument("--font")
    p.set_defaults(fn=cmd_stage)

    p = sub.add_parser("export", help="staged glyphs in the fix-up tools' input format")
    p.add_argument("--font", required=True)
    p.add_argument("--status", default="pending", choices=["pending", "unresolved", "any"])
    p.add_argument("--failing", action="store_true", help="only versions the checker failed")
    p.add_argument("--include-existing", action="store_true", help="also glyphs whose character has a training file")
    p.add_argument("--chars", default="")
    p.add_argument("--format", default="map", choices=["map", "list"],
                   help="map: char -> data (most tools); list: [{char, ...}] (fix_hole_strokes)")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("import", help="record a repair pass's output as new versions")
    p.add_argument("input", help="JSON map / list, or a directory of {char}.json")
    p.add_argument("--font", required=True)
    p.add_argument("--source", required=True, choices=["fixup_script", "llm_repair"])
    p.add_argument("--script", help="fixup_script: the script that produced the output")
    p.add_argument("--args", help="fixup_script: its command-line arguments")
    p.add_argument("--llm-model", help="llm_repair: model id")
    p.add_argument("--run", help="llm_repair: run id (YYYYMMDD-model-topic)")
    p.add_argument("--notes")
    p.add_argument("--parent", help="parent attempt (default: the glyph's staged version)")
    p.add_argument("--tags", help="comma-separated error tags this pass addresses")
    p.add_argument("--chars", default="")
    p.add_argument("--no-check", action="store_true")
    p.set_defaults(fn=cmd_import)

    for name, fn, hlp in (("reject", cmd_reject, "reject a version (a reason is required if it passed the checker)"),
                          ("unresolved", cmd_unresolved, "give up on a glyph for now, with a reason")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("char")
        p.add_argument("--font", required=True)
        p.add_argument("--reason", required=name == "unresolved")
        p.add_argument("--tags")
        p.add_argument("--attempt", help="attempt id (default: the staged version)")
        p.add_argument("--by")
        p.set_defaults(fn=fn)

    p = sub.add_parser("history", help="every version of a glyph")
    p.add_argument("char")
    p.add_argument("--font")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_history)

    p = sub.add_parser("stats", help="how glyphs got approved, by font / script / source / fix-up script")
    p.add_argument("--by", default="font", choices=["font", "script", "source", "fixup"])
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("check", help="run the checker on versions without a result")
    p.add_argument("--font")
    p.add_argument("--stale", action="store_true", help="also re-run results from an older checker version")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("backfill-hwdata", help="import the training-data submodule's git history as 'imported'")
    p.add_argument("--chars", default="")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_backfill)

    p = sub.add_parser("reindex", help="rebuild index.sqlite and the staging view from the logs")
    p.set_defaults(fn=cmd_reindex)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
