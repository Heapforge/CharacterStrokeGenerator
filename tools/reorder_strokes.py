#!/usr/bin/env python3
"""Reorder the strokes of a hanzi-writer JSON file.

The order is the new sequence of the current 1-based stroke numbers:

    tools/reorder_strokes.py 中.json 1 3 2 4

makes the current stroke 3 the new second stroke, and so on. `strokes` and
`medians` are reordered together, and `radStrokes` indices are remapped.
The file is overwritten unless -o is given.
"""

import argparse
import json
import sys
from pathlib import Path


def reorder(data: dict, order: list[int]) -> dict:
    n = len(data["strokes"])
    if len(data.get("medians", [])) != n:
        raise ValueError(f"file has {n} strokes but {len(data.get('medians', []))} medians")
    if sorted(order) != list(range(1, n + 1)):
        raise ValueError(f"order must use each of 1..{n} exactly once, got: {' '.join(map(str, order))}")

    idx = [i - 1 for i in order]
    out = dict(data)
    out["strokes"] = [data["strokes"][i] for i in idx]
    out["medians"] = [data["medians"][i] for i in idx]
    if "radStrokes" in data:
        new_pos = {old: new for new, old in enumerate(idx)}
        out["radStrokes"] = sorted(new_pos[i] for i in data["radStrokes"])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Reorder strokes in a hanzi-writer JSON file.")
    p.add_argument("file", type=Path, help="hanzi-writer JSON file")
    p.add_argument("order", type=int, nargs="*",
                   help="new order as current 1-based stroke numbers; omit to print the stroke count")
    p.add_argument("-o", "--output", type=Path, help="write here instead of overwriting the input")
    args = p.parse_args()

    data = json.loads(args.file.read_text(encoding="utf-8"))

    if not args.order:
        print(f"{args.file}: {len(data['strokes'])} strokes")
        return 0

    try:
        out = reorder(data, args.order)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    dest = args.output or args.file
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {dest}: {' '.join(map(str, args.order))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
