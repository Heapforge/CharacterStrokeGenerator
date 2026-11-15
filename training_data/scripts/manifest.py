"""Write font_glyphs/manifest.json and missing.txt: which exported fonts have a glyph for each
character of the hanzi-writer data.

usage: python manifest.py <hanzi-writer data dir> <font_glyphs dir>
"""
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from strokegen.fonts import FONT_PRIORITY  # noqa: E402

data_dir, out_dir = sys.argv[1:3]
fonts = list(FONT_PRIORITY)
have = {f: {n[:-4] for n in os.listdir(os.path.join(out_dir, f)) if n.endswith(".svg")}
        if os.path.isdir(os.path.join(out_dir, f)) else set() for f in fonts}
entries = {}
for n in sorted(os.listdir(data_dir)):
    if not n.endswith(".json"):
        continue
    stem = n[:-5]
    base = stem[:-3] if stem.endswith(("-jp", "-ko")) else stem
    entries.setdefault(base, {"data_files": [], "fonts": [f for f in fonts if base in have[f]]})["data_files"].append(n)
missing = sorted(c for c, e in entries.items() if not e["fonts"])
with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump({"fonts": fonts, "characters": entries, "missing": missing}, f, ensure_ascii=False, indent=1)
with open(os.path.join(out_dir, "missing.txt"), "w", encoding="utf-8") as f:
    for c in missing:
        name = unicodedata.name(c, "?") if len(c) == 1 else "?"
        code = f"U+{ord(c):04X}" if len(c) == 1 else "?"
        f.write(f"{c}\t{code}\t{name}\t{' '.join(entries[c]['data_files'])}\n")
print(len(entries), "characters,", len(missing), "missing")
