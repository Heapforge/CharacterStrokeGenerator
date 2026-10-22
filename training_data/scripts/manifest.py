import json, os, sys, unicodedata
data_dir, out_dir = sys.argv[1:3]
fonts = ["gkai00mp", "bkai00mp", "ukai-cn", "ukai-hk", "ukai-tw", "ukai-tw-mbe", "noto-sans"]
have = {f: {n[:-4] for n in os.listdir(os.path.join(out_dir, f))} for f in fonts}
entries = {}
for n in sorted(os.listdir(data_dir)):
    if not n.endswith(".json"): continue
    stem = n[:-5]
    base = stem[:-3] if stem.endswith(("-jp", "-ko")) else stem
    entries.setdefault(base, {"data_files": [], "fonts": [f for f in fonts if base in have[f]]})["data_files"].append(n)
missing = sorted(c for c, e in entries.items() if not e["fonts"])
json.dump({"fonts": fonts, "characters": entries, "missing": missing},
          open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(os.path.join(out_dir, "missing.txt"), "w", encoding="utf-8") as f:
    for c in missing:
        f.write(f"{c}\tU+{ord(c):04X}\t{unicodedata.name(c, '?')}\t{' '.join(entries[c]['data_files'])}\n")
print(len(entries), "characters,", len(missing), "missing")
