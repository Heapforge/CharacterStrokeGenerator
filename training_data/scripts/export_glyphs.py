"""Export the original font glyphs for every character in hanzi-writer-data-youyin.

usage: python -I export_glyphs.py <font path, e.g. dir/ukai.ttc(AR PL UKai CN)> <chars.txt | --all> <out_dir>
Writes <out_dir>/<char>.svg for each character the font contains.
"""
import os
import sys
import unicodedata

import fontforge

font_path, chars_file, out_dir = sys.argv[1:4]

os.makedirs(out_dir, exist_ok=True)
font = fontforge.open(font_path)
was_cid = bool(font.cidfontname)
if was_cid:
    font.cidFlatten()
by_code = {}
for g in font.glyphs():
    # cidFlatten leaves GPOS anchor points pointing at freed CID AnchorClasses, which crashes export().
    if was_cid:
        g.anchorPoints = ()
    if g.unicode >= 0:
        by_code[g.unicode] = g.glyphname
    for u, vs, _ in g.altuni or ():
        if u >= 0 and vs == -1:
            by_code.setdefault(u, g.glyphname)
if chars_file == "--all":
    chars = sorted(
        chr(u)
        for u in by_code
        if u <= 0x10FFFF and unicodedata.category(chr(u)) not in ("Co", "Cs")
    )
else:
    chars = [l.rstrip("\n") for l in open(chars_file, encoding="utf-8") if l.strip()]
by_glyph: dict[str, list[str]] = {}
for c in chars:
    if c.startswith(".") or any(ch in c for ch in "/\\\0") or c.strip() != c:
        continue
    gname = by_code.get(ord(c))
    if gname:
        by_glyph.setdefault(gname, []).append(c)

exported = 0
for gname, cs in by_glyph.items():
    g = font[gname]
    # Composite glyphs (À = A + grave, Cyrillic Т -> Latin T, ...) are built from references,
    # which export() drops; turn them into plain outlines first.
    if g.references:
        g.unlinkRef()
    if not g.isWorthOutputting() or len(g.foreground) == 0:
        continue
    first = os.path.join(out_dir, cs[0] + ".svg")
    g.export(first)
    if len(cs) > 1:
        data = open(first, "rb").read()
        for c in cs[1:]:
            open(os.path.join(out_dir, c + ".svg"), "wb").write(data)
    exported += len(cs)
# font.close() is skipped: it raises on some FontForge builds after export; process exit frees it.
print(f"{font_path}: exported {exported}/{len(chars)}")
