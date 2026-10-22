"""Export the original font glyphs for every character in hanzi-writer-data-youyin.

usage: python -I export_glyphs.py <font path, e.g. dir/ukai.ttc(AR PL UKai CN)> <chars.txt> <out_dir>
Writes <out_dir>/<char>.svg for each character the font contains.
"""
import os
import sys

import fontforge

font_path, chars_file, out_dir = sys.argv[1:4]
chars = [l.rstrip("\n") for l in open(chars_file, encoding="utf-8") if l.strip()]

os.makedirs(out_dir, exist_ok=True)
font = fontforge.open(font_path)
exported = 0
for c in chars:
    u = ord(c)
    if u not in font:
        continue
    g = font[u]
    # Composite glyphs (À = A + grave, Cyrillic Т -> Latin T, ...) are built from references,
    # which export() drops; turn them into plain outlines first.
    if g.references:
        g.unlinkRef()
    if not g.isWorthOutputting() or len(g.foreground) == 0:
        continue
    g.export(os.path.join(out_dir, c + ".svg"))
    exported += 1
# font.close() is skipped: it raises on some FontForge builds after export; process exit frees it.
print(f"{font_path}: exported {exported}/{len(chars)}")
