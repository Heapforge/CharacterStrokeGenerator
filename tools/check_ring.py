import json

with open("/tmp/queue100.json") as f:
    queue = json.load(f)

for c in ["Ů", "ů"]:
    item = next(it for it in queue if it["char"] == c)
    strokes = item["pred"]["strokes"]
    medians = item["pred"]["medians"]
    print(f"Char {c}: {len(strokes)} strokes")
    for si, (st, med) in enumerate(zip(strokes, medians)):
        print(f"  Stroke {si}: len(med)={len(med)}, med={med}")
