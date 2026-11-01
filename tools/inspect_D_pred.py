import json

with open("/tmp/queue100.json") as f:
    queue = json.load(f)

item = next(it for it in queue if it["char"] == "Đ")
print("Đ num strokes:", len(item["pred"]["strokes"]))
for i, m in enumerate(item["pred"]["medians"]):
    print(f"Stroke {i}: median {m}")
