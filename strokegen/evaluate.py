"""End-to-end evaluation on a held-out split, plus an HTML preview.

usage: python -m strokegen.evaluate [--split test] [--preview 40]
"""
from __future__ import annotations

import argparse
import html
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import ROOT, load_split
from .decode import assemble_strokes, fill_unassigned, match_strokes
from .medians import median_distance
from .predict import StrokeGenerator
from .raster import iou, render


def _match(pred: np.ndarray, gt: np.ndarray, w: np.ndarray) -> dict[int, int]:
    """pred stroke id -> gt stroke id (Hungarian on weighted segment IoU)."""
    v = gt >= 0
    P, G = int(pred.max()) + 1, int(gt.max()) + 1
    inter = np.zeros((P, G))
    np.add.at(inter, (pred[v], gt[v]), w[v])
    ps, gs = inter.sum(1), inter.sum(0)
    io = inter / (ps[:, None] + gs[None, :] - inter + 1e-9)
    r, c = linear_sum_assignment(-io)
    return {int(a): int(b) for a, b in zip(r, c) if io[a, b] > 0.3}


def evaluate(gen: StrokeGenerator, recs: list[dict]) -> tuple[dict, list[dict]]:
    agg = dict(glyphs=0, strokes=0, stroke_exact=0, glyph_part=0, union_iou=[], med_dist=[], dir_ok=0, dir_n=0,
               order_exact_e2e=0, full=0, gtp_order=0, gtp_dir_ok=0, gtp_dir_n=0)
    rows = []
    for r in recs:
        C = r["contours"]
        w = np.array([s.length() for c in C for s in c])
        gt = r["seg_stroke"]
        res = gen.predict_contours(C)
        part = res["_partition"]
        m = match_strokes(part, gt, w)
        agg["glyphs"] += 1
        agg["strokes"] += m["n_gt"]
        agg["stroke_exact"] += m["exact"]
        agg["glyph_part"] += m["glyph_exact"]
        loops = [s for s in assemble_strokes(C, part) if s]
        agg["union_iou"].append(iou(_union(loops), render(C, 256)))
        mp = _match(part, gt, w)
        ranks, dirs_ok = [], []
        for pos, k in enumerate(res["_order"]):
            g = mp.get(int(k))
            if g is None:
                ranks.append(-1)
                continue
            ranks.append(g)
            ours = np.array(res["medians"][pos], float)
            gm = r["medians"][g]
            d_f, d_r = median_distance(ours, gm), median_distance(ours[::-1], gm)
            agg["med_dist"].append(d_f)
            dirs_ok.append(d_f <= d_r)
        agg["dir_ok"] += sum(dirs_ok)
        agg["dir_n"] += len(dirs_ok)
        order_ok = m["glyph_exact"] and ranks == sorted(ranks) and -1 not in ranks
        agg["order_exact_e2e"] += order_ok
        full = bool(order_ok and all(dirs_ok))
        agg["full"] += full
        # Stage C alone, on the GT partition
        lab = fill_unassigned(C, gt)
        rg = gen.predict_contours(C, labels=lab)
        gt_ids = [k for k in range(int(lab.max()) + 1) if (lab == k).any()]
        gt_ranks = [gt_ids[k] for k in rg["_order"]]
        agg["gtp_order"] += gt_ranks == sorted(gt_ranks)
        for pos, k in enumerate(rg["_order"]):
            ours = np.array(rg["medians"][pos], float)
            gm = r["medians"][gt_ids[k]]
            agg["gtp_dir_ok"] += median_distance(ours, gm) <= median_distance(ours[::-1], gm)
            agg["gtp_dir_n"] += 1
        rows.append(dict(char=r["char"], pred=dict(strokes=res["strokes"], medians=res["medians"]),
                         gt=dict(strokes=r["strokes_gt"], medians=[m_.astype(int).tolist() for m_ in r["medians"]]),
                         partition_ok=bool(m["glyph_exact"]), order_ok=bool(order_ok), full=full,
                         strokes_exact=m["exact"], n_gt=m["n_gt"], n_pred=m["n_pred"], miou=m["miou"]))
    G = max(1, agg["glyphs"])
    ui = np.array(agg["union_iou"])
    summary = {
        "glyphs": agg["glyphs"],
        "A_stroke_exact": agg["stroke_exact"] / max(1, agg["strokes"]),
        "A_glyph_partition_exact": agg["glyph_part"] / G,
        "A_union_iou_mean": float(ui.mean()),
        "A_union_iou_min": float(ui.min()),
        "B_median_dist_mean": float(np.mean(agg["med_dist"])),
        "C_dir_acc_e2e": agg["dir_ok"] / max(1, agg["dir_n"]),
        "C_order_exact_e2e": agg["order_exact_e2e"] / G,
        "C_order_exact_gt_partition": agg["gtp_order"] / G,
        "C_dir_acc_gt_partition": agg["gtp_dir_ok"] / max(1, agg["gtp_dir_n"]),
        "full_glyph_correct": agg["full"] / G,
    }
    return summary, rows


def _union(loops_per_stroke):
    out = None
    for loops in loops_per_stroke:
        m = render(loops, 256)
        out = m if out is None else np.maximum(out, m)
    return out


PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#9a6324", "#469990",
           "#808000", "#000075", "#e6beff", "#aaffc3", "#ffd8b1", "#800000", "#fabed4"]


def _svg(data: dict) -> str:
    parts = ['<svg viewBox="0 0 1024 1024" class="g"><g transform="scale(1,-1) translate(0,-900)">']
    for i, d in enumerate(data["strokes"]):
        parts.append(f'<path d="{d}" fill="{PALETTE[i % len(PALETTE)]}" fill-opacity="0.55" stroke="#222" stroke-width="2"/>')
    for i, m in enumerate(data["medians"]):
        pts = " ".join(f"{x},{y}" for x, y in m)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#000" stroke-width="10" marker-end="url(#arr)"/>')
        x, y = m[0]
        parts.append(f'<circle cx="{x}" cy="{y}" r="16" fill="#000"/>')
        parts.append(f'<text x="{x}" y="{-y}" transform="scale(1,-1)" font-size="44" fill="#fff" stroke="#000" '
                     f'stroke-width="8" paint-order="stroke" text-anchor="middle" dy="-22">{i + 1}</text>')
    parts.append("</g></svg>")
    return "".join(parts)


# Sidebar where any hanzi-writer JSON file can be dropped (anywhere on the page) and animated.
# Self-contained: its own styles and script, only needs the hanzi-writer script on the page.
SIDEBAR = r"""
<style>
#jsb{position:fixed;top:0;right:0;bottom:0;width:280px;box-sizing:border-box;padding:12px;overflow-y:auto;z-index:50;
  background:var(--card,#fff);color:var(--fg,#111);border-left:1px solid var(--line,#ddd);font:13px system-ui;
  display:flex;flex-direction:column;gap:10px}
body.jsb-open{margin-right:296px}
body:not(.jsb-open) #jsb{display:none}
#jsb-tab{position:fixed;top:12px;right:12px;z-index:51;font:13px system-ui;padding:4px 8px;cursor:pointer}
body.jsb-open #jsb-tab{right:292px}
#jsb h2{font-size:15px;margin:0}
#jsb-drop{border:2px dashed var(--line,#ccc);border-radius:8px;padding:14px 8px;text-align:center;cursor:pointer;opacity:.8}
#jsb-drop.over,body.jsb-drag #jsb-drop{border-color:#4363d8;background:rgba(67,99,216,.12);opacity:1}
#jsb-stage{width:256px;height:256px;background:#fff;border-radius:6px;align-self:center}
#jsb-info{min-height:1.2em;word-break:break-all} #jsb-info.err{color:#d33}
#jsb-ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
#jsb-hist{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:2px}
#jsb-hist button{width:100%;text-align:left;cursor:pointer}
</style>
<button id="jsb-tab" title="toggle the JSON sidebar">JSON viewer</button>
<aside id="jsb">
  <h2>Drop a hanzi-writer JSON</h2>
  <div id="jsb-drop">Drop a .json file anywhere on the page<br><small>or click to choose one</small></div>
  <input id="jsb-file" type="file" accept=".json,application/json" hidden>
  <div id="jsb-stage"></div>
  <div id="jsb-info">Nothing loaded yet.</div>
  <div id="jsb-ctl">
    <button id="jsb-play">Replay</button>
    <label><input id="jsb-loop" type="checkbox"> loop</label>
    <label>speed <input id="jsb-speed" type="range" min="0.25" max="4" step="0.25" value="1" style="width:70px"></label>
  </div>
  <div><small>Loaded this session</small><ul id="jsb-hist"></ul></div>
</aside>
<script>
(()=>{
  const $=id=>document.getElementById(id);
  const files=[]; let cur=null, writer=null, run=0;
  try{ if(localStorage.getItem('jsb-open')!=='0') document.body.classList.add('jsb-open'); }
  catch(e){ document.body.classList.add('jsb-open'); }
  $('jsb-tab').onclick=()=>{
    const open=document.body.classList.toggle('jsb-open');
    try{ localStorage.setItem('jsb-open',open?'1':'0'); }catch(e){}
  };
  const info=(msg,err)=>{ const el=$('jsb-info'); el.textContent=msg; el.classList.toggle('err',!!err); };

  function play(){
    if(!cur) return;
    const id=++run, stage=$('jsb-stage'); stage.innerHTML='';
    writer=HanziWriter.create(stage,cur.name,{width:256,height:256,padding:8,showOutline:true,
      strokeAnimationSpeed:+$('jsb-speed').value,delayBetweenStrokes:300/+$('jsb-speed').value,
      charDataLoader:(c,onLoad)=>onLoad(cur.data)});
    writer.animateCharacter({onComplete:()=>{ if($('jsb-loop').checked&&id===run) setTimeout(()=>id===run&&play(),800); }});
  }
  function show(entry){
    cur=entry; info(entry.name+' — '+entry.data.strokes.length+' strokes'); play();
    document.body.classList.add('jsb-open');
  }
  function addHistory(entry){
    files.unshift(entry);
    const li=document.createElement('li'), b=document.createElement('button');
    b.textContent=entry.name+' ('+entry.data.strokes.length+')'; b.onclick=()=>show(entry);
    li.appendChild(b); $('jsb-hist').prepend(li);
  }
  function load(file){
    if(!file) return;
    file.text().then(txt=>{
      let d; try{ d=JSON.parse(txt); }catch(e){ throw new Error('not valid JSON'); }
      if(!Array.isArray(d.strokes)||!Array.isArray(d.medians)) throw new Error('missing "strokes" / "medians" arrays');
      if(d.strokes.length!==d.medians.length) throw new Error(d.strokes.length+' strokes but '+d.medians.length+' medians');
      const entry={name:file.name.replace(/\.json$/i,''),data:d};
      addHistory(entry); show(entry);
    }).catch(e=>info(file.name+': '+e.message,true));
  }

  $('jsb-play').onclick=play;
  $('jsb-speed').onchange=play;
  $('jsb-loop').onchange=()=>{ if($('jsb-loop').checked) play(); };
  $('jsb-drop').onclick=()=>$('jsb-file').click();
  $('jsb-file').onchange=e=>{ [...e.target.files].forEach(load); e.target.value=''; };

  // the whole window is a drop target, so files can be dropped at any time
  let depth=0;
  const hasFiles=e=>e.dataTransfer&&[...e.dataTransfer.types].includes('Files');
  addEventListener('dragenter',e=>{ if(hasFiles(e)){ depth++; document.body.classList.add('jsb-drag','jsb-open'); } });
  addEventListener('dragleave',()=>{ if(--depth<=0){ depth=0; document.body.classList.remove('jsb-drag'); } });
  addEventListener('dragover',e=>{ if(hasFiles(e)) e.preventDefault(); });
  addEventListener('drop',e=>{
    if(!hasFiles(e)) return;
    e.preventDefault(); depth=0; document.body.classList.remove('jsb-drag');
    [...e.dataTransfer.files].forEach(load);
  });
})();
</script>
"""


def write_preview(rows: list[dict], summary: dict, path: str) -> None:
    cards = []
    for r in rows:
        badge = "ok" if r["full"] else ("order/dir" if r["partition_ok"] else "cut")
        cards.append(
            f'<div class="card {"good" if r["full"] else "bad"}"><div class="hd"><b>{html.escape(r["char"])}</b>'
            f'<span>{badge}</span><button data-c="{html.escape(r["char"])}">animate</button></div>'
            f'<div class="row"><div><small>predicted</small>{_svg(r["pred"])}</div>'
            f'<div><small>ground truth</small>{_svg(r["gt"])}</div></div>'
            f'<div class="anim" id="a-{html.escape(r["char"])}"></div></div>')
    data = {r["char"]: r["pred"] for r in rows}
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stroke Preview</title>
<script src="https://cdn.jsdelivr.net/npm/hanzi-writer@3.5.0/dist/hanzi-writer.min.js"></script>
<style>
:root{{--bg:#fafafa;--fg:#111;--card:#fff;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--card:#222;--line:#444}}}}
body{{background:var(--bg);color:var(--fg);font:14px system-ui;margin:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}}
.card.bad{{border-color:#d33}} .hd{{display:flex;gap:8px;align-items:center}} .hd b{{font-size:22px}}
.hd span{{flex:1;opacity:.7}} .row{{display:flex;gap:6px}} .row>div{{flex:1}}
svg.g{{width:100%;background:#fff;border-radius:4px}} .anim:not(:empty){{background:#fff;border-radius:4px;margin-top:6px;width:200px}} pre{{white-space:pre-wrap}}
</style></head><body>
<h1>Stroke generator preview</h1><pre>{html.escape(json.dumps(summary, indent=1))}</pre>
<svg width="0" height="0"><defs><marker id="arr" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="#000"/></marker></defs></svg>
<div class="grid">{"".join(cards)}</div>
<script>
const DATA={json.dumps(data, ensure_ascii=False)};
document.querySelectorAll('button[data-c]').forEach(b=>b.onclick=()=>{{
  const c=b.dataset.c, el=document.getElementById('a-'+c); el.innerHTML='';
  const w=HanziWriter.create(el,c,{{width:200,height:200,padding:5,showOutline:true,
    charDataLoader:(ch,onLoad)=>onLoad(DATA[ch])}});
  w.animateCharacter();
}});
</script>{SIDEBAR}</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


def records_for_chars(chars: str) -> list[dict]:
    """Labelled records for specific characters, built straight from the font glyphs + hw data."""
    from .build_dataset import GLYPHS, process

    man = json.load(open(os.path.join(GLYPHS, "manifest.json"), encoding="utf-8"))["characters"]
    in_split = {}
    for name in ("train", "val", "test"):
        for r in load_split(name):
            in_split[r["char"]] = name
    recs = []
    for c in dict.fromkeys(chars):
        if c not in man or not man[c]["fonts"]:
            print(f"{c}: no font glyph for this character, skipped")
            continue
        _, rec, status = process((c, man[c]), min_coverage=0.8)
        if rec is None:
            print(f"{c}: no exact ground truth ({status}), skipped")
            continue
        split = in_split.get(c, "none")
        note = "  (seen in training: results are optimistic)" if split == "train" else ""
        print(f"{c}: ground truth is cut from the {rec['font']} outline, split {split}{note}")
        if rec["coverage"] < 0.97:
            print(f"{c}: ground truth leaves {1 - rec['coverage']:.0%} of the outline in no stroke; "
                  "those segments are ignored when scoring")
        recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--chars", help="evaluate only these characters, e.g. --chars 永林森")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preview", type=int, default=60, help="glyphs in the HTML preview (failures first)")
    args = ap.parse_args()
    if args.chars:
        recs = records_for_chars(args.chars)
        if not recs:
            raise SystemExit("nothing to evaluate")
        tag = "chars"
    else:
        recs = load_split(args.split)
        tag = args.split
    if args.limit:
        recs = recs[: args.limit]
    gen = StrokeGenerator()
    summary, rows = evaluate(gen, recs)
    print(json.dumps(summary, indent=1))
    if args.chars:
        for r in rows:
            cut = "ok" if r["partition_ok"] else "WRONG"
            print(f"{r['char']}: cut {cut} ({r['strokes_exact']}/{r['n_gt']} strokes exact, "
                  f"{r['n_pred']} predicted, segment IoU {r['miou']:.2f}), "
                  f"order {'ok' if r['order_ok'] else 'WRONG'}, fully correct: {r['full']}")
            if not r["partition_ok"]:
                print("   (order is only scored as ok when the cut is exactly right)")
    out = os.path.join(ROOT, "outputs")
    os.makedirs(out, exist_ok=True)
    json.dump(summary, open(os.path.join(out, f"eval_{tag}.json"), "w"), indent=1)
    bad = [r for r in rows if not r["full"]]
    good = [r for r in rows if r["full"]]
    pick = bad[: args.preview // 2] + good[: args.preview - min(len(bad), args.preview // 2)]
    path = os.path.join(out, "preview_chars.html" if args.chars else "preview.html")
    write_preview(pick, summary, path)
    print("preview:", path)


if __name__ == "__main__":
    main()
