"""Turn models/results.csv into a readable HTML report: models/results_report.html.

    python make_report.py

Open the HTML file in any browser. Re-run after new experiments. No internet needed.
Also reads models/best_gru.json (which run is exported) and models/cv/*.json (cv.py).
"""
import csv
import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "models" / "results.csv"
BEST_JSON = HERE / "models" / "best_gru.json"
CV_DIR = HERE / "models" / "cv"
OUT_PATH = HERE / "models" / "results_report.html"

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SignSense Results</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b1f27;--muted:#667085;--line:#e4e7ec;
--gru:#2f6fdb;--lstm:#d9822b;--val:#2f6fdb;--test:#2e9e6b;--best:#fff6d6;}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1d2128;--ink:#e8ebf0;--muted:#98a2b3;
--line:#2d333d;--gru:#6b9cf5;--lstm:#f0a35a;--val:#6b9cf5;--test:#58c796;--best:#3a3320;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:17px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 20px}.note{color:var(--muted);font-size:13px;margin:4px 0 12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:16px}
.card,.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.card .k{color:var(--muted);font-size:13px}.card .v{font-size:28px;font-weight:650;margin-top:2px}
.card .s{color:var(--muted);font-size:13px;margin-top:2px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px;margin-bottom:16px}
.toolbar{display:flex;gap:8px;margin:8px 0 12px;flex-wrap:wrap}
button{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:6px 12px;
border-radius:999px;cursor:pointer;font:inherit}button.on{background:var(--ink);color:var(--bg)}
.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:820px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-size:12px;color:var(--muted);font-weight:600;cursor:pointer;user-select:none}
th:hover{color:var(--ink)}tr.best td{background:var(--best)}
.pill{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;font-weight:600;color:#fff}
.pill.gru{background:var(--gru)}.pill.lstm{background:var(--lstm)}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;border:1px solid var(--line);color:var(--muted)}
.tag.exp{border-color:var(--test);color:var(--test);font-weight:600}
.barcell{display:flex;align-items:center;gap:8px;min-width:170px}
.bar{height:9px;border-radius:5px;flex:none}.num{font-variant-numeric:tabular-nums;font-weight:600;min-width:46px}
.legend{display:flex;gap:14px;font-size:13px;color:var(--muted);margin:2px 0 10px}
.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.row{display:grid;grid-template-columns:110px 1fr;gap:10px;align-items:center;margin:10px 0}
.track{position:relative;height:22px;background:var(--bg);border-radius:6px}
.fill{position:absolute;top:5px;height:12px;border-radius:4px;opacity:.9}
.mean{position:absolute;top:1px;width:3px;height:20px;border-radius:2px;background:var(--ink)}
.lbl{font-size:13px}.lbl small{display:block;color:var(--muted)}
.ppl{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px 14px;margin-top:6px}
.ppl .barcell{min-width:0}.pid{font-size:12px;color:var(--muted);min-width:44px;font-variant-numeric:tabular-nums}
</style></head><body><main>
<h1>SignSense: experiment results</h1>
<p class="sub" id="sub"></p>
<div class="cards" id="cards"></div>
<div class="grid2">
 <div class="panel"><h2>Training data</h2>
  <p class="note">Chosen GRU (2x128, dropout 0.3, lr 1e-3), 3 seeds each. Same model, only the training data changes.
  Bar = lowest to highest seed, black tick = mean.</p>
  <div class="legend"><span><i class="dot" style="background:var(--val)"></i>validation</span><span><i class="dot" style="background:var(--test)"></i>test</span></div>
  <div id="data"></div></div>
 <div class="panel"><h2>Cross-validation (all 21 participants)</h2>
  <p class="note">7 folds of 3 people, every participant tested once (cv.py). The honest estimate for a new person.</p>
  <div id="cv"></div></div>
</div>
<div class="grid2">
 <div class="panel"><h2>GRU vs LSTM</h2>
  <p class="note">Average over the 6 size/dropout settings each (64, 128, 256 units; dropout 0.3, 0.5),
  on the original training data and with 2 augmented copies.</p>
  <div class="legend"><span><i class="dot" style="background:var(--val)"></i>validation</span><span><i class="dot" style="background:var(--test)"></i>test</span></div>
  <div id="arch"></div></div>
 <div class="panel"><h2>Learning rate</h2>
  <p class="note">GRU, 128 units, dropout 0.3, original data, 3 seeds each. Bar = lowest to highest seed, black tick = mean.</p>
  <div class="legend"><span><i class="dot" style="background:var(--val)"></i>validation</span><span><i class="dot" style="background:var(--test)"></i>test</span></div>
  <div id="lr"></div></div>
</div>
<div class="panel"><h2>All runs</h2>
 <p class="note">Click a column header to sort. Bars start at 50%. Highlighted row = best validation accuracy.
 Choose models by validation; test is only a final check.</p>
 <div class="toolbar" id="filters"></div>
 <div class="tablewrap"><table id="tbl"><thead></thead><tbody></tbody></table></div></div>
</main>
<script>
const DATA = __DATA__;
const EXPORTED = __EXPORTED__;
const CV = __CV__;
const pct = v => (v*100).toFixed(1)+"%";
const mean = a => a.reduce((s,x)=>s+x,0)/a.length;
const el = (t,c,h)=>{const e=document.createElement(t);if(c)e.className=c;if(h!==undefined)e.innerHTML=h;return e};
const dataLabel = r => r.augment_copies && r.mirror ? "aug ×"+r.augment_copies+" + mirror"
  : r.augment_copies ? "aug ×"+r.augment_copies : r.mirror ? "mirror" : "original";
const isBase = r => r.name.includes("baseline");
const runs = DATA.filter(r=>!isBase(r));            // baseline duplicates gru_2x128_d0.3
runs.forEach(r=>r.data=dataLabel(r)+(r.velocity?" + velocity":""));

// header + cards
const best = runs.reduce((a,b)=>b.val_acc>a.val_acc?b:a);
const exp = runs.find(r=>r.name===EXPORTED);
document.getElementById("sub").textContent = runs.length+" runs · generated __DATE__ · 3 held-out validation and 3 test participants";
const cards = document.getElementById("cards");
const cvTop = CV.length ? CV.reduce((a,b)=>b.mean>a.mean?b:a) : null;
[...(exp?[["Exported model",pct(exp.val_acc)+" / "+pct(exp.test_acc),"validation / test · "+exp.name]]:[]),
 ...(cvTop?[["Cross-validation",pct(cvTop.mean),"mean per person, "+cvTop.name+" (honest estimate)"]]:[]),
 ["Best validation",pct(best.val_acc),best.name],
 ["Runs",runs.length,"GRU "+runs.filter(r=>r.model==="gru").length+" · LSTM "+runs.filter(r=>r.model==="lstm").length]
].forEach(([k,v,s])=>{const c=el("div","card");c.innerHTML=`<div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div>`;cards.append(c)});

// seed-range track (validation or test)
function rangeTrack(g,key,color){const xs=g.map(r=>r[key]);const lo=Math.min(...xs),hi=Math.max(...xs),m=mean(xs);
  const p=x=>((x-0.5)/0.4*100).toFixed(1);
  return `<div class="track" style="margin-bottom:4px"><div class="fill" style="left:${p(lo)}%;width:${Math.max(0.8,p(hi)-p(lo))}%;background:${color}"></div><div class="mean" style="left:${p(m)}%"></div></div>
   <div class="note" style="margin:0 0 6px">${pct(lo)} – ${pct(hi)} · mean ${pct(m)}</div>`}

// training data
const dataBox=document.getElementById("data");
const chosen=runs.filter(r=>r.model==="gru"&&r.units===128&&r.dropout===0.3&&r.lr===0.001&&r.min_hand_frames_train===0&&!r.velocity);
["original","mirror","aug ×2","aug ×2 + mirror"].forEach(lab=>{
  const g=chosen.filter(r=>r.data===lab); if(!g.length) return;
  const row=el("div","row");
  row.innerHTML=`<div class="lbl">${lab}<small>${g.length} seeds</small></div><div>${rangeTrack(g,"val_acc","var(--val)")}${rangeTrack(g,"test_acc","var(--test)")}</div>`;
  dataBox.append(row)});

// cross-validation
const cvBox=document.getElementById("cv");
if(!CV.length) cvBox.append(el("p","note","No cv.py results yet."));
CV.forEach(c=>{
  const w=((c.mean-0.4)/0.55*100).toFixed(1);
  const row=el("div","row");
  row.innerHTML=`<div class="lbl">${c.name}<small>${c.people} people</small></div>
   <div><div class="barcell"><div class="bar" style="width:${w}%;background:var(--test)"></div><span class="num">${pct(c.mean)}</span></div>
   <div class="note" style="margin:2px 0 0">mean per person · worst ${pct(c.worst)} · best ${pct(c.best)}</div></div>`;
  cvBox.append(row)});
if(cvTop){
  cvBox.append(el("p","note","Per person, "+cvTop.name+" (bars start at 40%):"));
  const ppl=el("div","ppl");
  Object.entries(cvTop.per_person).sort((a,b)=>a[1]-b[1]).forEach(([pid,acc])=>{
    const d=el("div","barcell");
    d.innerHTML=`<span class="pid">${pid}</span><div class="bar" style="width:${Math.max(2,(acc-0.4)/0.55*60)}px;background:var(--test)"></div><span class="num">${pct(acc)}</span>`;
    ppl.append(d)});
  cvBox.append(ppl);
  cvBox.append(el("p","note","The spread between people is much larger than any difference between model settings."))}

// GRU vs LSTM
const grid = runs.filter(r=>/^(gru|lstm)_2x(64|128|256)_d0\.[35](_aug2)?$/.test(r.name));
const arch = document.getElementById("arch");
[["gru",0],["lstm",0],["gru",2],["lstm",2]].forEach(([m,a])=>{
  const g = grid.filter(r=>r.model===m&&r.augment_copies===a); if(!g.length) return;
  const v=mean(g.map(r=>r.val_acc)), t=mean(g.map(r=>r.test_acc));
  const w=x=>((x-0.5)/0.4*100).toFixed(1);
  const row=el("div","row");
  row.innerHTML=`<div class="lbl"><span class="pill ${m}">${m.toUpperCase()}</span><small>${a?"augmented":"original"} · ${g.length} runs</small></div>
   <div><div class="barcell"><div class="bar" style="width:${w(v)}%;background:var(--val)"></div><span class="num">${pct(v)}</span></div>
   <div class="barcell" style="margin-top:4px"><div class="bar" style="width:${w(t)}%;background:var(--test)"></div><span class="num">${pct(t)}</span></div></div>`;
  arch.append(row)});
arch.append(el("p","note","Differences of 1–2 points are small next to the noise (each set has only 3 people)."));

// learning rate
const lrRuns = runs.filter(r=>r.model==="gru"&&r.units===128&&r.dropout===0.3&&r.min_hand_frames_train===0&&r.data==="original");
const lrBox = document.getElementById("lr");
[...new Set(lrRuns.map(r=>r.lr))].sort((a,b)=>a-b).forEach(lr=>{
  const g=lrRuns.filter(r=>r.lr===lr);
  const row=el("div","row");
  row.innerHTML=`<div class="lbl">${lr}<small>${g.length} seeds</small></div><div>${rangeTrack(g,"val_acc","var(--val)")}${rangeTrack(g,"test_acc","var(--test)")}</div>`;
  lrBox.append(row)});

// table
const cols=[["name","Run"],["model","Model"],["data","Training data"],["units","Units"],["dropout","Dropout"],["lr","LR"],["seed","Seed"],
 ["val_acc","Validation"],["test_acc","Test"],["epochs_run","Epochs"],["train_seconds","Time (s)"]];
let sortKey="val_acc",sortDir=-1,filter="all";
const thead=document.querySelector("#tbl thead"),tbody=document.querySelector("#tbl tbody");
const tr=el("tr");cols.forEach(([k,l])=>{const th=el("th","",l);th.onclick=()=>{sortDir=(sortKey===k)?-sortDir:-1;sortKey=k;draw()};tr.append(th)});thead.append(tr);
const fb=document.getElementById("filters");
const keep={all:()=>true,gru:r=>r.model==="gru",lstm:r=>r.model==="lstm",
  original:r=>r.data==="original",augmented:r=>r.data!=="original"};
[["all","All"],["gru","GRU"],["lstm","LSTM"],["original","Original data"],["augmented","Augmented / mirrored"]].forEach(([k,l])=>{
  const b=el("button",k==="all"?"on":"",l);b.onclick=()=>{filter=k;[...fb.children].forEach(c=>c.classList.remove("on"));b.classList.add("on");draw()};fb.append(b)});
function bar(x,color){return `<div class="barcell"><div class="bar" style="width:${Math.max(2,(x-0.5)/0.4*110)}px;background:${color}"></div><span class="num">${pct(x)}</span></div>`}
function draw(){
  tbody.innerHTML="";
  runs.filter(keep[filter])
   .sort((a,b)=>(a[sortKey]>b[sortKey]?1:-1)*sortDir)
   .forEach(r=>{const t=el("tr",r===best?"best":"");
     t.innerHTML=`<td>${r.name}${r.name===EXPORTED?' <span class="tag exp">exported</span>':''}</td><td><span class="pill ${r.model}">${r.model.toUpperCase()}</span></td><td><span class="tag">${r.data}</span></td><td>${r.units}</td><td>${r.dropout}</td><td>${r.lr}</td><td>${r.seed}</td>
      <td>${bar(r.val_acc,"var(--val)")}</td><td>${bar(r.test_acc,"var(--test)")}</td><td>${r.epochs_run}</td><td>${r.train_seconds}</td>`;
     tbody.append(t)});
}
draw();
</script></body></html>"""


def load_rows():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "name": r["name"], "model": r["model"], "layers": int(r["layers"]),
                "units": int(r["units"]), "dropout": float(r["dropout"]), "lr": float(r["lr"]),
                "seed": int(r["seed"]), "epochs_run": int(r["epochs_run"]),
                "min_hand_frames_train": int(r["min_hand_frames_train"]),
                "val_acc": float(r["val_acc"]), "test_acc": float(r["test_acc"]),
                "train_seconds": int(float(r["train_seconds"])),
                "augment_copies": int(r.get("augment_copies") or 0),
                "mirror": int(r.get("mirror") or 0),
                "velocity": int(r.get("velocity") or 0),
            })
    return rows


def exported_name():
    if not BEST_JSON.exists():
        return None
    return json.loads(BEST_JSON.read_text(encoding="utf-8"))["args"].get("name")


def load_cv():
    """One entry per cv.py setup: mean / worst / best test accuracy per person."""
    per = {}
    for js in sorted(CV_DIR.glob("cv_*_f*.json")):
        cfg = js.stem[3:].rsplit("_f", 1)[0]
        info = json.loads(js.read_text(encoding="utf-8"))
        for p, r in info.get("per_participant", {}).get("test", {}).items():
            per.setdefault(cfg, {})[p] = r["acc"]
    out = []
    for cfg, people in per.items():
        accs = list(people.values())
        out.append({"name": cfg, "people": len(accs), "mean": sum(accs) / len(accs),
                    "worst": min(accs), "best": max(accs), "per_person": people})
    return out


def main():
    rows = load_rows()
    html = (TEMPLATE.replace("__DATA__", json.dumps(rows))
            .replace("__EXPORTED__", json.dumps(exported_name()))
            .replace("__CV__", json.dumps(load_cv()))
            .replace("__DATE__", date.today().isoformat()))
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
