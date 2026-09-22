r"""
Training monitor -- a small web page that watches a model directory while
train.bat is running.

  monitor.bat <model-dir> [--port 8080] [--no-browser]

It reads two things the trainer writes and nothing else, so it never touches
training state and can be started, stopped and restarted at any time:

  <model-dir>/progress.csv       one row per epoch
  <model-dir>/eval/epochNNNN_i.png   [ masked input | mask | prediction | truth ]

The page polls for changes and redraws when a new epoch lands. Every sample of
the selected epoch is shown stacked; picking one from the `sample` dropdown and
dragging the epoch slider plays that same eval crop forward through training,
which is the only honest way to see whether it is improving.

No dependencies: stdlib http.server, and the chart is drawn on a canvas.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from os import path


EVAL_RE = re.compile(r"^epoch(\d+)_(\d+)\.png$")


def read_progress(model_dir):
    p = path.join(model_dir, "progress.csv")
    rows = []
    if not path.exists(p):
        return rows
    try:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                def num(key):
                    v = (r.get(key) or "").strip()
                    try:
                        return float(v)
                    except ValueError:
                        return None
                rows.append({
                    "epoch": int(float(r["epoch"])),
                    "phase": int(float((r.get("phase") or "1").strip() or 1)),
                    "train": num("train_loss"),
                    "eval": num("eval_loss"),
                    "lr": num("lr"),
                    "seconds": num("seconds"),
                    "timestamp": (r.get("timestamp") or "").strip(),
                })
    except Exception as e:                                           # noqa: BLE001
        # a half-written file is normal if we read mid-rewrite; the next poll wins
        print(f"monitor: progress.csv unreadable ({e})", file=sys.stderr)
    rows.sort(key=lambda r: r["epoch"])
    return rows


def read_previews(model_dir):
    """{epoch: {slot: filename}} plus the sorted slot list."""
    d = path.join(model_dir, "eval")
    by_epoch = {}
    slots = set()
    if not path.isdir(d):
        return by_epoch, []
    for name in os.listdir(d):
        m = EVAL_RE.match(name)
        if not m:
            continue
        epoch, slot = int(m.group(1)), int(m.group(2))
        by_epoch.setdefault(epoch, {})[slot] = name
        slots.add(slot)
    return by_epoch, sorted(slots)


def state(model_dir):
    rows = read_progress(model_dir)
    previews, slots = read_previews(model_dir)
    files = {str(e): {str(s): n for s, n in v.items()} for e, v in previews.items()}
    return {
        "model_dir": model_dir,
        "name": path.basename(path.abspath(model_dir)),
        "rows": rows,
        "previews": files,
        "preview_epochs": sorted(previews),
        "slots": slots,
        "running": rows[-1]["timestamp"] if rows else "",
    }


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>ntrainer &mdash; __NAME__</title>
<style>
:root{--bg:#14161a;--panel:#1d2026;--line:#2c313a;--fg:#e7e9ee;--dim:#8b93a3;
      --train:#5aa9ff;--eval:#ffb454;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 ui-sans-serif,system-ui,'Segoe UI',sans-serif}
header{display:flex;align-items:baseline;gap:16px;padding:14px 20px;
       border-bottom:1px solid var(--line)}
h1{font-size:15px;margin:0;font-weight:600}
.dim{color:var(--dim)}
main{padding:20px;display:flex;flex-direction:column;gap:20px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.stats{display:flex;flex-wrap:wrap;gap:24px;margin-bottom:12px}
.stat b{display:block;font-size:20px;font-weight:600}
.stat span{color:var(--dim);font-size:12px}
canvas{width:100%;height:320px;display:block}
.legend{display:flex;gap:18px;font-size:12px;color:var(--dim);margin-top:8px}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:12px}
input[type=range]{flex:1;min-width:180px;accent-color:var(--train)}
select,button{background:#262b33;color:var(--fg);border:1px solid var(--line);
              border-radius:6px;padding:5px 9px;font:inherit}
button{cursor:pointer}
.labels{display:grid;grid-template-columns:repeat(4,1fr);gap:0;font-size:12px;
        color:var(--dim);text-align:center;margin-bottom:4px;
        position:sticky;top:0;z-index:2;background:var(--panel);padding:4px 0}
img{width:100%;display:block;image-rendering:pixelated;border-radius:6px;background:#000}
.empty{color:var(--dim);padding:28px 0;text-align:center}
.shot{margin-bottom:14px}
.shot .cap{font-size:12px;color:var(--dim);margin:0 0 3px 2px}
</style></head><body>
<header><h1>__NAME__</h1><span class="dim" id="sub">waiting for the first epoch&hellip;</span></header>
<main>
  <section class="panel">
    <div class="stats" id="stats"></div>
    <canvas id="chart"></canvas>
    <div class="legend">
      <span><i class="swatch" style="background:var(--train)"></i>train loss</span>
      <span><i class="swatch" style="background:var(--eval)"></i>eval loss</span>
      <span id="scale"></span>
    </div>
  </section>
  <section class="panel">
    <div class="controls">
      <label>epoch <b id="epochLabel">-</b></label>
      <input type="range" id="epochRange" min="0" max="0" value="0">
      <label>sample <select id="slot"></select></label>
      <span class="dim" id="count"></span>
      <label><input type="checkbox" id="follow" checked> follow latest</label>
    </div>
    <div class="labels"><div>masked input</div><div>mask</div><div>prediction</div><div>ground truth</div></div>
    <div id="preview"><div class="empty">no eval previews yet</div></div>
  </section>
</main>
<script>
let S = null, slot = null, epoch = null;
const $ = id => document.getElementById(id);

function fmt(v, d=4){ return v===null||v===undefined ? "&ndash;" : (+v).toFixed(d); }

// Train and eval loss are not the same quantity: train is the average over
// augmented crops during the epoch, eval is a clean deterministic pass. Early on
// they can differ by orders of magnitude, and forcing them onto one axis flattens
// whichever is smaller into a straight line. So: two axes, train on the left in
// blue, eval on the right in orange, each autoscaled to its own range.
function drawChart(){
  const c = $("chart"), dpr = devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  c.width = W*dpr; c.height = H*dpr;
  const g = c.getContext("2d"); g.scale(dpr,dpr); g.clearRect(0,0,W,H);
  const rows = (S && S.rows) || [];
  const tr = rows.filter(r=>r.train!==null), ev = rows.filter(r=>r.eval!==null);
  if(!tr.length && !ev.length){ $("scale").textContent=""; return; }
  const pad = {l:62,r:62,t:12,b:26};
  const xs = rows.map(r=>r.epoch);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0+1);
  const X = e => pad.l + (e-x0)/(x1-x0)*(W-pad.l-pad.r);

  function range(vals){
    if(!vals.length) return null;
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if(hi===lo){ const d = Math.abs(hi)*0.05 || 1e-6; lo-=d; hi+=d; }
    const m = (hi-lo)*0.1;
    return [lo-m, hi+m];
  }
  const rTr = range(tr.map(r=>r.train)), rEv = range(ev.map(r=>r.eval));
  const mapY = r => v => H-pad.b - (v-r[0])/(r[1]-r[0])*(H-pad.t-pad.b);

  g.font="11px ui-sans-serif,system-ui,sans-serif";
  const css = n => getComputedStyle(document.body).getPropertyValue(n);
  // gridlines follow the train axis when present, otherwise the eval axis
  const grid = rTr || rEv;
  g.strokeStyle="#2c313a"; g.lineWidth=1;
  for(let i=0;i<=4;i++){
    const y = Math.round(H-pad.b-(H-pad.t-pad.b)*i/4)+0.5;
    g.beginPath(); g.moveTo(pad.l,y); g.lineTo(W-pad.r,y); g.stroke();
    if(rTr){ g.fillStyle=css("--train");
             g.fillText((rTr[0]+(rTr[1]-rTr[0])*i/4).toPrecision(3), 6, y+4); }
    if(rEv){ g.fillStyle=css("--eval"); g.textAlign="right";
             g.fillText((rEv[0]+(rEv[1]-rEv[0])*i/4).toPrecision(3), W-6, y+4);
             g.textAlign="left"; }
  }
  g.fillStyle="#8b93a3";
  g.fillText("epoch "+x0, pad.l, H-8);
  g.textAlign="right"; g.fillText("epoch "+x1, W-pad.r, H-8); g.textAlign="left";

  // Phase boundaries. Each 200-epoch run after the first restarts the LR
  // schedule on the same weights, so the jump in loss at these lines is the
  // restart, not a regression.
  let prevPhase = rows.length ? rows[0].phase : 1;
  for(const r of rows){
    if(r.phase > prevPhase){
      const x = Math.round(X(r.epoch))+0.5;
      g.save(); g.setLineDash([4,4]); g.strokeStyle="#8b93a355"; g.lineWidth=1;
      g.beginPath(); g.moveTo(x,pad.t); g.lineTo(x,H-pad.b); g.stroke(); g.restore();
      g.fillStyle="#8b93a3";
      g.fillText("phase "+r.phase, x+4, pad.t+11);
      prevPhase = r.phase;
    }
  }

  const line = (pts, color, r) => {
    if(!pts.length || !r) return;
    const Y = mapY(r);
    g.strokeStyle=color; g.lineWidth=2; g.beginPath();
    pts.forEach((p,i)=> i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));
    g.stroke();
    g.fillStyle=color;
    pts.forEach(p=>{ g.beginPath(); g.arc(X(p[0]),Y(p[1]),2.5,0,7); g.fill(); });
  };
  line(tr.map(r=>[r.epoch,r.train]), css("--train"), rTr);
  line(ev.map(r=>[r.epoch,r.eval]), css("--eval"), rEv);
  if(epoch!==null && epoch!==undefined){
    const x = Math.round(X(epoch))+0.5;
    g.strokeStyle="#ffffff33"; g.lineWidth=1;
    g.beginPath(); g.moveTo(x,pad.t); g.lineTo(x,H-pad.b); g.stroke();
  }
  $("scale").textContent = "left axis: train   right axis: eval";
}

function drawStats(){
  const rows = (S&&S.rows)||[];
  const last = rows[rows.length-1];
  const evs = rows.filter(r=>r.eval!==null);
  const best = evs.length ? evs.reduce((a,b)=>b.eval<a.eval?b:a) : null;
  const secs = rows.filter(r=>r.seconds).slice(-5).map(r=>r.seconds);
  const avg = secs.length ? secs.reduce((a,b)=>a+b,0)/secs.length : null;
  $("stats").innerHTML = !last ? "" : `
    <div class="stat"><b>${last.epoch}</b><span>epoch${last.phase>1?" (phase "+last.phase+")":""}</span></div>
    <div class="stat"><b>${fmt(last.train)}</b><span>train loss</span></div>
    <div class="stat"><b>${fmt(evs.length?evs[evs.length-1].eval:null)}</b><span>eval loss</span></div>
    <div class="stat"><b>${best?fmt(best.eval):"&ndash;"}</b><span>best eval${best?" (epoch "+best.epoch+")":""}</span></div>
    <div class="stat"><b>${last.lr!==null?(+last.lr).toExponential(2):"&ndash;"}</b><span>lr</span></div>
    <div class="stat"><b>${avg?avg.toFixed(0)+"s":"&ndash;"}</b><span>per epoch</span></div>`;
  $("sub").textContent = last ? ("last update " + (last.timestamp||"")) : "waiting for the first epoch…";
}

function drawPreview(){
  const eps = (S&&S.preview_epochs)||[];
  const r = $("epochRange");
  r.min = 0; r.max = Math.max(0, eps.length-1);
  if($("follow").checked && eps.length) epoch = eps[eps.length-1];
  let idx = eps.indexOf(epoch);
  if(idx < 0){ idx = eps.length-1; epoch = eps[idx]; }
  r.value = idx < 0 ? 0 : idx;
  $("epochLabel").textContent = epoch!==null && epoch!==undefined ? epoch : "-";

  // Every sample of this epoch is drawn, stacked, so a whole eval pass is one
  // scroll. The dropdown narrows to a single sample rather than hiding the rest.
  const sel = $("slot"), slots = (S&&S.slots)||[];
  if(sel.options.length !== slots.length+1){
    sel.innerHTML = `<option value="all">all</option>` +
                    slots.map(s=>`<option value="${s}">${s}</option>`).join("");
    if(slot===null) slot = "all";
    sel.value = String(slot);
  }
  const files = (S&&S.previews[String(epoch)])||{};
  let keys = Object.keys(files).map(Number).sort((a,b)=>a-b);
  if(slot!=="all" && slot!==null && files[String(slot)]!==undefined) keys = [+slot];
  $("count").textContent = keys.length ? `${keys.length} sample${keys.length>1?"s":""}` : "";
  $("preview").innerHTML = keys.length
    ? keys.map(k=>`<div class="shot"><div class="cap">sample ${k}</div>`+
                  `<img src="eval/${encodeURIComponent(files[String(k)])}" `+
                  `alt="epoch ${epoch} sample ${k}" loading="lazy"></div>`).join("")
    : `<div class="empty">no eval previews yet</div>`;
}

async function poll(){
  try{
    const res = await fetch("api/state", {cache:"no-store"});
    const next = await res.json();
    const changed = !S || JSON.stringify(next.rows.length + "|" + next.preview_epochs.join(","))
                        !== JSON.stringify(S.rows.length + "|" + S.preview_epochs.join(","))
                    || JSON.stringify(next.rows) !== JSON.stringify(S.rows);
    S = next;
    if(changed){ drawStats(); drawChart(); drawPreview(); }
  }catch(e){ /* trainer may be mid-write, or the server stopped */ }
  setTimeout(poll, 3000);
}
$("epochRange").addEventListener("input", e=>{
  const eps = (S&&S.preview_epochs)||[];
  $("follow").checked = false;
  epoch = eps[+e.target.value]; drawPreview(); drawChart();
});
$("slot").addEventListener("change", e=>{
  slot = e.target.value === "all" ? "all" : +e.target.value; drawPreview();
});
$("follow").addEventListener("change", ()=>{ drawPreview(); drawChart(); });
addEventListener("resize", drawChart);
poll();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    model_dir = "."
    server_version = "ntrainer-monitor"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        url = self.path.split("?", 1)[0]
        if url in ("/", "/index.html"):
            name = path.basename(path.abspath(self.model_dir))
            self._send(200, PAGE.replace("__NAME__", name), "text/html; charset=utf-8")
        elif url == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        elif url == "/api/state":
            self._send(200, json.dumps(state(self.model_dir)), "application/json")
        elif url.startswith("/eval/"):
            from urllib.parse import unquote
            name = unquote(url[len("/eval/"):])
            if not EVAL_RE.match(name):          # no traversal, no other files
                self._send(404, "not found", "text/plain")
                return
            p = path.join(self.model_dir, "eval", name)
            if not path.exists(p):
                self._send(404, "not found", "text/plain")
                return
            with open(p, "rb") as f:
                self._send(200, f.read(), "image/png")
        else:
            self._send(404, "not found", "text/plain")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("model_dir", type=str, nargs="?", default=None)
    p.add_argument("--model-dir", dest="model_dir_flag", type=str, default=None)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", type=str, default="127.0.0.1",
                   help="use 0.0.0.0 to reach it from another machine on your LAN")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    base = os.environ.get("NT_CWD") or os.getcwd()
    target = args.model_dir_flag or args.model_dir
    if not target:
        p.error("give the model directory, e.g. monitor.bat models\\test")
    model_dir = target if path.isabs(target) else path.abspath(path.join(base, target))
    if not path.isdir(model_dir):
        print(f"{model_dir} does not exist yet -- start training first, or check the path",
              file=sys.stderr)
        return 2

    Handler.model_dir = model_dir
    port = args.port
    for attempt in range(20):
        try:
            httpd = ThreadingHTTPServer((args.host, port), Handler)
            break
        except OSError:
            port += 1
    else:
        print(f"could not bind a port near {args.port}", file=sys.stderr)
        return 1

    url = f"http://{'localhost' if args.host == '127.0.0.1' else args.host}:{port}/"
    print(f"watching : {model_dir}")
    print(f"monitor  : {url}")
    print("press Ctrl+C to stop (this does not affect training)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
