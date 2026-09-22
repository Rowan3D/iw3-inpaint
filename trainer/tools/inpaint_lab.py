r"""
Inpaint Lab -- try an inpaint model on a real video, quickly, in isolation.

    inpaint-lab.bat                 (or: python tools\inpaint_lab.py)

Nothing here goes through iw3's GUI, its settings file or its job pipeline. It
imports iw3 only for the two pieces that are the thing under test -- the depth
model and the forward warp -- and it never writes to any of iw3's files.

What it is for
--------------
Judging an inpaint model in iw3 is slow and indirect: convert a clip, find the
right moment, squint, change one setting, convert again. And iw3 gives the model
a mask a tenth the width of the one it was trained on, so for a long time every
model looked identical. This renders the same twelve frames through several
pipelines at once and puts them side by side:

    warp only        the synthesised eye with no inpainting -- the damage
    iw3              iw3's warp, iw3's own mask (gen_mask2)
    iw3+patch        iw3's warp, our stretch mask (ntrainer.iw3_mask)
    dibr             our own warp and hole mask (ntrainer.geometry.dibr_warp)

Pick any subset, pick a model for each, flick between them on one keystroke.
The point is that the *only* thing differing between two tiles is the one thing
you changed.

Output goes to <gui>\data\lab\<run>\ as PNGs -- nothing is written anywhere
else, and a run is deleted by deleting its folder.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from os import path
from urllib.parse import parse_qs, urlparse

_HERE = path.dirname(path.abspath(__file__))
_GUI = path.dirname(_HERE)
_ROOT = path.dirname(_GUI)


def _config():
    """Where nunif and its Python actually are, as the GUI already resolved it.

    Guessing from the folder layout is not enough: this GUI is often a copy
    living somewhere else entirely, with nunif on another drive. The GUI stores
    the answer, and `app.config.paths()` fills in everything left blank -- in
    particular python_exe, which is usually empty in config.json because it is
    derived from nunif_dir. Re-deriving it here would be a second copy of that
    logic, free to drift; borrow theirs and fall back to the raw file.
    """
    try:
        sys.path.insert(0, _GUI)
        from app.config import paths
        p = paths()
        return {"nunif_dir": p.get("nunif_dir", ""), "python_exe": p.get("python_exe", ""),
                "models_dir": p.get("models_dir", "")}
    except Exception:                                               # noqa: BLE001
        try:
            with open(path.join(_GUI, "config.json"), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}


CONFIG = _config()

for _p in (_GUI, CONFIG.get("nunif_dir") or "",
           path.join(path.dirname(_ROOT), "nunif"), _ROOT):
    if _p and path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def _reexec_under_nunif_python():
    """torch lives in nunif's Python, not necessarily the one that started us.

    The main GUI gets away with any interpreter because it shells out for the
    heavy work; this tool runs the model in-process, so it has to *be* that
    interpreter. Re-exec once, guarded, rather than making the user care.
    """
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass
    exe = CONFIG.get("python_exe") or ""
    if (not exe or not path.isfile(exe) or os.environ.get("NT_LAB_REEXEC")
            or path.normcase(path.abspath(exe)) == path.normcase(path.abspath(sys.executable))):
        return
    os.environ["NT_LAB_REEXEC"] = "1"
    print(f"note: torch is not in this Python, switching to {exe}", flush=True)
    os.execv(exe, [exe, path.abspath(__file__)] + sys.argv[1:])

LAB_DIR = path.join(_GUI, "data", "lab")
VIDEO_EXT = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts", ".wmv", ".flv")

PIPELINES = ("none", "iw3", "iw3+patch", "dibr")
DEPTH_MODELS = ("Any_V3_Mono", "Any_V3_Large", "Any_B", "Any_L", "Any_S",
                "Distill_Any_B", "VDA_Stream_L", "ZoeD_N")


# ---- lazy heavy imports -------------------------------------------------
# Kept out of module import so that --help and the page load instantly, and so
# a torch/av problem is reported in the browser rather than on a black console.
_torch = None


def _load_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def read_frames(video, start_sec, count, max_width=0):
    """`count` consecutive frames from `start_sec`, as a (N,3,H,W) float tensor."""
    torch = _load_torch()
    import av
    import numpy as np

    with av.open(video) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if start_sec > 0:
            base = stream.time_base or 1
            container.seek(int(start_sec / float(base)), stream=stream, any_frame=False,
                           backward=True)
        frames, seen = [], 0
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else seen / 24.0
            seen += 1
            if t + 1e-6 < start_sec:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= count:
                break
    if not frames:
        raise RuntimeError(f"no frames decoded from {path.basename(video)} at {start_sec}s "
                           f"-- is the start time past the end of the clip?")
    while len(frames) < count:                      # short tail: hold the last frame
        frames.append(frames[-1])
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    if max_width and x.shape[-1] > max_width:
        import torch.nn.functional as F
        h = int(round(x.shape[-2] * max_width / x.shape[-1]))
        x = F.interpolate(x, size=(h - h % 2, max_width - max_width % 2),
                          mode="bilinear", align_corners=False, antialias=True)
    # every arch here wants a multiple of 8
    ph, pw = (-x.shape[-2]) % 8, (-x.shape[-1]) % 8
    if ph or pw:
        x = x[:, :, :x.shape[-2] - (x.shape[-2] % 8), :x.shape[-1] - (x.shape[-1] % 8)]
    return x.contiguous()


def load_depth(name, device, resolution, edge_dilation, depth_aa, x):
    """Depth for every frame, on `device`.

    `x` must already be on `device`: the depth model is loaded onto it, and a
    CPU frame against CUDA weights is the "Input type (torch.FloatTensor) and
    weight type (torch.cuda.FloatTensor) should be the same" error. Do not go
    looking for a get_device() on the wrapper to decide -- it does not have one,
    and hasattr() quietly answering False is how that happened the first time.
    """
    from iw3.depth_model_factory import create_depth_model
    torch = _load_torch()
    model = create_depth_model(name)
    model.load(gpu=_device_id(device), resolution=resolution, limit_resolution=True)
    model.disable_ema()
    out = []
    for i in range(x.shape[0]):
        d = model.infer(x[i:i + 1], edge_dilation=edge_dilation, depth_aa=depth_aa)
        d = d.squeeze(0) if d.ndim == 4 else d
        out.append(model.minmax_normalize_chw(d).unsqueeze(0))
    depth = torch.cat(out, dim=0).to(device)
    del model
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return depth


def load_inpaint(model_path, device):
    """`model_path` empty -> iw3's stock video model."""
    import ntrainer.models  # noqa: F401  -- registers our architectures
    from nunif.models import load_model
    if not model_path:
        from iw3.inpaint_utils import load_video_inpaint_model
        return load_video_inpaint_model(None, device_id=_device_id(device)).eval()
    model, _ = load_model(model_path)
    return model.to(device).eval()


def _device_id(device):
    torch = _load_torch()
    device = torch.device(device)
    return -1 if device.type == "cpu" else (device.index or 0)


def set_mask_params(grad, run, grow):
    """The knobs the patch ships with, overridable per run from the page.

    This is the same MaskParams the live patch reads, so what you tune here is
    what iw3 will do once you set NTRAINER_MASK_GRAD to match.
    """
    import ntrainer.iw3_mask as iw3_mask
    iw3_mask.MaskParams.max_grad = float(grad)
    iw3_mask.MaskParams.min_run = int(run)
    iw3_mask.MaskParams.grow = int(grow)


def warp_iw3(x, depth, divergence, convergence, patched):
    """iw3's own forward warp. `patched` switches our stretch mask on."""
    from iw3.forward_warp import apply_divergence_forward_warp
    import ntrainer.iw3_mask as iw3_mask
    if patched:
        iw3_mask.install()
    previous = getattr(iw3_mask._state, "on", False)
    iw3_mask._state.on = bool(patched)
    try:
        left, right, left_mask, right_mask = apply_divergence_forward_warp(
            x, depth, divergence, convergence=convergence, synthetic_view="both",
            return_mask=True, width_base=False)
    finally:
        iw3_mask._state.on = previous
    return right, right_mask


def warp_dibr(x, depth, divergence, convergence):
    """Our own DIBR warp: scatter disparity, repair it, resample colour once."""
    from ntrainer.geometry import dibr_warp
    # shift_scale() already uses iw3's single-view convention -- div*0.005*base,
    # exactly what apply_divergence_forward_warp uses per eye -- so the same
    # divergence number means the same pixel shift in both pipelines. Checked,
    # not assumed: a 110px shift at divergence 11.5 on a 1920 frame either way.
    rgb, mask = dibr_warp(x, depth, divergence, convergence,
                          view="right", width_base=False, return_mask=True, fill=True)
    return rgb, mask


def inpaint(model, eye, mask, device, amp=True):
    torch = _load_torch()
    from iw3.dilation import mask_closing
    mask = mask_closing(mask > 0) > 0
    eye = eye.to(device)
    mask = mask.to(device)
    use_amp = amp and torch.device(device).type == "cuda"
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model.infer(eye, mask)
    return out.float().clamp(0, 1).cpu()


def overlay(eye, mask):
    """The eye with the mask painted over it, for seeing what was repainted."""
    torch = _load_torch()
    m = (mask > 0).float()
    tint = torch.tensor([1.0, 0.25, 0.35]).view(1, 3, 1, 1).to(eye.dtype)
    return (eye * (1 - m * 0.65) + tint * (m * 0.65)).clamp(0, 1)


def save_png(t, filename):
    from PIL import Image
    a = (t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
    Image.fromarray(a).save(filename)


# ---- the run ------------------------------------------------------------
class Run:
    def __init__(self, opts):
        self.opts = opts
        self.id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = path.join(LAB_DIR, self.id)
        self.state = {"id": self.id, "step": "starting", "done": False, "error": "",
                      "frames": 0, "tiles": [], "log": [], "seconds": 0.0}
        self.lock = threading.Lock()

    def say(self, message):
        with self.lock:
            self.state["log"].append(message)
            self.state["step"] = message
        print(message, flush=True)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        t0 = time.time()
        try:
            self._render()
            self.state["seconds"] = round(time.time() - t0, 1)
            self.state["step"] = f"done in {self.state['seconds']:.0f}s"
        except Exception as e:                                       # noqa: BLE001
            text = f"{type(e).__name__}: {e}"
            if "out of memory" in text.lower():
                text += ("  --  this renders every frame at full size. Set 'Max width' "
                         "to 1280 under Depth & speed, or drop Frames to 12.")
            elif "should be the same" in text and "cuda" in text.lower():
                text += ("  --  a tensor stayed on the CPU; please report this with the "
                         "line number above.")
            self.state["error"] = text
            self.state["log"].append(self.state["error"])
            traceback.print_exc()
            try:
                torch = _load_torch()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:                                        # noqa: BLE001
                pass
        finally:
            self.state["done"] = True

    def _render(self):
        torch = _load_torch()
        o = self.opts
        os.makedirs(self.dir, exist_ok=True)
        device = "cuda" if (o["gpu"] >= 0 and torch.cuda.is_available()) else "cpu"
        if o["gpu"] >= 0 and device == "cpu":
            self.say("note: no CUDA device visible, running on the CPU (slow)")

        self.say(f"reading {o['frames']} frames from {path.basename(o['video'])} at {o['start']}s")
        x = read_frames(o["video"], o["start"], o["frames"], o["max_width"])
        self.state["size"] = f"{x.shape[-1]}x{x.shape[-2]}"
        set_mask_params(o["mask_grad"], o["mask_run"], o["mask_grow"])
        # everything from here on stays on one device: warping 12 frames of
        # 1920x1008 on the CPU takes longer than the model does on the GPU.
        x = x.to(device)
        self.say(f"depth: {o['depth_model']} at {o['resolution']} ({self.state['size']})")
        depth = load_depth(o["depth_model"], device, o["resolution"],
                           o["edge_dilation"], o["depth_aa"], x)

        models = {}
        tiles = []
        for tile in o["tiles"]:
            pipeline, model_path = tile["pipeline"], tile["model"]
            label = tile["label"]
            self.say(f"{label}: warping")
            if pipeline == "dibr":
                eye, mask = warp_dibr(x, depth, o["divergence"], o["convergence"])
            else:
                eye, mask = warp_iw3(x, depth, o["divergence"], o["convergence"],
                                     patched=(pipeline == "iw3+patch"))
            marked = float((mask > 0).float().mean()) * 100
            if pipeline == "none":
                out = eye.clamp(0, 1).float().cpu()
            else:
                if model_path not in models:
                    self.say(f"{label}: loading {path.basename(model_path) or 'stock model'}")
                    models[model_path] = load_inpaint(model_path, device)
                self.say(f"{label}: inpainting {marked:.2f}% of the frame")
                out = inpaint(models[model_path], eye, mask, device, amp=o["amp"])
            base = label.replace(" ", "_").replace("/", "_").replace("+", "plus")
            marks = overlay(eye.float().cpu(), (mask > 0).float().cpu())
            for i in range(out.shape[0]):
                save_png(out[i], path.join(self.dir, f"{base}_{i:03d}.png"))
                save_png(marks[i], path.join(self.dir, f"{base}_{i:03d}_mask.png"))
            del eye, mask, marks
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
            tiles.append({"label": label, "file": base, "mask_pct": round(marked, 3),
                          "pipeline": pipeline,
                          "model": ("no inpainting" if pipeline == "none" else
                                    (path.basename(model_path) or "stock light_inpaint_v1"))})
            self.state["tiles"] = tiles
            self.state["frames"] = out.shape[0]
        with open(path.join(self.dir, "run.json"), "w", encoding="utf-8") as f:
            json.dump({"opts": o, "tiles": tiles, "size": self.state["size"]}, f, indent=1)
        self.say("done")


# ---- server -------------------------------------------------------------
STATE = {"run": None}


def list_videos(folder):
    out = []
    if not folder or not path.isdir(folder):
        return out
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(VIDEO_EXT):
                full = path.join(root, name)
                try:
                    size = path.getsize(full)
                except OSError:
                    continue
                out.append({"path": full, "name": path.relpath(full, folder),
                            "mb": round(size / 1e6, 1)})
            if len(out) >= 500:
                return out
    return out


def find_models():
    """Every trained/installed .pth we can offer, newest first."""
    seen, out = set(), []
    roots = [path.join(_GUI, "data", "models"), CONFIG.get("models_dir") or ""]
    for root in roots:
        if not root or not path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if not name.endswith(".pth") or ".checkpoint." in name:
                    continue
                full = path.join(dirpath, name)
                # the configured models_dir is often the same folder as the
                # default one, reached by a different path -- dedupe by identity,
                # not by spelling, or every model appears twice in the list.
                key = path.normcase(path.realpath(full))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    mtime = path.getmtime(full)
                except OSError:
                    continue
                out.append({"path": full, "name": path.relpath(full, root), "mtime": mtime})
                if len(out) >= 200:
                    break
    out.sort(key=lambda m: -m["mtime"])
    # two roots can hold different files with the same name; make the list say
    # which is which rather than showing the same label twice.
    counts = {}
    for m in out:
        counts[m["name"]] = counts.get(m["name"], 0) + 1
    for m in out:
        if counts[m["name"]] > 1:
            m["name"] = path.join(path.basename(path.dirname(m["path"])), m["name"])
        m.pop("mtime")
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                                      # quiet
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if url.path == "/api/init":
            return self._send(200, json.dumps({
                "models": find_models(), "pipelines": list(PIPELINES),
                "depth_models": list(DEPTH_MODELS), "gui": _GUI,
                "cuda": _cuda_name()}))
        if url.path == "/api/status":
            run = STATE["run"]
            return self._send(200, json.dumps(run.state if run else {"idle": True}))
        if url.path == "/frame":
            q = parse_qs(url.query)
            run, name = q.get("run", [""])[0], q.get("f", [""])[0]
            fp = path.join(LAB_DIR, path.basename(run), path.basename(name))
            if not fp.endswith(".png") or not path.isfile(fp):
                return self._send(404, b"no", "text/plain")
            with open(fp, "rb") as f:
                return self._send(200, f.read(), "image/png",
                                  {"Cache-Control": "public, max-age=3600"})
        return self._send(404, b"no", "text/plain")

    def do_POST(self):
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._send(400, json.dumps({"error": "bad json"}))
        if url.path == "/api/videos":
            return self._send(200, json.dumps({"videos": list_videos(body.get("folder", ""))}))
        if url.path == "/api/run":
            run = STATE["run"]
            if run and not run.state["done"]:
                return self._send(409, json.dumps({"error": "a run is already going"}))
            try:
                opts = _clean(body)
            except ValueError as e:
                return self._send(400, json.dumps({"error": str(e)}))
            STATE["run"] = Run(opts).start()
            return self._send(200, json.dumps({"id": STATE["run"].id}))
        return self._send(404, json.dumps({"error": "no"}))


def _cuda_name():
    try:
        torch = _load_torch()
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception:                                               # noqa: BLE001
        return ""


def _clean(body):
    video = (body.get("video") or "").strip('"')
    if not video or not path.isfile(video):
        raise ValueError("pick a video first")
    tiles = [t for t in (body.get("tiles") or []) if t.get("pipeline") in PIPELINES]
    if not tiles:
        raise ValueError("pick at least one pipeline")
    for t in tiles:
        t["model"] = (t.get("model") or "").strip('"')
        if t["pipeline"] != "none" and t["model"] and not path.isfile(t["model"]):
            raise ValueError(f"model file not found: {t['model']}")
        t["label"] = t.get("label") or t["pipeline"]
    return {
        "video": video,
        "start": max(0.0, float(body.get("start") or 0)),
        "frames": max(1, min(48, int(body.get("frames") or 12))),
        "divergence": float(body.get("divergence") or 11.5),
        "convergence": float(body.get("convergence") or 0.5),
        "depth_model": body.get("depth_model") or "Any_V3_Mono",
        "resolution": max(128, min(2048, int(body.get("resolution") or 720))),
        "edge_dilation": max(0, min(8, int(body.get("edge_dilation") or 0))),
        "depth_aa": bool(body.get("depth_aa", True)),
        "max_width": max(0, min(4096, int(body.get("max_width") or 0))),
        "mask_grad": max(0.05, min(0.99, float(body.get("mask_grad") or 0.4))),
        "mask_run": max(1, min(16, int(body.get("mask_run") or 3))),
        "mask_grow": max(0, min(24, int(body.get("mask_grow") or 2))),
        "gpu": int(body.get("gpu", 0)),
        "amp": bool(body.get("amp", True)),
        "tiles": tiles,
    }


PAGE = ""        # filled in from inpaint_lab.html next to this file


def _load_page():
    global PAGE
    with open(path.join(_HERE, "inpaint_lab.html"), encoding="utf-8") as f:
        PAGE = f.read()


def free_port(start=8791):
    for port in range(start, start + 40):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port in 8791-8830")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    _reexec_under_nunif_python()
    _load_page()
    os.makedirs(LAB_DIR, exist_ok=True)
    port = args.port or free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Inpaint Lab  {url}")
    print(f"output       {LAB_DIR}")
    print("close this window to stop it")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
