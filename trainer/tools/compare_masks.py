r"""
Step 1 verification: old iw3 warp/mask vs the new analytic warp/mask.

Run from the repo root via ..\test_warp.bat, or manually:

    call setenv.bat
    pushd nunif
    python ..\New_Trainer\tools\compare_masks.py --input ..\New_Trainer\test_images

Produces, per input image, a labelled comparison sheet plus a zoomed crop of
the busiest mask region, and prints a run-length table (a broken-up mask shows
as many short runs per row; a correct mask shows few long ones).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from os import path

_HERE = path.dirname(path.dirname(path.abspath(__file__)))     # .../New_Trainer
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_NUNIF = path.join(path.dirname(_HERE), "nunif")
if path.isdir(_NUNIF) and _NUNIF not in sys.path:
    sys.path.insert(0, _NUNIF)

import torch                                                   # noqa: E402
import torch.nn.functional as F                                # noqa: E402
from PIL import Image                                          # noqa: E402

from ntrainer.geometry import dibr_warp, shift_scale, depth_edge_width  # noqa: E402
from ntrainer.pipeline import MaskConfig, prepare_depth, make_mask   # noqa: E402
from ntrainer import viz                                             # noqa: E402


IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
def load_images(input_dir, limit):
    files = []
    for fn in sorted(os.listdir(input_dir)):
        if path.splitext(fn)[1].lower() in IMG_EXT:
            files.append(path.join(input_dir, fn))
    if not files:
        raise SystemExit(f"no images found in {input_dir}")
    return files[:limit] if limit else files


def resize_to_width(im, width):
    if width is None or im.width == width:
        return im
    h = max(1, round(im.height * width / im.width))
    h -= h % 2
    return im.resize((width, h), Image.LANCZOS)


def run_stats(mask):
    """(1,1,H,W) binary -> runs per row, mean run length, coverage %"""
    m = (mask > 0.5).float()
    starts = (m[..., 1:] - m[..., :-1]).clamp_min(0).sum().item() + m[..., 0].sum().item()
    total = m.sum().item()
    H = m.shape[-2]
    # row-to-row jitter of the mask's inner (left) boundary -- how jagged it is
    W = m.shape[-1]
    idx = torch.arange(W, device=m.device, dtype=torch.float32).view(1, 1, 1, W)
    first = torch.where(m > 0.5, idx, torch.full_like(idx, float(W))).amin(dim=-1)[0, 0]
    last = torch.where(m > 0.5, idx, torch.full_like(idx, -1.0)).amax(dim=-1)[0, 0]
    valid = first < W
    both = valid[1:] & valid[:-1]
    jitter = ((first[1:] - first[:-1]).abs()[both].mean().item()
              if both.any() else 0.0)
    outer = ((last[1:] - last[:-1]).abs()[both].mean().item()
             if both.any() else 0.0)
    return {
        "coverage_pct": round(100.0 * total / m.numel(), 3),
        "runs_per_row": round(starts / H, 2),
        "mean_run_px": round(total / starts, 2) if starts > 0 else 0.0,
        "edge_jitter_px": round(jitter, 2),
        "outer_jitter_px": round(outer, 2),
    }


def busiest_crop(mask, size=384):
    """Find the size x size window with the most mask, return (top, left)."""
    m = (mask > 0.5).float()
    H, W = m.shape[-2:]
    s = min(size, H, W)
    pooled = F.avg_pool2d(m, kernel_size=s, stride=max(1, s // 8))
    _, idx = pooled.flatten().max(0)
    pw = pooled.shape[-1]
    st = max(1, s // 8)
    i = (idx.item() // pw) * st
    j = (idx.item() % pw) * st
    return min(i, H - s), min(j, W - s), s


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", type=str, default=path.join(_HERE, "test_images"))
    p.add_argument("--output", type=str, default=path.join(_HERE, "out", "step1"))
    p.add_argument("--model-type", type=str, default="Any_V3_Mono",
                   help="iw3 depth model (Any_V3_Mono = Depth Anything v3, Any_B, ZoeD_Any_L, ...)")
    p.add_argument("--resolution", type=int, default=784,
                   help="depth model internal short side. DA3's own default is 392, which "
                        "makes every depth pixel a ~2.75px block at 1920 wide; 784 halves "
                        "that and is the single biggest lever on mask sharpness")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--width", type=int, default=1920,
                   help="resize test images to this width (native-scale test)")
    p.add_argument("--divergence", type=float, nargs="+", default=[2.0, 5.0, 8.0, 12.0, 16.0])
    p.add_argument("--convergence", type=float, default=0.5)
    p.add_argument("--edge-dilation", type=int, default=0,
                   help="depth model edge dilation; >0 pushes the depth edge OUTWARD "
                        "past the real silhouette, so keep it at 0 for accurate masks")
    p.add_argument("--anchor", type=str, default="edge", choices=["edge", "background"],
                   help="where each mask band starts: on the silhouette, or on the "
                        "first pure-background column")
    p.add_argument("--limit", type=int, default=0, help="max images (0 = all)")
    p.add_argument("--tile-width", type=int, default=560, help="sheet tile width")
    p.add_argument("--no-refine", action="store_true", help="skip RGB-guided depth edge refinement")
    p.add_argument("--guided-radius", type=int, default=8)
    p.add_argument("--snap-iter", type=int, default=0,
                   help="toggle-contrast passes on the depth (0 = off)")
    p.add_argument("--bilateral-radius", type=int, default=3)
    p.add_argument("--bilateral-iter", type=int, default=1)
    p.add_argument("--ramp-px", type=int, default=0,
                   help="depth-edge ramp width to bridge; 0 = auto from image width")
    p.add_argument("--depth-step", type=float, default=0.02,
                   help="depth drop across --edge-window columns that counts as a real "
                        "discontinuity")
    p.add_argument("--edge-window", type=int, default=0,
                   help="columns the tear test is measured over; 0 = auto from the "
                        "measured depth edge ramp width")
    p.add_argument("--tear-rows", type=int, default=2,
                   help="vertical radius the tear test is pooled over (0 = off)")
    p.add_argument("--min-band-px", type=float, default=1.0)
    p.add_argument("--occlusion-frac", type=float, default=0.3)
    p.add_argument("--edge-frac", type=float, default=0.1,
                   help="with --anchor edge: how far below the foreground plateau still "
                        "counts as on the object, as a fraction of that edge's height. "
                        "Raise if a gap remains, lower if the mask eats into the object")
    p.add_argument("--anchor-px", type=int, default=0,
                   help="how far back the silhouette search reaches (0 = same as ramp-px)")
    p.add_argument("--despike", type=int, default=2, choices=[0, 1, 2],
                   help="median guard on the depth: 0 off, 1 horizontal, 2 horizontal+"
                        "vertical. Removes single-pixel depth spikes that notch the mask "
                        "into the object, and straightens the outer profile")
    p.add_argument("--outer-smooth-rows", type=int, default=1,
                   help="rows of vertical median on the mask's outer boundary")
    p.add_argument("--smooth-px", type=int, default=1,
                   help="majority smoothing radius of the final mask (0 = off)")
    p.add_argument("--depth-aa", dest="depth_aa", action="store_true", default=True,
                   help="run iw3's DepthAA anti-alias model on the depth (on by default; "
                        "DA3 output is upsampled from --resolution and is blocky)")
    p.add_argument("--no-depth-aa", dest="depth_aa", action="store_false",
                   help="disable DepthAA")
    p.add_argument("--tear-px", type=float, default=0.0,
                   help="pixel-gap tear criterion; 0 = off. Being a pixel measure it "
                        "scales with divergence and overrides --depth-step at high "
                        "divergence, admitting smooth gradients as tears")
    p.add_argument("--sharpness", type=float, default=0.6,
                   help="require the drop over --edge-window columns to be this fraction "
                        "of the drop over 4x that many (step=1.0, linear slope=0.25)")
    p.add_argument("--skip-old", action="store_true")
    p.add_argument("--dump", type=int, default=0,
                   help="save the RGB + raw depth + refined depth of the busiest crop of "
                        "the first N images to debug.pt, so the mask can be reproduced "
                        "and instrumented off-machine")
    p.add_argument("--dump-size", type=int, default=512)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    from iw3.depth_model_factory import create_depth_model
    from iw3.utils import get_mapper
    from iw3.dilation import mask_closing
    from iw3.forward_warp import nonwarp_mask as old_forward_nonwarp_mask
    from torchvision.transforms import functional as TF

    device = f"cuda:{args.gpu}" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    print(f"device={device}  torch={torch.__version__}")
    if device.startswith("cuda"):
        print(f"gpu={torch.cuda.get_device_name(args.gpu)}")

    depth_model = create_depth_model(args.model_type)
    depth_model.load(gpu=args.gpu, resolution=args.resolution)
    depth_model.disable_ema()
    low = getattr(depth_model.model, "prep_lower_bound", None)
    print(f"depth model={args.model_type}  internal short side={low}px "
          f"(depth is upsampled from this to the full frame)  depth_aa={args.depth_aa}")

    files = load_images(args.input, args.limit)
    print(f"{len(files)} image(s) from {args.input}")

    cfg = MaskConfig.from_args(args)
    print(f"mask: {cfg.describe()}")
    report = []
    dumps = []
    for fi, fn in enumerate(files):
        stem = path.splitext(path.basename(fn))[0]
        im = Image.open(fn).convert("RGB")
        im = resize_to_width(im, args.width)
        W, H = im.size
        print(f"\n[{fi + 1}/{len(files)}] {path.basename(fn)}  {W}x{H}")

        with torch.inference_mode():
            t0 = time.time()
            depth_raw = depth_model.infer(im, edge_dilation=args.edge_dilation,
                                          depth_aa=args.depth_aa, tta=False, enable_amp=True)
            depth_raw = depth_model.minmax_normalize_chw(depth_raw).unsqueeze(0).float()
            t_depth = time.time() - t0

            c = TF.to_tensor(im).unsqueeze(0).to(depth_raw.device)
            if depth_raw.shape[-2:] != c.shape[-2:]:
                depth_raw = F.interpolate(depth_raw, size=c.shape[-2:], mode="bilinear",
                                          align_corners=True, antialias=True)

            depth_new = prepare_depth(c, depth_raw, cfg)

            # How badly is the silhouette smeared in the depth map?  This is
            # what ramp_px / edge_window have to span, and it is dominated by
            # the upsample from the depth model's internal resolution.
            raw_ramp = depth_edge_width(depth_raw)
            new_ramp = depth_edge_width(depth_new)
            print(f"  depth edge ramp: raw={raw_ramp:.1f}px  refined={new_ramp:.1f}px")
            ramp_px = args.ramp_px if args.ramp_px > 0 else max(8, int(round(new_ramp * 3)))
            edge_window = (args.edge_window if args.edge_window > 0
                           else max(3, int(round(new_ramp * 1.5))))
            anchor_px = args.anchor_px if args.anchor_px > 0 else ramp_px
            img_report_ramp = {"raw": round(raw_ramp, 2), "refined": round(new_ramp, 2),
                               "ramp_px": ramp_px, "edge_window": edge_window}
            header = [
                viz.label(viz.to_pil(c, args.tile_width), f"source {W}x{H}"),
                viz.label(viz.to_pil(depth_raw, args.tile_width),
                          f"depth {args.model_type} ({t_depth:.2f}s)"),
                viz.label(viz.to_pil(depth_new, args.tile_width),
                          f"depth refined  ramp {raw_ramp:.1f}px -> {new_ramp:.1f}px"
                          if not args.no_refine else "depth (no refine)"),
            ]
            rows = [header]
            zoom_rows = []
            img_report = {"file": path.basename(fn), "size": [W, H], "levels": {}}

            for div in args.divergence:
                lvl = {}
                shift_px = shift_scale(div, W)

                # ---------------- old pipeline ----------------
                old_mask = None
                t_old = float("nan")
                if not args.skip_old:
                    try:
                        t0 = time.time()
                        d_old = get_mapper("none")(depth_raw)
                        _, old_mask = old_forward_nonwarp_mask(
                            c, d_old, divergence=div, convergence=args.convergence, view="right")
                        old_mask = mask_closing(old_mask)
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        t_old = time.time() - t0
                        lvl["old"] = run_stats(old_mask)
                        lvl["old"]["seconds"] = round(t_old, 3)
                    except Exception as e:                       # noqa: BLE001
                        print(f"    old pipeline failed at divergence {div}: {e}")
                        old_mask = torch.zeros_like(depth_raw)
                        lvl["old"] = {"error": str(e)}

                # ---------------- new pipeline ----------------
                t0 = time.time()
                new_mask = make_mask(depth_new, div, args.convergence, cfg,
                                     base_size=W, view="right", soft=True, ramp=new_ramp)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t_new = time.time() - t0
                lvl["new"] = run_stats(new_mask)
                lvl["new"]["seconds"] = round(t_new, 4)
                lvl["shift_px"] = round(shift_px, 1)

                t0 = time.time()
                warped, hole = dibr_warp(c, depth_new, div, args.convergence,
                                         view="right", base_size=W)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t_warp = time.time() - t0
                t_render0 = time.time()

                msg = f"  div={div:5.1f} shift={shift_px:6.1f}px | "
                if not args.skip_old:
                    o = lvl.get("old", {})
                    msg += (f"old runs/row={o.get('runs_per_row', '-'):>5} "
                            f"jitter={o.get('edge_jitter_px', '-'):>5}px ({t_old:.2f}s) | ")
                msg += (f"new runs/row={lvl['new']['runs_per_row']:>5} "
                        f"jitter in/out={lvl['new']['edge_jitter_px']:>5}/"
                        f"{lvl['new']['outer_jitter_px']:<5}px "
                        f"masked={lvl['new']['coverage_pct']:>5}%")

                row = []
                if not args.skip_old:
                    row.append(viz.label(
                        viz.overlay_mask(c, old_mask, (255, 45, 45), width=args.tile_width),
                        f"OLD mask  div={div}  ({shift_px:.0f}px)  runs/row={lvl.get('old', {}).get('runs_per_row', '?')}"))
                row.append(viz.label(
                    viz.overlay_mask(c, new_mask, (40, 230, 90), width=args.tile_width),
                    f"NEW mask  div={div}  runs/row={lvl['new']['runs_per_row']}"))
                row.append(viz.label(
                    viz.overlay_mask(warped, hole, (255, 0, 220), width=args.tile_width),
                    f"NEW warped right eye + real holes  div={div}"))
                if not args.skip_old:
                    row.append(viz.label(
                        viz.overlay_two(c, old_mask, new_mask, width=args.tile_width),
                        f"DIFF  red=old only  green=new only  yellow=both  div={div}"))
                rows.append(row)

                # ---------------- zoom ----------------
                top, left, s = busiest_crop(new_mask, size=min(384, H, W))
                def _crop(t):
                    return t[..., top:top + s, left:left + s]
                zrow = []
                if not args.skip_old:
                    zrow.append(viz.label(viz.overlay_mask(_crop(c), _crop(old_mask), (255, 45, 45), width=420),
                                          f"OLD div={div}"))
                zrow.append(viz.label(viz.overlay_mask(_crop(c), _crop(new_mask), (40, 230, 90), width=420),
                                      f"NEW div={div}"))
                zrow.append(viz.label(viz.to_pil(_crop(depth_new), 420), "refined depth"))
                if not args.skip_old:
                    zrow.append(viz.label(
                        viz.overlay_two(_crop(c), _crop(old_mask), _crop(new_mask), width=420),
                        "DIFF red=old green=new"))
                zoom_rows.append(zrow)

                t_render = time.time() - t_render0
                lvl["new"]["warp_seconds"] = round(t_warp, 4)
                lvl["render_seconds"] = round(t_render, 3)
                print(msg + f" | mask {t_new * 1000:5.1f}ms  warp {t_warp * 1000:5.1f}ms  "
                            f"sheet {t_render * 1000:6.0f}ms")
                img_report["levels"][str(div)] = lvl

            sheet = viz.grid(rows)
            out_fn = path.join(args.output, f"{stem}_compare.png")
            sheet.save(out_fn)
            zsheet = viz.grid(zoom_rows)
            zout_fn = path.join(args.output, f"{stem}_zoom.png")
            zsheet.save(zout_fn)
            print(f"  -> {out_fn}")
            print(f"  -> {zout_fn}")
            if args.dump and fi < args.dump:
                top, left, sz = busiest_crop(new_mask, size=min(args.dump_size, H, W))
                dumps.append({
                    "file": path.basename(fn),
                    "crop": [int(top), int(left), int(sz)],
                    "full_size": [int(W), int(H)],
                    "rgb": c[..., top:top + sz, left:left + sz].half().cpu(),
                    "depth_raw": depth_raw[..., top:top + sz, left:left + sz].float().cpu(),
                    "depth_refined": depth_new[..., top:top + sz, left:left + sz].float().cpu(),
                    "args": vars(args),
                    "ramp_px": ramp_px, "edge_window": edge_window, "anchor_px": anchor_px,
                })
                print(f"  dumped crop top={top} left={left} size={sz}")

            img_report["depth_ramp"] = img_report_ramp
            report.append(img_report)

    with open(path.join(args.output, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "images": report}, f, indent=2)
    if dumps:
        dump_fn = path.join(args.output, "debug.pt")
        torch.save(dumps, dump_fn)
        print(f"debug dump -> {dump_fn}  ({path.getsize(dump_fn) / 1e6:.1f} MB)")
    print(f"\nreport -> {path.join(args.output, 'report.json')}")


if __name__ == "__main__":
    main()
