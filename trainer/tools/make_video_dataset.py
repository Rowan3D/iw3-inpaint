r"""
Source videos -> clip dataset for the temporal model.

The image version of this is make_dataset.py; everything about the depth, the
mask and the geometry is the same code (`ntrainer.pipeline`), so the two datasets
are consistent with each other. What changes is that the unit of work is a clip,
and three things must be held CONSTANT across the frames of a clip or the clip
is not a clip:

  * divergence, convergence and the depth mapper -- a clip whose divergence
    drifts frame to frame has no coherent geometry to learn.
  * the frame width -- resizing per frame would add scale jitter that looks like
    motion.
  * the crop rectangle -- this is the important one. The crop is chosen once for
    the whole clip and applied to every frame, so pixels correspond across
    frames. A per-frame random crop would destroy exactly the correspondence the
    temporal blocks exist to exploit.

Output layout, which is what `iw3.training.inpaint.dataset_video` reads:

    <out>/train/<clip-name>/0000_C.png
                            0000_M.png
                            0001_C.png ...
    <out>/eval/<clip-name>/...

Frames are read back in sorted filename order, hence the zero padding.

Run via ..\make_video_dataset.bat

NOTE: this GUI copy adds --stride and --start (see ntrainer/video_io.py);
New_Trainer\tools\make_video_dataset.py has neither.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from os import path

_HERE = path.dirname(path.dirname(path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_NUNIF = path.join(path.dirname(_HERE), "nunif")
if path.isdir(_NUNIF) and _NUNIF not in sys.path:
    sys.path.insert(0, _NUNIF)

import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

from ntrainer.geometry import depth_edge_width, hole_run_length  # noqa: E402
from ntrainer.pipeline import MaskConfig, MAPPER_SETS, prepare_depth, make_mask  # noqa: E402
from ntrainer.dataset_io import detail_map                       # noqa: E402
from ntrainer.video_io import list_videos, probe, iter_clips     # noqa: E402
from iw3.depth_scaler import minmax_normalize                    # noqa: E402


def stable_id(p, root):
    rel = path.relpath(p, root).replace("\\", "/")
    return hashlib.md5(rel.encode("utf-8")).hexdigest()[:12]


def resize_clip(frames, target_width, allow_upscale=False):
    """(T,3,H,W) resized so WIDTH == target_width. Width, not the short side:
    hole size scales with width (hole_px = depth_drop * divergence * 0.005 * W),
    so normalising the short side makes portrait clips produce tiny holes."""
    _, _, h, w = frames.shape
    if w == target_width:
        return frames, False
    capped = False
    if target_width > w and not allow_upscale:
        target_width, capped = w, True
    if target_width == w:
        return frames, capped
    new_h = max(1, int(round(h * (target_width / w))))
    new_h -= new_h % 2
    frames = F.interpolate(frames, size=(new_h, target_width), mode="bilinear",
                           align_corners=False, antialias=True)
    return frames, capped


def pick_clip_crop(mask, detail, size, tries, rng):
    """One rectangle for the whole clip.

    Scored on the clip total of `mask_area * detail_under_mask`, the same rule
    the image dataset uses, summed over frames so a rectangle that only holds a
    hole in frame 0 loses to one that holds it throughout.
    """
    T, _, H, W = mask.shape
    if H < size or W < size:
        return None
    best = None
    for _ in range(tries):
        i = rng.randint(0, H - size)
        j = rng.randint(0, W - size)
        m = mask[:, :, i:i + size, j:j + size]
        area = m.sum().item()
        if area <= 0:
            continue
        d = detail[:, :, i:i + size, j:j + size]
        dsc = (d * m).sum().item() / area          # detail of what must be filled
        if best is None or area * dsc > best[0]:
            best = (area * dsc, i, j, area, dsc)
    return best


INCOMPLETE_DIR = ".incomplete"


def clear_incomplete(output):
    """Throw away clip folders left half-written by a run that was stopped."""
    import shutil
    d = path.join(output, INCOMPLETE_DIR)
    if not path.isdir(d):
        return 0
    n = 0
    for name in os.listdir(d):
        try:
            shutil.rmtree(path.join(d, name))
            n += 1
        except OSError:
            pass
    return n


def write_clip(rgb, mask, out_dir, png_level=1):
    """Write one clip, appearing at `out_dir` only once it is complete.

    Frames used to be written straight into the final folder. Stop a prep run
    part way -- which is a normal thing to do on a job this long -- and it left
    a folder holding, say, five of twelve frames. The next run sees that folder
    exists, skips the whole video, and the short clip is in the dataset for
    good: training then refuses to start at all, because check_video_dataset
    rejects any clip with fewer than seq frames, and the only cure is to find
    and delete it by hand.

    Writing to a staging folder and renaming into place means a killed run
    leaves nothing behind that the skip check can mistake for finished work.
    The rename is atomic on both NTFS and ext4, and staging sits beside train/
    and eval/ rather than inside them, so a partial folder is never mistaken
    for a clip either.
    """
    from torchvision.transforms import functional as TF
    root = path.dirname(path.dirname(path.abspath(out_dir)))     # <output>
    tmp = path.join(root, INCOMPLETE_DIR, path.basename(out_dir))
    if path.isdir(tmp):
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    rgb = (rgb.clamp(0, 1) * 255).round().to(torch.uint8)
    # dataset_video does `mask > 0`, so a soft mask would dilate by its whole
    # falloff -- binarise here, exactly as dataset_io.save_pair does for images.
    mask = ((mask > 0.5).float() * 255).round().to(torch.uint8)
    for t in range(rgb.shape[0]):
        TF.to_pil_image(rgb[t]).save(path.join(tmp, f"{t:04d}_C.png"),
                                     compress_level=png_level)
        TF.to_pil_image(mask[t]).save(path.join(tmp, f"{t:04d}_M.png"),
                                      compress_level=png_level)
    os.makedirs(path.dirname(path.abspath(out_dir)), exist_ok=True)
    try:
        os.rename(tmp, out_dir)
    except OSError:
        # Someone else got there first (a re-run of the same clip), or the
        # destination survived from an --overwrite pass. Replace it.
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        os.rename(tmp, out_dir)


def write_manifest(args, cfg):
    """Record how this dataset was built, next to the dataset.

    Nothing about a finished dataset says which depth model made it, at what
    resolution, or whether depth refinement was on -- and refinement in
    particular changes every mask in it. Without this the only evidence months
    later is whatever the GUI happens to have in config.json, which is the
    CURRENT settings, not the ones that produced these files. Appended to, not
    overwritten, so resuming or extending a dataset keeps the earlier entries.
    """
    import json
    from dataclasses import asdict
    from datetime import datetime

    fp = path.join(args.output, "dataset.json")
    entry = {
        "written": datetime.now().isoformat(timespec="seconds"),
        "tool": path.basename(__file__),
        "depth": {"model": args.model_type, "resolution": args.resolution,
                  "depth_aa": bool(args.depth_aa)},
        "mask": asdict(cfg),
        "clips": {"seq": args.seq, "fps": args.fps, "stride": args.stride,
                  "skip": args.skip, "start": args.start,
                  "per_video": args.clips_per_video, "crop": args.size,
                  "mirror": not args.no_mirror},
        "warp": {"divergence": list(args.divergence),
                 "convergence": list(args.convergence),
                 "divergence_power": args.divergence_power,
                 "frame_width": list(args.frame_width),
                 "mapper_set": args.mapper_set},
        "source": args.input,
    }
    try:
        os.makedirs(args.output, exist_ok=True)
        runs = []
        if path.isfile(fp):
            with open(fp, encoding="utf-8") as f:
                old = json.load(f)
            runs = old.get("runs", []) if isinstance(old, dict) else []
        runs.append(entry)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump({"runs": runs}, f, indent=2)
        if not cfg.refine:
            print("WARNING: depth edge refinement is OFF for this dataset. Every mask "
                  "follows a depth ramp of ~4.9px instead of ~1.3px, and the only fix "
                  "is to build the dataset again.\n")
    except OSError as e:
        print(f"note  : could not write {fp} ({e})")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("src", type=str, nargs="?", default=None)
    p.add_argument("dst", type=str, nargs="?", default=None)
    p.add_argument("--input", type=str, default=None, help="folder of video files")
    p.add_argument("--output", type=str, default=None, help="dataset root")
    p.add_argument("--eval-ratio", type=float, default=0.03)
    p.add_argument("--prefix", type=str, default="ntv")
    # clip sampling
    p.add_argument("--seq", type=int, default=12,
                   help="frames per clip. Must be >= the model's seq_len (12)")
    p.add_argument("--fps", type=float, default=30.0,
                   help="sample rate. Consecutive frames of 60fps footage barely move, "
                        "which teaches the temporal blocks that nothing ever does")
    p.add_argument("--skip", type=int, default=24,
                   help="sampled frames dropped between clips from one video")
    p.add_argument("--clips-per-video", type=int, default=4, help="0 = as many as fit")
    p.add_argument("--stride", type=int, default=1,
                   help="extra frame step inside a clip. 1 keeps every sampled frame; "
                        "2 takes every second one, so consecutive frames move twice as "
                        "far without changing the clip length")
    p.add_argument("--start", type=str, default="0",
                   help="where in each video to start sampling: a number of seconds, "
                        "or 'random' for a different section each run (never so late "
                        "that the clips would run past the end)")
    p.add_argument("--min-width", type=int, default=1280, help="skip videos narrower than this")
    # depth (same defaults as make_dataset.py)
    p.add_argument("--model-type", type=str, default="Any_V3_Mono")
    p.add_argument("--resolution", type=int, default=784)
    p.add_argument("--depth-aa", dest="depth_aa", action="store_true", default=True)
    p.add_argument("--no-depth-aa", dest="depth_aa", action="store_false")
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--depth-batch", type=int, default=4,
                   help="frames per depth forward pass")
    p.add_argument("--mask-batch", type=int, default=4,
                   help="frames per depth-refine / mask pass. Both are per-frame, so this "
                        "only trades speed for VRAM and changes nothing in the output")
    # geometry (same as make_dataset.py)
    p.add_argument("--frame-width", type=int, nargs="+", default=[1280, 1920])
    p.add_argument("--allow-upscale", action="store_true")
    p.add_argument("--divergence", type=float, nargs=2, default=[2.0, 16.0],
                   metavar=("MIN", "MAX"))
    p.add_argument("--convergence", type=float, nargs=2, default=[0.0, 1.0],
                   metavar=("MIN", "MAX"))
    p.add_argument("--divergence-power", type=float, default=1.0)
    p.add_argument("--mapper-set", type=str, default="safe", choices=sorted(MAPPER_SETS))
    # output
    p.add_argument("--size", type=int, default=640,
                   help="saved crop size. Matches make_dataset.bat's 640 so the image and "
                        "video sets crop the same way; 512 saves ~35%% of the disk")
    p.add_argument("--png-level", type=int, default=1, choices=list(range(10)),
                   help="PNG compression. Measured on a 512px frame: level 6 (PIL's "
                        "default) 48ms/503KB, level 1 36ms/547KB, level 0 21ms/769KB. "
                        "Masks are near-free at any level")
    p.add_argument("--crop-tries", type=int, default=8)
    p.add_argument("--min-detail", type=float, default=0.0,
                   help="drop a clip whose MASKED region is flatter than this (mean "
                        "|gradient|). That region is exactly what the model must "
                        "reconstruct, so an out-of-focus one can only teach blur -- and "
                        "stock video is far worse for shallow depth of field than the "
                        "image corpora were. Reference: sharp texture ~0.13, bokeh ~0.007. "
                        "0 = keep everything but still report the distribution")
    p.add_argument("--min-mask", type=float, default=300.0,
                   help="drop a clip whose mean per-frame mask sum is below this, matching "
                        "create_training_data's own floor")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--limit", type=int, default=0, help="max source videos")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    p.add_argument("--scan", action="store_true",
                   help="report what the source videos are (resolution, fps, length, how "
                        "many clips they would yield) and exit")
    args = p.parse_args()

    args.input = args.input or args.src
    args.output = args.output or args.dst
    if not args.input or not args.output:
        p.error("need a source and an output folder, either positionally "
                "(make_video_dataset.bat SRC DST) or as --input SRC --output DST")
    base = os.environ.get("NT_CWD") or os.getcwd()
    args.input = path.abspath(path.join(base, path.expandvars(path.expanduser(args.input))))
    args.output = path.abspath(path.join(base, path.expandvars(path.expanduser(args.output))))
    if path.abspath(args.input) == path.abspath(args.output):
        p.error("output folder must not be the same as the source folder")
    if args.seq < 12:
        p.error(f"--seq {args.seq} is below the model's seq_len of 12; the dataset would "
                f"raise at load time")

    videos = list_videos(args.input)
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit(f"no video files found under {args.input}")
    print(f"source: {args.input}\noutput: {args.output}\n{len(videos)} video file(s)")

    if args.scan:
        return scan(videos, args)

    rng0 = random.Random(args.seed)
    order = list(videos)
    rng0.shuffle(order)
    n_eval = max(1, int(len(order) * args.eval_ratio))
    split_of = {f: ("eval" if i < n_eval else "train") for i, f in enumerate(order)}
    print(f"holding out {n_eval} video(s) for eval "
          f"(split by VIDEO, so no clip of a training video leaks into eval)")

    from iw3.depth_model_factory import create_depth_model
    from iw3.utils import get_mapper

    device_ok = args.gpu >= 0 and torch.cuda.is_available()
    for s in ("train", "eval"):
        os.makedirs(path.join(args.output, s), exist_ok=True)

    cfg = MaskConfig.from_args(args)
    depth_model = create_depth_model(args.model_type)
    depth_model.load(gpu=args.gpu, resolution=args.resolution)
    depth_model.disable_ema()
    print(f"depth={args.model_type} @ {getattr(depth_model.model, 'prep_lower_bound', '?')}px "
          f"depth_aa={args.depth_aa}")
    print(f"clips : {args.seq} frames @ {args.fps}fps, stride {args.stride}, "
          f"skip {args.skip}, start {args.start}, "
          f"up to {args.clips_per_video or 'all'} per video, crop {args.size}")
    print(f"mask  : {cfg.describe()}\n")
    dropped = clear_incomplete(args.output)
    if dropped:
        print(f"note  : discarded {dropped} half-written clip(s) from a stopped run\n")
    write_manifest(args, cfg)

    n_clips = n_drop = n_skip = n_flat = 0
    detail_scores = []
    hole_hist = torch.zeros(4096, dtype=torch.int64)
    t_start = time.time()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = []

    try:
        for vi, fn in enumerate(videos):
            uid = stable_id(fn, args.input)
            split = split_of[fn]
            out_root = path.join(args.output, split)
            if not args.overwrite and path.isdir(
                    path.join(out_root, f"{args.prefix}_{uid}_0_a")):
                n_skip += 1
                continue

            rng = random.Random((args.seed * 1000003) ^ int(uid, 16))
            for ci, frames in enumerate(iter_clips(
                    fn, seq=args.seq, fps=args.fps, skip=args.skip,
                    max_clips=args.clips_per_video, min_width=args.min_width,
                    stride=args.stride, start=args.start, rng=rng)):

                # one geometry for the whole clip
                divergence_lo, divergence_hi = args.divergence
                u = rng.random() ** (1.0 / max(args.divergence_power, 1e-3))
                divergence = divergence_lo + (divergence_hi - divergence_lo) * u
                convergence = rng.uniform(*args.convergence)
                mapper = rng.choice(MAPPER_SETS[args.mapper_set])
                target_w = rng.choice(args.frame_width)

                dev = "cuda" if device_ok else "cpu"
                with torch.inference_mode():
                    # uint8 to the GPU first, then float and resize there. Doing
                    # this on the CPU cost 11.5 s per clip at 4K -- more than the
                    # depth model, the mask and the PNG encoding put together.
                    c = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
                    c = c.to(dev, non_blocking=True).float().div_(255)
                    c, _ = resize_clip(c, target_w, allow_upscale=args.allow_upscale)
                    if min(c.shape[-2:]) < args.size:
                        n_drop += 1
                        continue
                    depths = []
                    for chunk in c.split(max(1, args.depth_batch), dim=0):
                        d = depth_model.infer(chunk, edge_dilation=0, depth_aa=args.depth_aa,
                                              tta=False, enable_amp=True)
                        depths.append(d)
                    d = torch.cat(depths, dim=0).float()
                    if d.ndim == 3:
                        d = d.unsqueeze(1)
                    # ONE min/max for the whole clip, not one per frame.
                    #
                    # `disable_ema()` leaves the scaler at buffer_size=1, so
                    # minmax_normalize_chw() normalises each frame by its own
                    # min and max. Min-max is scale invariant, so a uniform
                    # wobble cancels -- but the moment anything NEARER enters
                    # the shot (a person walks in, a car passes, a branch
                    # crosses) that frame's max jumps and everything else is
                    # rescaled to compensate. Measured on a clip where the
                    # subject is perfectly static and an intruder appears at
                    # frame 6: the static subject's band jumped 16px and lost
                    # 44.7% of its area, mid-clip. Per clip: 0px, 0.0%.
                    #
                    # That instability would be baked into the ground truth --
                    # the exact flicker the temporal model exists to remove.
                    d = minmax_normalize(d, d.amin(), d.amax())
                    if d.shape[-2:] != c.shape[-2:]:
                        d = F.interpolate(d, size=c.shape[-2:], mode="bilinear",
                                          align_corners=True, antialias=True)
                    # prepare_depth and make_mask are both per-frame (the depth
                    # refine is spatial, the mask is per row/column within a
                    # frame), so chunking the time axis is exactly equivalent and
                    # keeps 12 frames of 1920px intermediates out of VRAM.
                    mfn = None if mapper == "none" else get_mapper(mapper)
                    d = torch.cat([prepare_depth(cc_, dd_, cfg, mapper_fn=mfn)
                                   for cc_, dd_ in zip(c.split(args.mask_batch, dim=0),
                                                       d.split(args.mask_batch, dim=0))], dim=0)
                    # one ramp for the whole clip, so the geometry does not drift
                    # frame to frame
                    ramp = depth_edge_width(d)

                    views = [("a", c, d)]
                    if not args.no_mirror:
                        views.append(("b", c.flip(-1), d.flip(-1)))

                    for tag, cc, dd in views:
                        mm = torch.cat([
                            make_mask(dd_, divergence, convergence, cfg,
                                      base_size=cc.shape[-1], view="right", ramp=ramp).float()
                            for dd_ in dd.split(args.mask_batch, dim=0)], dim=0)
                        if mm.sum().item() / mm.shape[0] < args.min_mask:
                            n_drop += 1
                            continue
                        det = torch.stack([detail_map(f) for f in cc])
                        pick = pick_clip_crop(mm, det, args.size, args.crop_tries, rng)
                        if pick is None:
                            n_drop += 1
                            continue
                        _, i, j, _, dsc = pick
                        if dsc < args.min_detail:
                            n_flat += 1
                            continue
                        rgb_c = cc[:, :, i:i + args.size, j:j + args.size].cpu()
                        mask_c = mm[:, :, i:i + args.size, j:j + args.size].cpu()
                        if mask_c.sum().item() / mask_c.shape[0] < args.min_mask:
                            n_drop += 1
                            continue

                        mb = mask_c > 0.5
                        if mb.any():
                            rl = hole_run_length(~mb)[mb]
                            hole_hist += torch.bincount(rl.clamp(max=4095).long(),
                                                        minlength=4096)[:4096]
                        detail_scores.append(dsc)
                        out_dir = path.join(out_root, f"{args.prefix}_{uid}_{ci}_{tag}")
                        futures.append(pool.submit(write_clip, rgb_c, mask_c, out_dir,
                                                   args.png_level))
                        n_clips += 1

            if (vi + 1) % 5 == 0 or vi + 1 == len(videos):
                el = time.time() - t_start
                rate = (vi + 1 - n_skip) / el if el else 0
                print(f"  [{vi + 1}/{len(videos)}] {n_clips} clips, {n_drop} dropped, "
                      f"{n_skip} skipped | {rate * 60:.1f} videos/min", flush=True)
    finally:
        for f in futures:
            try:
                f.result()
            except Exception as e:                               # noqa: BLE001
                print(f"  write failed: {e}")
        pool.shutdown(wait=True)

    print(f"\n{n_clips} clips written, {n_drop} dropped, {n_skip} videos skipped, "
          f"in {(time.time() - t_start) / 60:.1f} min")
    if n_flat:
        print(f"{n_flat} clip(s) dropped by --min-detail {args.min_detail}")
    if detail_scores:
        ds = sorted(detail_scores)
        def q(f):
            return ds[min(len(ds) - 1, int(f * len(ds)))]
        print(f"detail under the mask: p10={q(.1):.4f} median={q(.5):.4f} p90={q(.9):.4f}"
              f"   (bokeh ~0.007, sharp texture ~0.13)")
        soft = sum(1 for v in ds if v < 0.02) / len(ds) * 100
        print(f"  {soft:.0f}% of clips are below 0.02 -- these teach blur, not detail."
              + ("  Consider --min-detail 0.02." if soft > 20 else ""))
    try:
        total = int(hole_hist.sum().item())
        if total:
            cdf = torch.cumsum(hole_hist, 0).double() / total
            def pct(q):
                return int(torch.searchsorted(cdf, torch.tensor(q, dtype=torch.float64)).item())
            print(f"hole width under the mask: p10={pct(.1)} median={pct(.5)} "
                  f"p90={pct(.9)} p99={pct(.99)}")
    except Exception as e:                                       # noqa: BLE001
        print(f"(hole stats unavailable: {e})")
    print(f"\nnext:\n  train.bat \"{args.output}\" <model-dir> --video "
          f"--crop-size 256 --checkpoint-file <image-model>.pth")
    return 0


def scan(videos, args):
    print(f"\n{'file':44s} {'res':>11s} {'fps':>6s} {'len':>7s} {'clips':>6s}")
    total_clips = 0
    usable = 0
    per_frame = args.seq + args.skip
    for fn in videos[:200]:
        info = probe(fn)
        if info is None:
            print(f"{path.basename(fn)[:44]:44s} {'unreadable':>11s}")
            continue
        w, h, fps, n, dur = info
        stride = (max(1, int(round(fps / args.fps))) if fps > args.fps else 1) * max(1, args.stride)
        sampled = n // stride if n else 0
        clips = sampled // per_frame if per_frame else 0
        if args.clips_per_video:
            clips = min(clips, args.clips_per_video)
        if w < args.min_width:
            clips = 0
        else:
            usable += 1
        total_clips += clips
        print(f"{path.basename(fn)[:44]:44s} {w}x{h:<5d} {fps:6.1f} {dur:6.1f}s {clips:6d}")
    if len(videos) > 200:
        print(f"... and {len(videos) - 200} more (not scanned)")
    mult = 1 if args.no_mirror else 2
    print(f"\n{usable}/{min(len(videos), 200)} videos at >= {args.min_width}px")
    print(f"~{total_clips * mult} clips ({total_clips} x {mult} for the mirror), "
          f"{total_clips * mult * args.seq} frames, "
          f"~{total_clips * mult * args.seq * 0.35:.0f} MB at {args.size}px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
