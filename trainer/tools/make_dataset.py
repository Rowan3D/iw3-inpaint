r"""
Source images -> finished inpaint training dataset, in one pass.

Replaces `iw3/training/inpaint/create_training_data.py`.  For each source image:
depth (DA3) -> depth refine -> analytic disocclusion mask -> crops -> a
`_C.png` / `_M.png` pair per crop, written through `ntrainer.dataset_io.save_pair`
so the format is the one `verify_format.bat` proves.

Randomisation mirrors nagadomi's `gen_data()` so the data keeps the same
diversity: random rotate+crop, random resize, random depth mapper, random
convergence, random divergence.  Two differences, both deliberate:

  * divergence is sampled across the WHOLE range in one pass (default 2-16)
    rather than in four fixed "levels", and needs no downscale hack, because
    the analytic mask has no iteration cap.
  * every source yields the frame and its mirror, both masked in the SAME
    right-view convention, instead of a random left/right handedness.
    `BaseImageInpaint._inpaint_single()` flips the left eye before inference,
    so the model only ever sees one handedness -- mirroring doubles the data
    without teaching it a convention it will never meet.

Run via ..\make_dataset.bat
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
from PIL import Image                                            # noqa: E402

from ntrainer.geometry import depth_edge_width, hole_run_length  # noqa: E402
from ntrainer.pipeline import MaskConfig, MAPPER_SETS, prepare_depth, make_mask  # noqa: E402
from ntrainer.dataset_io import save_pair, detail_map, TRAIN_CROP_SIZE  # noqa: E402


IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(root):
    out = []
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            if path.splitext(n)[1].lower() in IMG_EXT:
                out.append(path.join(dirpath, n))
    return sorted(out)


def stable_id(p, root):
    rel = path.relpath(p, root).replace("\\", "/")
    return hashlib.md5(rel.encode("utf-8")).hexdigest()[:12]


def crop_resize(im, target_width, rng, rotate_prob=0.25, allow_upscale=False):
    """
    Random rotate+centre-crop, then resize to `target_width`.

    Normalises WIDTH, not the short side.  The warp scale is
    `shift = divergence * 0.005 * width`, so width alone sets how wide the holes
    come out.  nagadomi's original normalises the short side, which means a
    portrait frame collapses to the short-side value -- a 1920x3194 source ends
    up 1080px wide and can only ever produce 73px holes at divergence 16,
    instead of 131px.  Same for the rotate path, which yields a square.

    Never upscales unless asked: the frame is capped at the source's own width,
    so a rotated crop simply produces a narrower frame rather than mush.
    Returns (image, was_capped).
    """
    if rng.random() < rotate_prob:
        from torchvision.transforms import functional as TF
        from torchvision.transforms import InterpolationMode
        crop = int(min(im.size) * (1 / 2 ** 0.5))
        im = TF.rotate(im, angle=rng.uniform(-45, 45), interpolation=InterpolationMode.BILINEAR)
        im = TF.center_crop(im, (crop, crop))
    w, h = im.size
    capped = False
    if not allow_upscale and target_width > w:
        target_width, capped = w, True
    nh = max(1, int(round(h * (target_width / w))))
    im = im.resize((target_width, nh), Image.BICUBIC if rng.random() < 0.5 else Image.BILINEAR)
    return im, capped


def scan(sources, args):
    """Audit the corpus: what hole widths can it actually reach, and what will be lost."""
    import statistics
    widths, small, bad = [], 0, 0
    for fn, _ in sources:
        try:
            with Image.open(fn) as im:
                w, h = im.size
        except Exception:                                        # noqa: BLE001
            bad += 1
            continue
        if max(w, h) > min(w, h) * 3:
            continue
        widths.append(w)
        if min(w, h) < args.size:
            small += 1
    if not widths:
        print("no usable images")
        return 1
    widths.sort()
    q = lambda f: widths[min(len(widths) - 1, int(len(widths) * f))]   # noqa: E731
    want = max(args.frame_width)
    print(f"\n{len(widths)} usable image(s)" + (f", {bad} unreadable" if bad else ""))
    print(f"  width: min={widths[0]}  p10={q(.1)}  median={q(.5)}  p90={q(.9)}  max={widths[-1]}")
    for t in (1280, 1920, 2560):
        n = sum(1 for w in widths if w >= t)
        print(f"  {n:6d} ({n / len(widths) * 100:5.1f}%) are at least {t}px wide")
    if small:
        print(f"  !! {small} ({small / len(widths) * 100:.0f}%) have a side under --size "
              f"{args.size} and will be SKIPPED")
    eff = statistics.median([min(w, want) for w in widths])
    print(f"\nwith --frame-width up to {want} (no upscaling), the median frame will be "
          f"{eff:.0f}px wide")
    print(f"  strong-silhouette hole at divergence {args.divergence[1]:.0f}: "
          f"{0.85 * args.divergence[1] * 0.005 * eff:.0f}px "
          f"({0.85 * args.divergence[1] * 0.005 * eff / TRAIN_CROP_SIZE * 100:.0f}% of a "
          f"{TRAIN_CROP_SIZE}px training crop)")
    if eff < want * 0.8:
        print(f"  !! the corpus is the binding constraint, not --frame-width. For holes at "
              f"the\n     full {want}px scale you need sources at least {want}px wide.")
    print(f"\n  --rotate-prob {args.rotate_prob} shrinks a frame to 1/sqrt(2) = 71% of its "
          f"width when it fires")
    return 0


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("src", type=str, nargs="?", default=None,
                   help="source image folder (positional form of --input)")
    p.add_argument("dst", type=str, nargs="?", default=None,
                   help="output dataset folder (positional form of --output)")
    p.add_argument("--input", type=str, default=None,
                   help="source image folder (searched recursively). If it contains "
                        "train/ and eval/ subfolders those are used as the split.")
    p.add_argument("--output", type=str, default=None,
                   help="dataset root; train/ and eval/ are created inside it")
    p.add_argument("--eval-ratio", type=float, default=0.02,
                   help="fraction held out for eval when the input is not pre-split")
    p.add_argument("--prefix", type=str, default="nt")
    # depth
    p.add_argument("--model-type", type=str, default="Any_V3_Mono")
    p.add_argument("--resolution", type=int, default=784)
    p.add_argument("--depth-aa", dest="depth_aa", action="store_true", default=True)
    p.add_argument("--no-depth-aa", dest="depth_aa", action="store_false")
    p.add_argument("--no-refine", action="store_true", help="skip depth edge refinement")
    p.add_argument("--gpu", type=int, default=0)
    # geometry
    p.add_argument("--frame-width", type=int, nargs="+", default=[1280, 1920],
                   help="frame WIDTH sampled before warping. This is what sets hole size: "
                        "hole_px = depth_drop * divergence * 0.005 * frame_width. Never "
                        "upscales past the source's own width.")
    p.add_argument("--rotate-prob", type=float, default=0.25,
                   help="probability of the rotate+centre-crop augmentation. It yields a "
                        "square, which narrows the frame, so it now costs hole width")
    p.add_argument("--allow-upscale", action="store_true",
                   help="let --frame-width upscale a smaller source")
    p.add_argument("--divergence", type=float, nargs=2, default=[2.0, 16.0],
                   metavar=("MIN", "MAX"))
    p.add_argument("--convergence", type=float, nargs=2, default=[0.0, 1.0],
                   metavar=("MIN", "MAX"))
    p.add_argument("--divergence-power", type=float, default=1.0,
                   help="bias of the divergence sampling. 1.0 = uniform; 2-3 weights it "
                        "toward the high end, which is where the wide holes come from. "
                        "Uniform means only ~1 image in 7 gets divergence >= 14.")
    p.add_argument("--mapper-set", type=str, default="safe", choices=sorted(MAPPER_SETS),
                   help="which depth mappers to sample. 'safe' excludes mul_1/2/3, which "
                        "measurably double the mask's edge jitter (2.1px vs 1.0px) by "
                        "crushing background depth structure")
    # output crops
    p.add_argument("--size", type=int, default=640,
                   help="saved crop size. Must exceed --train-crop with room to spare, or "
                        "random cropping has no jitter left. 640 gives headroom for a 512 "
                        "training crop; nunif's own default was 512 for a 256 crop.")
    p.add_argument("--train-crop", type=int, default=TRAIN_CROP_SIZE,
                   help="the crop size training will actually feed the model. Used to "
                        "sanity-check --size and to report hole-to-input ratios.")
    p.add_argument("--num-samples", type=int, default=2, help="crops per view")
    p.add_argument("--min-detail", type=float, default=0.0,
                   help="drop crops whose MASKED region is flatter than this (mean "
                        "|gradient|). The masked region is exactly what the model must "
                        "reconstruct, so an out-of-focus one can only teach blur. "
                        "Reference: sharp texture ~0.13, bokeh ~0.007, flat 0.0. "
                        "0 = keep everything but still report the distribution.")
    p.add_argument("--min-parallax", type=float, default=0.0,
                   help="skip images whose mask covers less than this %% of the frame at a "
                        "REFERENCE divergence of 8. Measures the image's own parallax, "
                        "independent of the divergence sampled for it, so it drops flat "
                        "scenes that would teach nothing. 0 = keep everything; ~0.5 is a "
                        "reasonable floor.")
    p.add_argument("--no-mirror", action="store_true",
                   help="only write the original orientation, not its mirror")
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate even if the output already exists")
    p.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    p.add_argument("--scan", action="store_true",
                   help="audit the source folder and exit: width distribution, how many "
                        "are too small for --size, and the hole width the corpus can "
                        "actually reach. Run this before committing to a long job.")
    args = p.parse_args()

    # Accept either form:  make_dataset.bat SRC DST   or   --input SRC --output DST
    args.input = args.input or args.src
    args.output = args.output or args.dst
    if not args.input or not args.output:
        p.error("need a source and an output folder, either positionally "
                "(make_dataset.bat SRC DST) or as --input SRC --output DST")
    # The launcher pushd's into the nunif dir, so resolve the user's relative
    # paths against the directory they actually typed the command in.
    base = os.environ.get("NT_CWD") or os.getcwd()
    args.input = path.abspath(path.join(base, path.expandvars(path.expanduser(args.input))))
    args.output = path.abspath(path.join(base, path.expandvars(path.expanduser(args.output))))
    if args.size <= args.train_crop:
        p.error(f"--size {args.size} must be larger than --train-crop {args.train_crop}; "
                f"try --size {args.train_crop + 128} or more")
    if not path.isdir(args.input):
        p.error(f"source folder does not exist: {args.input}")
    if path.abspath(args.output) == path.abspath(args.input):
        p.error("output folder must not be the same as the source folder")
    print(f"source: {args.input}\noutput: {args.output}")

    from torchvision.transforms import functional as TF
    from iw3.depth_model_factory import create_depth_model
    from iw3.utils import get_mapper

    device_ok = args.gpu >= 0 and torch.cuda.is_available()
    print(f"torch={torch.__version__}  cuda={device_ok}"
          + (f"  gpu={torch.cuda.get_device_name(args.gpu)}" if device_ok else ""))

    # ------------------------------------------------------------ input split
    pre_split = all(path.isdir(path.join(args.input, s)) for s in ("train", "eval"))
    if pre_split:
        sources = [(f, "train") for f in list_images(path.join(args.input, "train"))]
        sources += [(f, "eval") for f in list_images(path.join(args.input, "eval"))]
        print(f"input is pre-split: {sum(1 for _, s in sources if s == 'train')} train, "
              f"{sum(1 for _, s in sources if s == 'eval')} eval")
    else:
        files = list_images(args.input)
        rng = random.Random(args.seed)
        rng.shuffle(files)
        n_eval = max(1, int(len(files) * args.eval_ratio)) if files else 0
        sources = [(f, "eval") for f in files[:n_eval]] + [(f, "train") for f in files[n_eval:]]
        sources.sort()
        print(f"{len(files)} image(s); holding out {n_eval} for eval "
              f"({args.eval_ratio * 100:.1f}%)")
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        raise SystemExit(f"no images found under {args.input}")

    if args.scan:
        return scan(sources, args)

    for s in ("train", "eval"):
        os.makedirs(path.join(args.output, s), exist_ok=True)

    cfg = MaskConfig.from_args(args)

    depth_model = create_depth_model(args.model_type)
    depth_model.load(gpu=args.gpu, resolution=args.resolution)
    depth_model.disable_ema()
    print(f"depth={args.model_type} @ {getattr(depth_model.model, 'prep_lower_bound', '?')}px "
          f"depth_aa={args.depth_aa}  divergence {args.divergence[0]}-{args.divergence[1]}  "
          f"frame width {args.frame_width}")
    print(f"mask: {cfg.describe()}")
    print(f"mappers[{args.mapper_set}]: {sorted(set(MAPPER_SETS[args.mapper_set]))}\n")

    t_depth = t_mask = 0.0
    n_pairs = n_done = n_skip = n_capped = n_flat = 0
    HIST_BINS = 4096
    hole_hist = torch.zeros(HIST_BINS, dtype=torch.int64)
    widest = (0, 0.0, 0)
    detail_scores = []
    frame_w = []
    div_used = []
    t_start = time.time()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = []

    try:
        for idx, (fn, split) in enumerate(sources):
            uid = stable_id(fn, args.input)
            out_dir = path.join(args.output, split)
            base0 = path.join(out_dir, f"{args.prefix}_{args.model_type}_{uid}_a")
            if not args.overwrite and path.exists(f"{base0}_0_C.png"):
                n_skip += 1
                continue

            rng = random.Random((args.seed * 1000003) ^ int(uid, 16))
            try:
                im = Image.open(fn).convert("RGB")
            except Exception as ex:                               # noqa: BLE001
                print(f"  skip {path.basename(fn)}: {ex}")
                continue
            if max(im.size) > min(im.size) * 3:
                continue                                          # extreme aspect ratio

            im, capped = crop_resize(im, rng.choice(args.frame_width), rng,
                                     rotate_prob=args.rotate_prob,
                                     allow_upscale=args.allow_upscale)
            n_capped += int(capped)
            frame_w.append(im.size[0])
            if min(im.size) < args.size:
                # RandomCrop would raise; catch it here with a useful message
                print(f"  skip {path.basename(fn)}: {im.size} smaller than --size {args.size}")
                continue
            lo, hi = args.divergence
            divergence = lo + (hi - lo) * (rng.random() ** (1.0 / max(args.divergence_power, 1e-3)))
            div_used.append(divergence)
            convergence = rng.uniform(*args.convergence)
            mapper = rng.choice(MAPPER_SETS[args.mapper_set])

            with torch.inference_mode():
                t0 = time.time()
                d = depth_model.infer(im, edge_dilation=0, depth_aa=args.depth_aa,
                                      tta=False, enable_amp=True)
                d = depth_model.minmax_normalize_chw(d).unsqueeze(0).float()
                if device_ok:
                    torch.cuda.synchronize()
                t_depth += time.time() - t0

                c = TF.to_tensor(im).unsqueeze(0).to(d.device)
                if d.shape[-2:] != c.shape[-2:]:
                    d = F.interpolate(d, size=c.shape[-2:], mode="bilinear",
                                      align_corners=True, antialias=True)

                t0 = time.time()
                d = prepare_depth(c, d, cfg,
                                  mapper_fn=None if mapper == "none" else get_mapper(mapper))
                ramp = depth_edge_width(d)
                if args.min_parallax > 0:
                    # score the IMAGE, not this sample: a low sampled divergence would
                    # otherwise look like a flat scene
                    ref = make_mask(d, 8.0, 0.5, cfg, base_size=c.shape[-1], ramp=ramp)
                    par = ref.float().mean().item() * 100
                    if par < args.min_parallax:
                        n_flat += 1
                        continue
                views = [("a", c, d)]
                if not args.no_mirror:
                    views.append(("b", c.flip(-1), d.flip(-1)))
                masks = [(tag, cc, make_mask(dd, divergence, convergence, cfg,
                                             base_size=c.shape[-1], view="right", ramp=ramp))
                         for tag, cc, dd in views]
                if device_ok:
                    torch.cuda.synchronize()
                t_mask += time.time() - t0

            for tag, cc, mm in masks:
                m = mm[0].cpu()
                # Area-weighted hole width: for every pixel the model will have to
                # fill, how wide is the hole it sits in.  NOT the mean run length --
                # that averages a few wide silhouette holes together with hundreds of
                # tiny ones and tells you nothing about whether wide holes exist.
                mb = m.unsqueeze(0) > 0.5
                if mb.any():
                    rl = hole_run_length(~mb)[mb]
                    # Histogram, not the raw values: 500 images is already hundreds of
                    # millions of masked pixels, which blows past torch.quantile's
                    # ~16M input cap and would hold GBs of RAM for a few percentiles.
                    hole_hist += torch.bincount(rl.clamp(max=HIST_BINS - 1).long(),
                                                minlength=HIST_BINS)[:HIST_BINS]
                    rmax = int(rl.max().item())
                    if rmax > widest[0]:
                        widest = (rmax, float(divergence), int(cc.shape[-1]))
                base = path.join(out_dir, f"{args.prefix}_{args.model_type}_{uid}_{tag}")
                futures.append(pool.submit(
                    save_pair, cc[0].cpu(), m, base, args.size, args.num_samples,
                    detail=detail_map(cc[0].cpu()), min_detail=args.min_detail,
                    stats=detail_scores))
            n_done += 1

            if n_done % 25 == 0 or idx + 1 == len(sources):
                el = time.time() - t_start
                rate = n_done / el if el else 0
                remain = (len(sources) - n_skip - n_done) / rate if rate else 0
                print(f"  [{idx + 1}/{len(sources)}] {n_done} done, {n_skip} skipped | "
                      f"{rate:.2f} img/s | depth {t_depth / n_done * 1000:.0f}ms "
                      f"mask {t_mask / n_done * 1000:.0f}ms/img | "
                      f"eta {remain / 60:.1f} min")
    finally:
        for f in futures:
            n_pairs += f.result() or 0
        pool.shutdown()

    el = time.time() - t_start
    # Reporting must never lose a finished run: the pairs are already on disk
    # by this point, and a stats bug here once threw away the summary of a
    # completed job.
    try:
        if args.min_detail > 0:
            print(f"(crops below --min-detail {args.min_detail} were dropped)")
        if n_flat:
            print(f"\n{n_flat} image(s) rejected as too flat (< {args.min_parallax}% mask at "
                  f"reference divergence 8)")
        took = f"{el:.0f} s" if el < 120 else f"{el / 60:.1f} min"
        print(f"\n{n_pairs} pairs from {n_done} images ({n_skip} already present) in {took}")
        if n_done:
            print(f"throughput: {n_done / el:.2f} img/s  "
                  f"({t_depth / n_done * 1000:.0f} ms depth, {t_mask / n_done * 1000:.0f} ms mask, "
                  f"rest is IO)")
            print(f"projected: 10k images ~ {10000 / (n_done / el) / 3600:.1f} h, "
                  f"50k ~ {50000 / (n_done / el) / 3600:.1f} h")
        if frame_w:
            import statistics
            print(f"\nframe width actually used: min={min(frame_w)} median="
                  f"{int(statistics.median(frame_w))} max={max(frame_w)}"
                  + (f"  ({n_capped} capped to the source's own width)" if n_capped else ""))
            print(f"  ceiling at divergence {args.divergence[1]:.0f}: "
                  f"{0.85 * args.divergence[1] * 0.005 * max(frame_w):.0f}px hole for a strong silhouette")
        if div_used:
            dv = sorted(div_used)
            qd = lambda f: dv[min(len(dv) - 1, int(len(dv) * f))]     # noqa: E731
            print(f"divergence sampled (power={args.divergence_power}): median={qd(.5):.1f} "
                  f"p90={qd(.9):.1f}  {sum(1 for d in dv if d >= 12) / len(dv) * 100:.0f}% "
                  f"at >= 12")
        total = int(hole_hist.sum().item())
        if total:
            cum = torch.cumsum(hole_hist, 0)
            q = lambda f: int(torch.searchsorted(cum, int(total * f)).item())   # noqa: E731
            print(f"\nhole width (area-weighted: for each masked pixel, how wide is its hole)")
            print(f"  p10={q(.1)}px  median={q(.5)}px  p90={q(.9)}px  p99={q(.99)}px  "
                  f"max={widest[0]}px")
            for t in (32, 64, 128):
                print(f"  {hole_hist[t:].sum().item() / total * 100:5.1f}% of masked pixels "
                      f"sit in holes >= {t}px wide")
            print(f"  training feeds the model {args.train_crop}px crops, so a {q(.99)}px "
                  f"hole spans {q(.99) / args.train_crop * 100:.0f}% of its input "
                  f"(above ~40% is not plausibly fillable)")
            print(f"  widest hole seen: {widest[0]}px (divergence {widest[1]:.1f}, "
                  f"frame {widest[2]}px wide)")
        total_bytes = 0
        for s in ("train", "eval"):
            files = os.listdir(path.join(args.output, s))
            n = len([f for f in files if f.endswith("_M.png")])
            total_bytes += sum(path.getsize(path.join(args.output, s, f)) for f in files)
            print(f"  {s}/: {n} pairs")
        if n_done and total_bytes:
            per_img = total_bytes / n_done
            print(f"disk: {total_bytes / 1e6:.0f} MB for {n_done} images "
                  f"({per_img / 1e6:.1f} MB each) -> 10k ~ {per_img * 10000 / 1e9:.0f} GB, "
                  f"50k ~ {per_img * 50000 / 1e9:.0f} GB at --size {args.size}")
        print(f"\ntrain with:\n  python train.py inpaint -i \"{args.output}\" --model-dir models/inpaint_v2/")
    except Exception as ex:                                       # noqa: BLE001
        print(f"\n(summary failed: {ex!r} -- the dataset itself is written and fine)")


if __name__ == "__main__":
    main()
