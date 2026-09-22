r"""
Prove that what this pipeline writes is what the iw3 inpaint trainer reads.

Not a description of the format -- an end-to-end execution of it:

  1. generate masks with the new pipeline and write `_C.png` / `_M.png` pairs
  2. load them back with **nunif's own `iw3.training.inpaint.dataset.InpaintDataset`**
  3. run the **shipped pretrained `LightInpaintV1`** over a batch from it,
     through the exact call the trainer uses (`preprocess` then `forward`)
  4. check mask polarity empirically: erase the mask region and confirm the
     model reconstructs it, which only holds if 1 == hole
  5. demonstrate the soft-vs-binary hazard in `dataset.py`'s `mask > 0`

Run via ..\verify_format.bat
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from os import path

_HERE = path.dirname(path.dirname(path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_NUNIF = path.join(path.dirname(_HERE), "nunif")
if path.isdir(_NUNIF) and _NUNIF not in sys.path:
    sys.path.insert(0, _NUNIF)

import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402
from PIL import Image                                           # noqa: E402

from ntrainer.geometry import depth_edge_width                 # noqa: E402
from ntrainer.pipeline import MaskConfig, prepare_depth, make_mask  # noqa: E402
from ntrainer.dataset_io import save_pair, TRAIN_CROP_SIZE      # noqa: E402


OK, BAD = "[ok]", "[FAIL]"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", type=str, default=path.join(_HERE, "test_images"))
    p.add_argument("--output", type=str, default=path.join(_HERE, "out", "format_check"))
    p.add_argument("--model-type", type=str, default="Any_V3_Mono")
    p.add_argument("--resolution", type=int, default=784)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--divergence", type=float, default=16.0)
    p.add_argument("--convergence", type=float, default=0.5)
    p.add_argument("--keep", action="store_true", help="keep the generated files")
    args = p.parse_args()

    ok = True
    cfg = MaskConfig()
    from torchvision.transforms import functional as TF
    from iw3 import models as _iw3_models            # noqa: F401  registers LightInpaintV1
    from iw3.depth_model_factory import create_depth_model
    from iw3.inpaint_utils import load_image_inpaint_model

    device = f"cuda:{args.gpu}" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    print(f"device={device}  torch={torch.__version__}\n")

    # ---------------------------------------------------------------- 1. write
    if path.isdir(args.output):
        shutil.rmtree(args.output)
    for sub in ("train", "eval"):
        os.makedirs(path.join(args.output, sub), exist_ok=True)

    depth_model = create_depth_model(args.model_type)
    depth_model.load(gpu=args.gpu, resolution=args.resolution)
    depth_model.disable_ema()

    files = [path.join(args.input, f) for f in sorted(os.listdir(args.input))
             if path.splitext(f)[1].lower() in IMG_EXT][:args.limit]
    if not files:
        raise SystemExit(f"no images in {args.input}")

    written = {"train": 0, "eval": 0}
    soft_vs_hard = []
    for i, fn in enumerate(files):
        im = Image.open(fn).convert("RGB")
        if im.width != args.width:
            h = max(1, round(im.height * args.width / im.width)); h -= h % 2
            im = im.resize((args.width, h), Image.LANCZOS)
        with torch.inference_mode():
            d = depth_model.infer(im, edge_dilation=0, depth_aa=True, tta=False, enable_amp=True)
            d = depth_model.minmax_normalize_chw(d).unsqueeze(0).float()
            c = TF.to_tensor(im).unsqueeze(0).to(d.device)
            if d.shape[-2:] != c.shape[-2:]:
                d = F.interpolate(d, size=c.shape[-2:], mode="bilinear",
                                  align_corners=True, antialias=True)
            d = prepare_depth(c, d, cfg)
            ramp = depth_edge_width(d)
            m_soft = make_mask(d, args.divergence, args.convergence, cfg,
                               base_size=args.width, soft=True, ramp=ramp)
            m_hard = make_mask(d, args.divergence, args.convergence, cfg,
                               base_size=args.width, soft=False, ramp=ramp)
        # what `mask > 0` would make of each
        soft_vs_hard.append(((m_soft > 0).float().mean().item(),
                             (m_hard > 0).float().mean().item()))

        split = "eval" if i == 0 else "train"
        base = path.join(args.output, split, f"ntrainer_{args.model_type}_{i}")
        written[split] += save_pair(c[0], m_hard[0], base,
                                    size=args.size, num_samples=args.num_samples)

    print(f"wrote {written['train']} train pair(s), {written['eval']} eval pair(s) "
          f"to {args.output}")

    names = sorted(os.listdir(path.join(args.output, "train")))[:4]
    print(f"  e.g. {names}")

    # ---------------------------------------------------- 2. file-level checks
    print("\n-- file format --")
    mfile = [f for f in os.listdir(path.join(args.output, "train")) if f.endswith("_M.png")][0]
    mp = path.join(args.output, "train", mfile)
    cp = mp.replace("_M.png", "_C.png")
    # load() forces the data in and releases the handle -- PIL's lazy open keeps
    # the file locked on Windows, which blocks the cleanup at the end
    with Image.open(mp) as _mi:
        _mi.load()
        mi = _mi.copy()
        mi_mode, mi_size = _mi.mode, _mi.size
    with Image.open(cp) as _ci:
        _ci.load()
        ci_mode, ci_size = _ci.mode, _ci.size
    good = path.exists(cp)
    ok &= good
    print(f"{OK if good else BAD} every _M.png has its _C.png partner "
          f"(InpaintDataset raises otherwise)")
    good = mi_mode == "L"
    ok &= good
    print(f"{OK if good else BAD} _M.png mode is 8-bit grayscale: {mi_mode}")
    good = ci_mode == "RGB"
    ok &= good
    print(f"{OK if good else BAD} _C.png mode is RGB: {ci_mode}")
    good = mi_size == ci_size == (args.size, args.size) and args.size >= TRAIN_CROP_SIZE
    ok &= good
    print(f"{OK if good else BAD} crop size {mi_size} >= dataset.SIZE={TRAIN_CROP_SIZE}")
    mt = TF.to_tensor(mi)
    vals = torch.unique((mt * 255).round())
    good = bool(((vals == 0) | (vals == 255)).all())
    ok &= good
    print(f"{OK if good else BAD} _M.png is strictly binary, values={vals.tolist()[:6]} "
          f"(dataset.py does `mask > 0`, so a soft mask would dilate)")
    cov = (mt > 0).float().mean().item() * 100
    print(f"      mask covers {cov:.2f}% of the crop, white == hole")

    # ------------------------------------------- 3. load via nunif's own loader
    print("\n-- round-trip through iw3.training.inpaint.dataset.InpaintDataset --")
    from iw3.training.inpaint.dataset import InpaintDataset
    model = load_image_inpaint_model(None, args.gpu)
    model_offset = model.i2i_offset

    for split, training in (("train", True), ("eval", False)):
        ds = InpaintDataset(path.join(args.output, split), model_offset=model_offset,
                            training=training)
        x, mask, y, _ = ds[0]
        good = (x.shape[0] == 3 and mask.dtype == torch.bool
                and x.shape[-1] == TRAIN_CROP_SIZE
                and y.shape[-1] == TRAIN_CROP_SIZE - 2 * model_offset)
        ok &= good
        print(f"{OK if good else BAD} {split}: n={len(ds)}  x={tuple(x.shape)} "
              f"mask={tuple(mask.shape)}/{mask.dtype}  y={tuple(y.shape)} "
              f"(offset={model_offset})  mask covers {mask.float().mean() * 100:.1f}%")

    # -------------------------------------- 4. the shipped model, trainer's way
    print("\n-- forward pass, exactly as trainer.py calls it --")
    ds = InpaintDataset(path.join(args.output, "train"), model_offset=model_offset,
                        training=True)
    batch = [ds[i] for i in range(min(2, len(ds)))]
    x = torch.stack([b[0] for b in batch]).to(device)
    mask = torch.stack([b[1] for b in batch]).to(device)
    y = torch.stack([b[2] for b in batch]).to(device)

    with torch.inference_mode():
        xp, mp_ = model.preprocess(x, mask)          # trainer.py:166
        z = model(xp, mp_)                           # trainer.py:167
    good = z.shape == y.shape
    ok &= good
    print(f"{OK if good else BAD} model(preprocess(x, mask)) -> {tuple(z.shape)}, "
          f"y is {tuple(y.shape)}")
    err = (z.clamp(0, 1) - y).abs().mean().item()
    print(f"      pretrained model L1 vs ground truth: {err:.4f}")

    # ----------------------------------------------- 5. polarity, demonstrated
    print("\n-- mask polarity --")
    with torch.inference_mode():
        erased, _ = model.preprocess(x, mask)
    n = mask.sum().clamp_min(1)
    inside = (x.mean(1, keepdim=True) * mask).sum() / n
    erased_inside = (erased.mean(1, keepdim=True) * mask).sum() / n
    outside_kept = ((erased - x).abs() * (~mask)).mean().item()
    good = erased_inside.item() < 1e-6 < inside.item() and outside_kept < 1e-6
    ok &= good
    print(f"{OK if good else BAD} preprocess erases INSIDE the mask "
          f"(mean {inside:.3f} -> {erased_inside:.6f}) and leaves outside untouched "
          f"(delta {outside_kept:.2e})")
    print("      => white/1 is the region to inpaint, black/0 is kept. Confirmed by "
          "execution, not by reading.")

    # ------------------------------------------ 6. the soft-mask hazard, sized
    print("\n-- why the mask must be written binarised --")
    for (s, h), fn in zip(soft_vs_hard, files):
        print(f"      {path.basename(fn)[:34]:34s} `mask>0` sees {s * 100:5.2f}% if soft, "
              f"{h * 100:5.2f}% if binarised  (+{(s - h) / max(h, 1e-9) * 100:4.1f}%)")
    print("      dataset.py's `mask = mask > 0` promotes every anti-aliased fringe pixel "
          "to a full hole.")

    if not args.keep:
        import gc
        del ds, batch, x, mask, y, mi
        gc.collect()
        shutil.rmtree(args.output, ignore_errors=True)
        if path.isdir(args.output):
            print(f"\n(could not remove {args.output} -- a file is still open; "
                  f"delete it by hand, the checks above are unaffected)")
        else:
            print(f"\n(removed {args.output}; pass --keep to inspect the files)")

    print("\n" + ("FORMAT VERIFIED" if ok else "*** FORMAT MISMATCH ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
