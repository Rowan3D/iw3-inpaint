r"""
Measure the model on YOUR GPU before committing to a size.

  bench_model.bat                       inference at 1080p, all sizes
  bench_model.bat --res 2160            4K
  bench_model.bat --train --crop 256 --batch 8      training step time + VRAM

Inference numbers are what the iw3 app will feel; the training numbers tell you
whether a batch fits in VRAM and roughly how long an epoch will take.

Both are measured, not estimated: warm-up passes first, then CUDA-synchronised
timing, and `torch.cuda.max_memory_allocated` for VRAM.
"""
from __future__ import annotations

import argparse
import sys
import time
from os import path

_HERE = path.dirname(path.abspath(__file__))
_ROOT = path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

ARCHS = ["inpaint.light_inpaint_v1", "inpaint.nt_inpaint_v2_s",
         "inpaint.nt_inpaint_v2_b", "inpaint.nt_inpaint_v2_l"]


def sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_infer(name, height, width, batch, iters, device, dtype):
    import torch
    from nunif.models import create_model

    model = create_model(name).to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    x = torch.rand((batch, 3, height, width), device=device)
    mask = torch.zeros((batch, 1, height, width), device=device)
    mask[:, :, :, width // 3:width // 3 + max(8, width // 24)] = 1

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype,
                                                enabled=device.type == "cuda"):
        for _ in range(3):
            model.infer(x, mask)
        sync(device)
        t = time.time()
        for _ in range(iters):
            model.infer(x, mask)
        sync(device)
        el = time.time() - t
    vram = (torch.cuda.max_memory_allocated(device) / 1024 ** 2) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return params, (batch * iters) / el, vram


def _build_criterion(video, lpips_frames, device):
    """The criterion training actually uses, not a stand-in.

    The benchmark measured an L1 loss, so its "training step" left out the
    perceptual term entirely -- and that term runs a VGG16 over both the
    prediction and the target for every frame. Measured on nt_video_inpaint_v2_b
    at a 384 crop it is 3.64 TFLOP a step against the model's own 2.27, so a
    benchmark without it reports roughly a third of the real cost and makes a
    GPU-bound run look data-bound.
    """
    import torch
    from nunif.modules.weighted_loss import WeightedLoss
    from nunif.modules.clamp_loss import ClampLoss
    from iw3.training.inpaint.trainer import TemporalGradientLoss
    from ntrainer.train.trainer import TemporalSmoothingPenalty, SampledLPIPSWith

    if video:
        base = WeightedLoss(
            (ClampLoss(torch.nn.L1Loss()), TemporalGradientLoss(), TemporalSmoothingPenalty()),
            weights=(0.8, 0.2, 0.01))
    else:
        base = WeightedLoss((ClampLoss(torch.nn.L1Loss()),), weights=(1.0,))
    return SampledLPIPSWith(base, weight=0.2, frames=lpips_frames if video else 0).to(device)


def bench_train(name, crop, batch, iters, device, dtype, criterion=None,
                discriminator=None, video=False):
    import torch
    import torch.nn.functional as F
    from nunif.models import create_model
    from ntrainer.optim import CAME

    model = create_model(name).to(device).train()
    params = sum(p.numel() for p in model.parameters())
    opt = CAME(model.parameters(), lr=8e-5)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and dtype == torch.float16))
    x = torch.rand((batch, 3, crop, crop), device=device)
    mask = torch.zeros((batch, 1, crop, crop), device=device)
    mask[:, :, :, crop // 3:crop // 3 + crop // 8] = 1
    y = F.pad(x, (-model.i2i_offset,) * 4)

    disc = d_opt = d_scaler = None
    d_params = 0
    if discriminator:
        from iw3.training.inpaint.trainer import create_discriminator
        disc = create_discriminator(discriminator, [0 if device.type == "cuda" else -1], device)
        disc = disc.train()
        d_params = sum(p.numel() for p in disc.parameters())
        d_opt = torch.optim.AdamW(disc.parameters(), lr=8e-5)
        d_scaler = torch.amp.GradScaler(device.type,
                                        enabled=(device.type == "cuda" and dtype == torch.float16))

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            xx, mm = model.preprocess(x, mask)
            z = model(xx, mm)
            loss = criterion(z, y) if criterion is not None else F.l1_loss(z.float().clamp(0, 1), y)
            if disc is not None:
                # The generator's side of the adversarial loss: the critic
                # judges the fake, and that gradient flows back through it.
                disc.requires_grad_(False)
                adv = disc(z.clamp(0, 1), xx, mask=None)
                adv = adv[0] if isinstance(adv, (tuple, list)) else adv
                loss = loss + adv.float().mean() * 0.2
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if disc is not None:
            # and the critic's own update, on the detached fake and the real
            disc.requires_grad_(True)
            d_opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                fake = disc(z.detach().clamp(0, 1), xx, mask=mm)
                real = disc(y, xx, mask=None)
                fake = fake[0] if isinstance(fake, (tuple, list)) else fake
                real = real[0] if isinstance(real, (tuple, list)) else real
                d_loss = (F.relu(1 + fake.float()).mean() + F.relu(1 - real.float()).mean())
            d_scaler.scale(d_loss).backward()
            d_scaler.step(d_opt)
            d_scaler.update()

    for _ in range(3):
        step()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    t = time.time()
    for _ in range(iters):
        step()
    sync(device)
    el = time.time() - t
    vram = (torch.cuda.max_memory_allocated(device) / 1024 ** 2) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return params, el / iters, vram, d_params


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--arch", type=str, nargs="*", default=ARCHS)
    p.add_argument("--res", type=int, default=1080, help="frame height; width is 16:9")
    p.add_argument("--batch", type=int, default=1, help="inference batch (frames at once)")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--train", action="store_true", help="benchmark a training step instead")
    p.add_argument("--crop", type=int, default=256, help="training crop size")
    p.add_argument("--train-batch", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    p.add_argument("--amp-float", type=str, default="fp16", choices=["fp16", "bfloat16"])
    p.add_argument("--video", action="store_true",
                   help="measure the temporal losses too, as --video training uses them")
    p.add_argument("--lpips-frames", type=int, default=4,
                   help="frames the perceptual term is measured on, matching --lpips-frames "
                        "in training. 0 = every frame")
    p.add_argument("--discriminator", type=str, default=None,
                   choices=["l3c", "l3ce", "ffc", "ffce", "l3cffce"],
                   help="also measure what adding a critic costs, as a third row")
    p.add_argument("--simple-loss", action="store_true",
                   help="measure with a plain L1 loss instead of the real criterion. "
                        "Faster, and what this tool used to do, but it leaves out the "
                        "VGG16 the perceptual term runs and so under-reports badly")
    args = p.parse_args()

    import torch
    import iw3.models  # noqa: F401  registers light_inpaint_v1
    import ntrainer.models  # noqa: F401  registers nt_inpaint_v2*

    device = torch.device("cpu" if args.gpu < 0 else f"cuda:{args.gpu}")
    dtype = torch.float16 if args.amp_float == "fp16" else torch.bfloat16
    if device.type == "cuda":
        print(f"device : {torch.cuda.get_device_name(device)} "
              f"({torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB), "
              f"autocast {args.amp_float}")
    else:
        print("device : CPU (pass --gpu 0 for the real numbers)")

    if args.train:
        print(f"\ntraining step, crop {args.crop}, batch {args.train_batch}, CAME + GradScaler")
        if args.simple_loss:
            print("loss   : plain L1 -- NOT what training does; the perceptual term is "
                  "left out")
        else:
            print(f"loss   : the real criterion"
                  + (f", perceptual on {args.lpips_frames or 'all'} frames" if args.video else ""))
        crit = None if args.simple_loss else _build_criterion(
            args.video, args.lpips_frames, device)

        # Each stage adds one thing, so the cost of that thing is the difference
        # between two rows measured the same way.
        stages = [("model + loss", None)]
        if args.discriminator:
            stages.append((f"+ critic ({args.discriminator})", args.discriminator))

        print()
        print(f"{'arch':30s} {'stage':22s} {'s/step':>9s} {'samples/s':>10s} {'peak VRAM':>11s}")
        for name in args.arch:
            base = None
            for label, disc in stages:
                try:
                    params, sec, vram, dparams = bench_train(
                        name, args.crop, args.train_batch, args.iters, device, dtype,
                        criterion=crit, discriminator=disc, video=args.video)
                    extra = ""
                    if base is not None:
                        extra = (f"   +{(sec / base[0] - 1) * 100:.0f}% time, "
                                 f"+{vram - base[1]:.0f}MB, critic {dparams / 1e6:.1f}M params")
                    else:
                        base = (sec, vram)
                    print(f"{name:30s} {label:22s} {sec:9.3f} "
                          f"{args.train_batch / sec:10.1f} {vram:10.0f}MB{extra}")
                except torch.cuda.OutOfMemoryError:
                    print(f"{name:30s} {label:22s} out of memory at this crop/batch")
                    torch.cuda.empty_cache()
        return 0

    h = args.res
    w = int(round(h * 16 / 9 / 8)) * 8
    print(f"\ninference at {w}x{h}, batch {args.batch}\n")
    print(f"{'arch':30s} {'params':>9s} {'file':>8s} {'FPS':>8s} {'peak VRAM':>11s}")
    for name in args.arch:
        try:
            params, fps, vram = bench_infer(name, h, w, args.batch, args.iters, device, dtype)
            print(f"{name:30s} {params/1e6:8.2f}M {params*4/1024/1024:7.0f}MB "
                  f"{fps:8.1f} {vram:10.0f}MB")
        except torch.cuda.OutOfMemoryError:
            print(f"{name:30s} out of memory at this resolution")
            torch.cuda.empty_cache()
    print("\nNote: iw3 runs the inpaint model on tiles, not whole frames, so real app\n"
          "throughput also depends on tile size and how much of the frame is masked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
