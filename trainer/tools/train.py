r"""
Train the inpainting model.

  train.bat <dataset-dir> <model-dir> [options]

<dataset-dir> is what make_dataset.bat produced: it must contain train/ and
eval/ full of _C.png / _M.png pairs.

Defaults are iw3's inpaint defaults except for the three things this project
changes on purpose: the arch (nt_inpaint_v2_b instead of light_inpaint_v1), the
optimizer (CAME instead of AdamW) and the crop size (settable).

NOTE: this GUI copy adds --eval-samples and --save-epoch-step;
New_Trainer\tools\train.py has neither.

Quick smoke run on a small dataset:
  train.bat data\dataset_test models\test --num-samples 2000 --max-epoch 6 --eval-step 1

Watch it while it runs:
  monitor.bat models\test
"""
from __future__ import annotations

import argparse
import os
import sys
from os import path

_HERE = path.dirname(path.abspath(__file__))
_ROOT = path.dirname(_HERE)
for _p in (_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# iw3's own defaults for the inpaint task, from
# iw3/training/inpaint/trainer.py register(). Repeated here (rather than
# imported) because that function only exists as a subparser factory.
IW3_DEFAULTS = dict(
    batch_size=8,
    backward_step=4,          # effective batch 8*4 = 32
    learning_rate=1e-4,
    learning_rate_cosine_min=1e-8,
    scheduler="cosine_wd",
    learning_rate_cycles=5,
    max_epoch=200,
    learning_rate_decay=0.99,
    learning_rate_decay_step=[1],
    momentum=0.9,
    weight_decay=0.001,
    weight_decay_end=0.01,
    eval_step=4,
    ignore_nan=True,
    seed=-1,
)

# CAME's README: "0.5-0.9x the AdamW learning rate". 0.8 of iw3's 1e-4.
CAME_LR_SCALE = 0.8


def build_parser():
    from nunif.training.trainer import create_trainer_default_parser

    parser = create_trainer_default_parser()
    # --optimizer is declared with a fixed `choices` list upstream, so extend it
    # in place rather than redeclaring the argument.
    for action in parser._actions:
        if action.dest == "optimizer" and action.choices is not None:
            action.choices = list(action.choices) + ["came"]

    parser.add_argument("--arch", type=str, default="inpaint.nt_inpaint_v2_b",
                        help="model architecture. With --video the image arch names are "
                             "mapped to their video counterparts automatically")
    parser.add_argument("--video", action="store_true",
                        help="train the temporal model on clip folders instead of the "
                             "image model on loose pairs")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="training crop size. Must be >= 128 and a multiple of 64. "
                             "Bigger means more context around wide holes, and more VRAM")
    parser.add_argument("--num-samples", type=int, default=20000,
                        help="crops drawn per epoch (sampled with replacement, so this is "
                             "independent of dataset size)")
    parser.add_argument("--loss", type=str, default=None,
                        choices=["dct", "l1lpips", "l1dinov2", "l1dctlpips",
                                 "temporal_l1lpips", "temporal_l1dctlpips"],
                        help="loss. Default l1lpips, or temporal_l1lpips with --video")
    parser.add_argument("--discriminator", type=str, default=None,
                        choices=["l3c", "l3ce", "ffc", "ffce", "l3cffce"],
                        help="optional GAN discriminator. Adds a critic that judges "
                             "whether a filled hole looks real, which buys sharpness "
                             "that L1+LPIPS cannot: those average over everything the "
                             "hole could plausibly contain, and an average of plausible "
                             "fills is a blurry fill")
    parser.add_argument("--discriminator-optimizer", type=str, default=None,
                        help="optimizer for the discriminator only. Default adamw; "
                             "'came' shares the generator's")
    parser.add_argument("--discriminator-lr", type=float, default=None,
                        help="learning rate for the discriminator only. "
                             "Default: the same as the generator's")
    parser.add_argument("--generator-warmup-iteration", type=int, default=500,
                        help="warm-up iterations for the discriminator loss on the generator")
    parser.add_argument("--discriminator-weight", type=float, default=0.2)
    parser.add_argument("--adaptive-weight-interval", type=int, default=8,
                        help="recompute the GAN adaptive weight every Nth step "
                             "instead of every step. It is smoothed into a 100-step "
                             "EMA either way, and measuring it costs an extra "
                             "backward through the critic. 1 = upstream behaviour")
    parser.add_argument("--save-eval-step", type=int, default=0,
                        help="save a preview every Nth eval batch. 0 = auto: spread "
                             "--eval-previews of them over the eval pass, which is the "
                             "only thing that works when the eval set is small")
    parser.add_argument("--eval-previews", type=int, default=8,
                        help="preview images kept per eval pass, spread evenly over the "
                             "eval set. In video mode each one is a different clip")
    parser.add_argument("--eval-preview-rows", type=int, default=4,
                        help="rows per preview image, spread evenly over the batch. In video "
                             "mode a batch is one clip, so 4 rows are frames 0/3/7/11 of it. "
                             "One row per batch item would be a 1408x5632 PNG, several MB, "
                             "hundreds of times over a long run")
    parser.add_argument("--disable-hard-example", action="store_true",
                        help="disable hard example mining")
    parser.add_argument("--eval-samples", type=int, default=256,
                        help="samples measured per eval pass, spread evenly over the "
                             "eval set and identical every time so the curve is "
                             "comparable. 0 = use the whole eval set")
    parser.add_argument("--save-epoch-step", type=int, default=0,
                        help="with --save-epoch, keep a numbered copy only every Nth "
                             "epoch. 0 = every epoch (nunif's behaviour)")

    parser.add_argument("--lpips-frames", type=int, default=4,
                        help="video only: frames per clip the LPIPS term is measured on, "
                             "redrawn at random each step. Measured at a 384 crop, LPIPS "
                             "runs a VGG16 over prediction and target for all 12 frames "
                             "and costs 3.64 TFLOP a step against the model's own 2.27 -- "
                             "62%% of the compute. Frames in one clip are near-identical, "
                             "so a random subset is the same gradient in expectation with "
                             "more per-step noise, which 400+ steps an epoch average out. "
                             "0 = all frames (nunif's behaviour)")

    # ---- throughput -------------------------------------------------------
    # None of these change what the model learns. They are separate flags so a
    # run can be reproduced exactly, and so each can be measured on its own.
    parser.add_argument("--no-cudnn-benchmark", action="store_true",
                        help="do not let cuDNN pick the fastest algorithm for these "
                             "shapes, and leave TF32 off. Only useful if you suspect "
                             "the autotuner is misbehaving")
    parser.add_argument("--no-persistent-workers", action="store_true",
                        help="respawn the data loader workers every epoch, as nunif "
                             "does. On Windows each respawn is a process start plus an "
                             "`import torch`, so this is slower; it exists as an escape "
                             "hatch if a worker leaks memory over a long run")
    parser.add_argument("--channels-last", action="store_true",
                        help="use channels-last memory layout. This helps convolutional "
                             "models on tensor cores and does nothing (or slightly "
                             "hurts) for the mostly-MLP blocks here -- BENCHMARK IT, do "
                             "not switch it on by assumption")
    parser.add_argument("--compile", dest="compile_model", action="store_true",
                        help="torch.compile the model. This install has triton 3.7 so it "
                             "can work, but the first epoch pays minutes of compilation "
                             "and a shape it has not seen recompiles. Experimental: "
                             "measure a short run before committing three days to it")

    parser.add_argument("--came-beta2", type=float, default=0.999)
    parser.add_argument("--came-beta3", type=float, default=0.9999,
                        help="CAME confidence EMA. Must be > beta2; the paper suggests "
                             "0.9995-0.99995 when beta1/beta2 are 0.9/0.999")
    parser.add_argument("--came-eps1", type=float, default=1e-30)
    parser.add_argument("--came-eps2", type=float, default=1e-16)
    parser.add_argument("--came-clip-threshold", type=float, default=1.0,
                        help="RMS clip applied to the update before momentum")
    parser.add_argument("--came-warmup-steps", type=int, default=1000,
                        help="linear LR warmup in optimizer steps. CAME does not bias-correct "
                             "its confidence term, so its first step moves ~11x further than "
                             "intended and it takes ~1000 steps to settle; the paper's own setup "
                             "uses warmup. 0 = published behaviour exactly")

    # Upstream makes -i/--data-dir and --model-dir required. Accept them as bare
    # positionals too, so `train.bat <data> <out>` works like the other tools.
    for action in parser._actions:
        if action.dest in {"data_dir", "model_dir"}:
            action.required = False
    parser.add_argument("data_dir_pos", type=str, nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("model_dir_pos", type=str, nargs="?", help=argparse.SUPPRESS)
    parser.set_defaults(optimizer="came", **IW3_DEFAULTS)
    return parser


def resolve(base, p):
    return p if path.isabs(p) else path.abspath(path.join(base, p))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = build_parser()
    args = parser.parse_args(argv)

    base = os.environ.get("NT_CWD") or os.getcwd()
    spare = [p for p in (args.data_dir_pos, args.model_dir_pos) if p]
    if not args.data_dir and spare:
        args.data_dir = spare.pop(0)
    if not args.model_dir and spare:
        args.model_dir = spare.pop(0)
    if not args.data_dir:
        parser.error("give the dataset directory (containing train/ and eval/), "
                     "e.g. train.bat data\\dataset_test models\\test")
    if not args.model_dir:
        parser.error("give the output model directory, "
                     "e.g. train.bat data\\dataset_test models\\test")
    args.data_dir = resolve(base, args.data_dir)
    args.model_dir = resolve(base, args.model_dir)

    if args.crop_size % 64 != 0 or args.crop_size < 128:
        parser.error(f"--crop-size {args.crop_size} must be a multiple of 64 and >= 128")
    if args.crop_size % 128 != 0:
        # The three-scale model aligns to 128, so anything else is padded up and
        # you pay the larger size's compute while training on the smaller one.
        padded = args.crop_size + ((-args.crop_size) % 128)
        print(f"note   : --crop-size {args.crop_size} is padded to {padded} internally, so it "
              f"costs the same as --crop-size {padded} but trains on less context. "
              f"Use a multiple of 128 ({padded - 128} or {padded}).")

    if args.video:
        from ntrainer.train.trainer import VIDEO_ARCH
        if args.arch in VIDEO_ARCH:
            print(f"note   : --video -- arch {args.arch} -> {VIDEO_ARCH[args.arch]}")
            args.arch = VIDEO_ARCH[args.arch]
        elif "video" not in args.arch:
            parser.error(f"--video with --arch {args.arch}, which is not a temporal model. "
                         f"Use one of: {', '.join(sorted(VIDEO_ARCH.values()))}")
    if args.loss is None:
        args.loss = "temporal_l1lpips" if args.video else "l1lpips"
    if args.video and not args.loss.startswith("temporal_"):
        print(f"note   : --loss {args.loss} has no temporal term, so the temporal blocks "
              f"have nothing asking them to reduce flicker")

    # CAME wants a lower lr than AdamW; only auto-scale when the user did not
    # set one, and say so, so the number in the log is never a surprise.
    lr_given = any(a in ("--learning-rate",) or a.startswith("--learning-rate=") for a in argv)
    if args.optimizer == "came" and not lr_given:
        args.learning_rate = IW3_DEFAULTS["learning_rate"] * CAME_LR_SCALE
        print(f"note   : CAME -- learning rate set to {args.learning_rate:g} "
              f"({CAME_LR_SCALE:g}x iw3's AdamW default {IW3_DEFAULTS['learning_rate']:g}); "
              f"override with --learning-rate")
    if args.optimizer == "came" and not (args.came_beta3 > args.came_beta2):
        parser.error(f"--came-beta3 ({args.came_beta3}) must be greater than "
                     f"--came-beta2 ({args.came_beta2})")

    # nunif's cosine scheduler does `T_0 = max_epoch // cycles` and then
    # `max_epoch -= (max_epoch % cycles) + 1`. With the default 5 cycles and a
    # short run that is T_0=0 (torch raises) or a max_epoch that lands below the
    # epochs you asked for. Keep at least 8 epochs per cycle.
    if args.scheduler in {"cosine", "cosine_wd", "cosine_fixed_wd"}:
        cycles = min(args.learning_rate_cycles, max(1, args.max_epoch // 8))
        if cycles != args.learning_rate_cycles:
            print(f"note   : --learning-rate-cycles {args.learning_rate_cycles} -> {cycles} "
                  f"(only {args.max_epoch} epochs; a cosine cycle needs room to anneal)")
            args.learning_rate_cycles = cycles

    # In video mode one loader item is a whole clip, so a "batch" is seq frames
    # and batch_size is forced to 1 -- the step count is different.
    if args.video:
        from ntrainer.models import SEQ_LEN
        steps_per_epoch = max(1, (args.num_samples // SEQ_LEN) // args.backward_step)
    else:
        steps_per_epoch = max(1, args.num_samples // (args.batch_size * args.backward_step))
    total_steps = steps_per_epoch * args.max_epoch
    if args.optimizer == "came" and args.came_warmup_steps > total_steps // 4:
        capped = max(1, total_steps // 4)
        print(f"note   : --came-warmup-steps {args.came_warmup_steps} -> {capped} "
              f"(a quarter of this run's {total_steps} steps)")
        args.came_warmup_steps = capped

    args.diff_aug = args.loss not in {"dct"}

    os.makedirs(args.model_dir, exist_ok=True)
    print(f"data   : {args.data_dir}")
    print(f"out    : {args.model_dir}")
    if args.video:
        from ntrainer.models import SEQ_LEN
        print(f"mode   : video, clips of {SEQ_LEN} frames")
        print(f"crop   : {args.crop_size}  1 clip x {args.backward_step} "
              f"= {SEQ_LEN * args.backward_step} frames effective")
    else:
        print(f"crop   : {args.crop_size}  batch {args.batch_size} x {args.backward_step} "
              f"= {args.batch_size * args.backward_step} effective")
    print(f"epochs : {args.max_epoch}, {args.num_samples} samples each, eval every {args.eval_step}")
    print(f"loss   : {args.loss}" + (f" + {args.discriminator} discriminator" if args.discriminator else ""))

    from ntrainer.train.trainer import NTInpaintTrainer
    NTInpaintTrainer(args).fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
