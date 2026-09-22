r"""
ntrainer's training loop.

It subclasses iw3's InpaintTrainer/InpaintEnv rather than replacing them, so the
loss functions, hard-example mining, EMA, GAN machinery, checkpointing and
resume behaviour are nunif's, unchanged and already proven. What is overridden:

  * `create_dataloader` -- our dataset, with a configurable crop size.
  * `create_optimizer`  -- adds CAME.
  * `save_eval`         -- writes one preview per epoch (upstream overwrites a
                           single file, so there is nothing to compare against),
                           and includes the mask as its own panel.
  * progress logging    -- `progress.csv`, rewritten whole each time, one row per
                           epoch with train loss, eval loss, lr, wd and duration.
                           This is what the web monitor reads.

Why not use nunif's own `write_log`: `Trainer.fit()` calls
`self.write_log(epoch, train_loss, loss)` where `loss` is a local that only
exists after the first eval pass, inside a bare `except: pass`. Before the first
eval it raises NameError and the row is silently dropped, and on non-eval epochs
it logs the *previous* eval's value as if it were this epoch's. Logging from
`train_end`/`eval_end` instead means every epoch is recorded exactly once, with
the eval column empty when no eval ran.
"""
from __future__ import annotations

import contextlib
import csv
import math
import os
import sys
import time
from datetime import datetime
from os import path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

from nunif.models import create_model
from nunif.training.weight_decay_config import configure_optim_groups
from iw3.training.inpaint.trainer import InpaintEnv, InpaintTrainer

from iw3.training.inpaint.dataset_video import SEQ as VIDEO_SEQ

from ..optim import CAME
from .. import models as _nt_models  # noqa: F401  (registers inpaint.nt_inpaint_v2*)
from ..models import transfer_image_weights
from .dataset import NTInpaintDataset, check_dataset
from .dataset_video import NTVideoInpaintDataset, check_video_dataset


__all__ = ["NTInpaintTrainer", "NTInpaintEnv", "get_last_layer"]

PROGRESS_CSV = "progress.csv"
PROGRESS_HEADER = ["epoch", "phase", "train_loss", "eval_loss", "lr", "weight_decay",
                   "seconds", "timestamp", "preview",
                   "recon_loss", "gen_loss", "disc_loss"]


class TemporalSmoothingPenalty(torch.nn.Module):
    """Second-order difference along TIME.

    Replaces `iw3.training.inpaint.trainer.TemporalSmoothingPenalty`, which
    indexes `input[:, :-2]` -- dim 1, the *channel* axis, not the frame axis.
    On the (SEQ, C, H, W) tensors the video path actually produces it therefore
    measures curvature across R/G/B. Measured on synthetic inputs:

        perfectly smooth in time        penalty 42.17
        alternating frames (max flicker) penalty  0.00
        constant in time, channels differ penalty 4.00

    i.e. it rewards flicker and punishes smooth motion -- backwards. Its weight
    in `temporal_l1lpips` is only 0.01, so it is a small term, but it is pulling
    the wrong way, and this is one of the two losses that exist specifically to
    make the temporal blocks do their job.
    """

    def forward(self, input, target=None):
        if input.shape[0] < 3:
            return input.new_zeros(())
        second_order = input[2:] - 2 * input[1:-1] + input[:-2]
        return (second_order ** 2).mean()


class SampledLPIPSWith:
    """LPIPS on a random subset of a clip's frames.

    Measured on this architecture at a 384 crop, one training step:

        the model itself          2.27 TFLOP
        the LPIPS VGG16           3.64 TFLOP   <- 62% of the total

    LPIPS runs a full VGG16 over the prediction and the target, all twelve
    frames, every step, and backpropagates through the prediction branch. It
    costs more than the model it is supervising.

    Frames inside one clip are at most half a second apart and are highly
    correlated, so LPIPS over a random k of them is an unbiased estimate of
    LPIPS over all twelve: the same gradient in expectation, more variance per
    step. With 400+ optimizer steps an epoch that variance averages out, and
    because the subset is redrawn every step, every frame still receives
    perceptual supervision -- just not on every step.

    The L1 and temporal terms still see the whole clip. They are cheap, and the
    temporal ones are meaningless without consecutive frames.

    Written as a wrapper rather than a subclass so it works whatever nunif's
    LPIPSWith does internally; `frames` <= 0 or a clip shorter than `frames`
    falls straight through to the original.
    """

    def __new__(cls, base_loss, weight=1.0, std_mask=False, frames=0):
        from nunif.modules.lpips import LPIPSWith
        from nunif.modules.pad import get_pad_size
        from nunif.modules.reflection_pad2d import reflection_pad2d_naive
        from nunif.modules.local_std_mask import local_std_mask

        class _Sampled(LPIPSWith):
            def forward(self, input, target):
                if frames <= 0 or input.ndim != 4 or input.shape[0] <= frames:
                    return super().forward(input, target)
                pad = get_pad_size(input, 16, random_shift=False)
                a = reflection_pad2d_naive(input, pad, detach=True)
                b = reflection_pad2d_naive(target, pad, detach=True)
                base = self.base_loss(a, b)
                idx = torch.randperm(a.shape[0], device=a.device)[:frames]
                pa, pb = a[idx], b[idx]
                if self.std_mask:
                    lp = self.lpips(local_std_mask(pa, pb), pb, normalize=True).mean()
                else:
                    lp = self.lpips(pa, pb, normalize=True).mean()
                return base + lp * self.weight

        return _Sampled(base_loss, weight=weight, std_mask=std_mask)


VIDEO_ARCH = {
    "inpaint.nt_inpaint_v2": "inpaint.nt_video_inpaint_v2",
    "inpaint.nt_inpaint_v2_s": "inpaint.nt_video_inpaint_v2_s",
    "inpaint.nt_inpaint_v2_b": "inpaint.nt_video_inpaint_v2_b",
    "inpaint.nt_inpaint_v2_l": "inpaint.nt_video_inpaint_v2_l",
}


def get_last_layer(model):
    """Final conv weight, used only for the GAN adaptive-weight balance."""
    to_image = getattr(model, "to_image", None)
    if to_image is None:
        raise NotImplementedError(f"no to_image on {model.name}")
    if isinstance(to_image, torch.nn.Sequential):
        return to_image[-1].weight
    return to_image.weight


class NTInpaintEnv(InpaintEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_started = None
        self.epoch_lr = ("", "")
        self.preview_files = []
        self.preview_step = 1

    # ---- progress log ----------------------------------------------------
    def _log(self, train_loss=None, eval_loss=None, parts=None):
        trainer = self.trainer
        rows = trainer.progress
        epoch = trainer.log_epoch()
        row = rows.setdefault(epoch, dict.fromkeys(PROGRESS_HEADER, ""))
        row["epoch"] = epoch
        row["phase"] = trainer.phase
        if train_loss is not None:
            row["train_loss"] = f"{float(train_loss):.6g}"
            row["seconds"] = f"{time.time() - (self.epoch_started or time.time()):.1f}"
            # captured in train_begin: env.train() steps the scheduler before
            # calling train_end, so reading it here would log the *next* epoch's
            row["lr"], row["weight_decay"] = self.epoch_lr
            row["timestamp"] = datetime.now().isoformat(timespec="seconds")
            for k, v in (parts or {}).items():
                row[k] = f"{float(v):.6g}"
        if eval_loss is not None:
            row["eval_loss"] = f"{float(eval_loss):.6g}"
            row["preview"] = ";".join(self.preview_files)
        trainer.write_progress()

    def train_begin(self):
        super().train_begin()
        self.epoch_started = time.time()
        try:
            group = self.trainer.optimizers[0].param_groups[0]
            self.epoch_lr = (f"{group['lr']:.6g}", f"{group.get('weight_decay', 0):.6g}")
        except Exception:                                            # noqa: BLE001
            self.epoch_lr = ("", "")

    def train_end(self):
        # Snapshot before super() clears anything, and before the GAN branch
        # folds the discriminator loss into the return value.
        steps = max(1, self.sum_step)
        gan = self.discriminator is not None
        parts = {}
        if gan:
            parts = {"recon_loss": self.sum_p_loss / steps,
                     "gen_loss": self.sum_g_loss / steps,
                     "disc_loss": self.sum_d_loss / steps}
        mean_loss = super().train_end()
        try:
            # In GAN mode upstream never increments sum_loss (only the p/g/d
            # accumulators), so its "loss" is 0 + mean discriminator loss --
            # a number that sits near 0.5 by construction and says nothing
            # about reconstruction quality. Graph the reconstruction loss
            # instead, so the curve means the same thing in every phase.
            self._log(train_loss=parts["recon_loss"] if gan else mean_loss, parts=parts)
        except Exception as e:                                       # noqa: BLE001
            print(f"warning: could not write {PROGRESS_CSV}: {e}")
        return mean_loss

    def calc_weight(self, recon_loss, generator_loss, grad_scaler):
        # Same as upstream except for which tensor counts as the last layer;
        # upstream's module-level get_last_layer() only knows iw3's own archs.
        #
        # ...and except that it is not recomputed every step. Measuring it costs
        # two extra torch.autograd.grad calls, and the second one backpropagates
        # the generator loss through the whole critic -- a real cost on a GAN
        # step. What comes out is then folded into a 100-step EMA, so 99% of
        # each measurement is averaged away immediately. Measuring every Nth
        # step and using a correspondingly larger EMA step gives the same
        # smoothing over the same number of steps for a fraction of the work.
        interval = max(1, int(getattr(self.trainer.args,
                                      "adaptive_weight_interval", 0) or 1))
        if (interval > 1 and self.adaptive_weight_ema is not None
                and (self.get_current_iteration() % interval) != 0):
            return self.adaptive_weight_ema, True

        last_layer = get_last_layer(self.model)
        weight = self.calculate_adaptive_weight(
            recon_loss, generator_loss, last_layer, grad_scaler,
            min=1e-3, max=10.0, mode="norm",
            adaptive_weight=1.0 if self.adaptive_weight_ema is None else self.adaptive_weight_ema)
        weight_is_nan = math.isnan(weight)
        if not weight_is_nan:
            if self.adaptive_weight_ema is None:
                self.adaptive_weight_ema = weight
            else:
                # 1 - 0.99**interval: one measurement every `interval` steps
                # moves the average as far as `interval` measurements at 0.01.
                alpha = 1.0 - 0.99 ** interval
                self.adaptive_weight_ema = self.adaptive_weight_ema * (1 - alpha) + weight * alpha
            weight = self.adaptive_weight_ema
        elif self.adaptive_weight_ema is not None:
            weight = self.adaptive_weight_ema
        else:
            weight = 1.0
        return weight, not weight_is_nan

    # ---- eval ------------------------------------------------------------
    def eval_begin(self):
        super().eval_begin()
        self.preview_files = []
        # Spread the previews over the whole eval pass. Upstream's fixed
        # `--save-eval-step 10` writes nothing at all when the eval set is
        # smaller than 10 batches, which is exactly the case on a small dataset.
        want = max(1, self.trainer.args.eval_previews)
        explicit = self.trainer.args.save_eval_step
        if explicit and explicit > 0:
            self.preview_step = explicit
        else:
            try:
                n = len(self.trainer.eval_loader)
            except TypeError:
                n = 0
            self.preview_step = max(1, n // want)

    def eval_step(self, data):
        # Upstream's version, plus the mask is kept so it can be drawn.
        x, mask, y, *_ = data
        x, mask, y = self.to_device(x), self.to_device(mask), self.to_device(y)
        if x.ndim == 5:
            # video: the loader adds a batch axis of 1 around the clip,
            # (1, SEQ, C, H, W) -> (SEQ, C, H, W). Dropping this is not cosmetic:
            # the model would see 1 "frame" of SEQ channels.
            assert x.shape[0] == 1
            x = x.reshape(*x.shape[1:])
            mask = mask.reshape(*mask.shape[1:])
            y = y.reshape(*y.shape[1:])
        model = self.get_eval_model()
        with self.autocast():
            x, mask = self.model.preprocess(x, mask)
            z = model(x, mask)
            loss = self.eval_criterion(z, y)
            self.eval_count += 1
            step = getattr(self, "preview_step", 1)
            if (self.eval_count % step == 0
                    and len(self.preview_files) < self.trainer.args.eval_previews):
                self.save_eval(x, y, z, self.eval_count // step, mask=mask)
        self.sum_loss += loss.item()
        self.sum_step += 1

    def save_eval(self, x, y, z, i, mask=None):
        """[ masked input | mask | prediction | ground truth ], one row per sample.

        Written as eval/epochNNNN_i.png so epochs can be compared; upstream
        overwrites eval/i.png every time, which makes progress invisible.
        """
        batch_offset = (x.shape[0] - z.shape[0]) // 2
        offset = (x.shape[2] - z.shape[2]) // 2
        x = F.pad(x, (-offset,) * 4)
        if batch_offset > 0:
            x = x[batch_offset:-batch_offset]
        panels = [x]
        if mask is not None:
            m = F.pad(mask, (-offset,) * 4)
            if batch_offset > 0:
                m = m[batch_offset:-batch_offset]
            panels.append(m.expand_as(x).to(x.dtype))
        panels += [z, y]
        # Cap the rows, and SPREAD them over the batch rather than taking the
        # first N. In video mode a batch is one clip, so the first 4 rows are
        # frames 0-3 of a 12 frame clip -- four near-identical pictures. Evenly
        # spaced gives frames 0/3/7/11, which actually shows the clip moving.
        rows = max(1, getattr(self.trainer.args, "eval_preview_rows", 4))
        n = panels[0].shape[0]
        if rows < n:
            idx = torch.linspace(0, n - 1, rows).round().long().to(panels[0].device)
            panels = [p.index_select(0, idx) for p in panels]
        grid = torch.cat([p.float().clamp(0, 1) for p in panels], dim=3)

        eval_dir = path.join(self.trainer.args.model_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        name = f"epoch{self.trainer.log_epoch():04d}_{i}.png"
        TF.to_pil_image(make_grid(grid, nrow=1)).save(path.join(eval_dir, name))
        self.preview_files.append(name)

    def eval_end(self, file=None):
        import sys
        out = file or sys.stdout
        mean_loss = self.sum_loss / max(self.sum_step, 1)
        try:
            self._log(eval_loss=mean_loss)
        except Exception as e:                                       # noqa: BLE001
            print(f"warning: could not write {PROGRESS_CSV}: {e}")
        result = super().eval_end(file=out)
        if self.discriminator is None:
            return result
        # With a discriminator, upstream's eval_end returns inf_loss() --
        # -time.time()/1e9, a number that gets smaller with the wall clock. The
        # trainer compares that against best_loss, so EVERY eval pass counts as
        # an improvement and "best model" quietly becomes "most recent model".
        # That is a defensible default when nothing is tracking reconstruction
        # quality, but we already measure it: sum_loss here is the same eval
        # criterion the non-GAN path uses, on the same fixed eval subset.
        # Returning it keeps best_loss meaningful, so a GAN that collapses in
        # the last twenty epochs cannot overwrite a good model.
        self.print_eval_result(mean_loss, file=out)
        return mean_loss


class NTInpaintTrainer(InpaintTrainer):
    def initialize(self):
        self.progress = {}
        self.progress_offset = 0
        self.phase = 1
        self._tune_backend()
        super().initialize()
        self._unwrap_ema()
        self._load_progress()

    def _unwrap_ema(self):
        """The EMA copy must be a plain Model, not a compiled wrapper.

        nunif builds it with AveragedModel(self.model); with --compile that
        deep-copies the OptimizedModule, and the EMA copy is then what
        save_best_model() hands to save_model(), which asserts isinstance(Model).
        Unwrapping keeps the same parameter tensors -- the averaging is
        untouched -- and eval runs on the uncompiled module, which also avoids
        compiling a second graph for the eval shapes.
        """
        ema = getattr(self, "ema_model", None)
        if ema is not None and hasattr(getattr(ema, "module", None), "_orig_mod"):
            ema._modules["module"] = ema.module._orig_mod

    def _tune_backend(self):
        """Backend switches nunif never sets, applied before the model is built.

        cudnn.benchmark spends the first few steps trying algorithms for each
        distinct input shape and then reuses the winner. It is a loss when
        shapes keep changing; here every step is the same clip length at the
        same crop, so it is measured once and reused for the rest of the run.

        TF32 only affects float32 matmul and convolution. Under AMP most of the
        work is already fp16, so this is about the layers that stay in float32
        -- small, but free on Ampere and later.

        Neither changes what the model learns. Both are off by default in
        PyTorch, not because they are unsafe, but because PyTorch cannot know
        the shapes are fixed.
        """
        if getattr(self.args, 'no_cudnn_benchmark', False):
            return
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception as e:                                      # noqa: BLE001
            print(f"note   : could not set backend flags ({e})")

    # ---- progress.csv ----------------------------------------------------
    def progress_path(self):
        return path.join(self.args.model_dir, PROGRESS_CSV)

    def log_epoch(self):
        """Epoch number as the curve sees it.

        nagadomi's recipe is several 200-epoch phases, each started with
        `--resume --reset-state`: the weights carry over but nunif leaves
        `start_epoch` at 1, so every phase counts 1..199 again. Logging those raw
        would overwrite the previous phase's rows one by one and throw the
        history away. Adding `progress_offset` makes phase 2 log 200..398, so
        progress.csv, the monitor's graph and the eval preview filenames all run
        continuously across the whole project.
        """
        return self.epoch + self.progress_offset

    def _load_progress(self):
        p = self.progress_path()
        if not path.exists(p):
            return
        try:
            with open(p, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if (r.get("epoch") or "").strip()]
        except Exception as e:                                        # noqa: BLE001
            print(f"warning: could not read {p}: {e}")
            return

        if not self.args.resume:
            # A fresh run writing into a directory that already has a curve: the
            # weights are new, so the old curve is not comparable to the new one.
            # Keep it, but out of the way.
            archive = path.join(self.args.model_dir, f"progress.{self.runtime_id}.csv")
            try:
                os.replace(p, archive)
                print(f"note   : training from scratch; previous curve kept as "
                      f"{path.basename(archive)}")
            except OSError as e:
                print(f"warning: could not archive {p}: {e}")
            return
        if not rows:
            return

        last = max(int(float(r["epoch"])) for r in rows)
        prev_phase = max(int(float(r.get("phase") or 1)) for r in rows)
        # start_epoch is 1 for a new phase (--reset-state) and >1 when picking a
        # crashed phase back up, so one expression covers both cases.
        self.progress_offset = max(0, last - (self.start_epoch - 1))
        new_phase = self.start_epoch <= 1 and self.progress_offset > 0
        self.phase = prev_phase + 1 if new_phase else prev_phase

        keep_below = self.start_epoch + self.progress_offset
        for r in rows:
            epoch = int(float(r["epoch"]))
            if epoch < keep_below:
                row = {k: r.get(k, "") for k in PROGRESS_HEADER}
                # progress.csv written before the phase column existed
                row["phase"] = row["phase"] or 1
                self.progress[epoch] = row

        if new_phase:
            self._archive_best_model(prev_phase)
            print(f"note   : phase {self.phase} continues the curve at epoch "
                  f"{self.progress_offset + 1} (weights kept, LR schedule restarts)")
        elif self.progress_offset:
            print(f"note   : resuming phase {self.phase} at epoch {keep_below}")

    def _archive_best_model(self, prev_phase):
        """--reset-state zeroes best_loss, so the first eval of the new phase
        overwrites the best-model file even when it is worse. Keep a copy of what
        the finished phase produced."""
        import shutil
        src = self.best_model_filename
        dst = f"{path.splitext(src)[0]}.phase{prev_phase}.pth"
        if path.exists(src) and not path.exists(dst):
            try:
                shutil.copy2(src, dst)
                print(f"note   : phase {prev_phase}'s best model saved as {path.basename(dst)}")
            except OSError as e:
                print(f"warning: could not back up {src}: {e}")

    def write_progress(self):
        p = self.progress_path()
        tmp = p + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=PROGRESS_HEADER)
            w.writeheader()
            for epoch in sorted(self.progress):
                w.writerow(self.progress[epoch])
        os.replace(tmp, p)

    def write_log(self, epoch, train_loss, eval_loss):
        # progress.csv is written from the env; keep nunif's loss_*.csv too.
        try:
            super().write_log(epoch, train_loss, eval_loss)
        except Exception:                                            # noqa: BLE001
            pass

    # ---- model / data ----------------------------------------------------
    def create_model(self):
        model = create_model(self.args.arch, device_ids=self.args.gpu)
        model = model.to(self.device)
        n = sum(p.numel() for p in model.parameters())
        print(f"model  : {model.name}  {n / 1e6:.2f}M params ({n * 4 / 1024 / 1024:.1f} MB fp32)")
        if getattr(self.args, "channels_last", False):
            model = model.to(memory_format=torch.channels_last)
            print("note   : channels-last layout on. Confirm with the benchmark that it "
                  "is actually faster for this arch before trusting it")
        if getattr(self.args, "compile_model", False):
            model = self._try_compile(model)
        return model

    def _try_compile(self, model):
        """Compile at the REAL training shape, with a backward, before the run starts.

        Two things make a naive `torch.compile(model)` a trap here.

        It is LAZY: it returns a wrapper immediately and does the work on the
        first forward, so try/except around the call catches nothing -- the
        failure lands inside the training loop and kills the run.

        And it compiles PER SHAPE, forward and backward separately. A probe at a
        token size therefore proves almost nothing: the real 384-crop forward
        and its backward still compile from scratch on the first step, several
        silent minutes into epoch 1, behind a progress bar stuck at 0/1666 that
        is indistinguishable from a hang.

        So the probe uses the crop and clip length this run will actually use,
        and runs a real backward. The compile cost is paid here, out loud, with
        a timer -- and if the toolchain is broken it is caught here too.
        """
        import time as _time
        crop = int(getattr(self.args, "crop_size", 256) or 256)
        seq = getattr(model, "seq_len", 0) or 0
        n = seq if (getattr(self.args, "video", False) and seq) else \
            int(getattr(self.args, "batch_size", 1) or 1)
        print(f"note   : compiling for {n}x{crop}x{crop} (forward and backward). This is "
              f"done once,")
        print("note   : here rather than inside epoch 1, and can take SEVERAL MINUTES. "
              "Nothing is wrong.")
        sys.stdout.flush()
        t0 = _time.time()
        # Inductor "donates" buffers it believes the backward will not need
        # again, which forbids retain_graph=True. The GAN's adaptive weight
        # calls torch.autograd.grad(..., retain_graph=True) on the last layer
        # every step, so with a critic on, a compiled run dies on step 1 with
        # "compiled with non-empty donated buffers". Turning the optimisation
        # off costs a little memory and keeps both features usable together.
        try:
            import torch._functorch.config as _functorch_config
            _functorch_config.donated_buffer = False
        except Exception:                                           # noqa: BLE001
            pass
        try:
            compiled = torch.compile(model)
            # Own generator: the probe must not consume the global RNG, or the
            # same seed gives a different run with compile on than with it off.
            g = torch.Generator(device=self.device).manual_seed(0)
            x = torch.rand((n, 3, crop, crop), device=self.device, generator=g)
            m = torch.zeros((n, 1, crop, crop), device=self.device)
            m[:, :, :, crop // 3:crop // 3 + crop // 8] = 1
            xx, mm = model.preprocess(x, m)
            with self.autocast_for_probe():
                z = compiled(xx, mm)
                loss = z.float().mean()
            loss.backward()                      # the backward graph compiles too
            model.zero_grad(set_to_none=True)
            print(f"note   : compiled in {_time.time() - t0:.0f}s. Epoch 1 runs at full "
                  f"speed from here;")
            print("note   : compare epoch times against an uncompiled run to see if it "
                  "was worth it.")
            sys.stdout.flush()
            return compiled
        except Exception as e:                                      # noqa: BLE001
            text = f"{type(e).__name__}: {e}"
            hint = ""
            low = text.lower()
            if "python.h" in low or (".lib" in low and "python3" in low):
                hint = (" -- this Python has no C headers. Press 'Enable compile support' "
                        "on the training page, which fetches them for this exact version")
            elif "out of memory" in low:
                hint = (" -- compiling needs memory on top of the model; try a smaller crop "
                        "or leave compile off")
            print(f"note   : torch.compile does not work on this install{hint}.")
            print(f"note   : ({text.splitlines()[0][:200]})")
            print("note   : continuing WITHOUT compile -- training is unaffected")
            sys.stdout.flush()
            try:
                model.zero_grad(set_to_none=True)
                torch._dynamo.reset()
            except Exception:                                       # noqa: BLE001
                pass
            return model

    def autocast_for_probe(self):
        """Same dtype the run will train in, so the probe compiles the same graph."""
        if self.device.type != "cuda":
            return contextlib.nullcontext()
        dtype = (torch.bfloat16 if getattr(self.args, "amp_float", "fp16") == "bfloat16"
                 else torch.float16)
        return torch.autocast(device_type="cuda", dtype=dtype,
                              enabled=not getattr(self.args, "disable_amp", False))

    # ---- compile: keep it out of everything that saves or loads -----------
    #
    # torch.compile returns an OptimizedModule, not a nunif Model. Training is
    # happy with it, but nunif's save_model() asserts isinstance(Model) and its
    # load_model() does a strict load_state_dict -- and a compiled module's keys
    # all carry an "_orig_mod." prefix. Left alone, a compiled run dies at the
    # first "best model updated" and could never be resumed.
    #
    # The wrapper shares its parameters with the module it wraps, so handing the
    # original to those calls is exact, not an approximation.
    @staticmethod
    def _bare(model):
        return getattr(model, "_orig_mod", model)

    @contextlib.contextmanager
    def _plain_model(self):
        """Run a block with self.model as the uncompiled module."""
        compiled = self.model
        self.model = self._bare(compiled)
        try:
            yield
        finally:
            self.model = compiled

    def save_best_model(self):
        with self._plain_model():
            super().save_best_model()

    def create_dataloader(self, type):
        assert type in {"train", "eval"}
        model_offset = self.model.i2i_offset

        if self.args.video:
            # One sample is a whole clip, so the batch axis IS time: batch_size
            # must be 1 and num_samples counts frames, not clips.
            seq = getattr(self.model, "seq_len", VIDEO_SEQ)
            if type == "train":
                check_video_dataset(self.args.data_dir, self.args.crop_size, seq)
            dataset = NTVideoInpaintDataset(
                path.join(self.args.data_dir, type), model_offset,
                model_sequence_offset=getattr(self.model, "sequence_offset", 0),
                training=(type == "train"), size=self.args.crop_size, seq=seq)
            batch_size = 1
            num_samples = max(1, self.args.num_samples // seq)
        else:
            if type == "train":
                check_dataset(self.args.data_dir, self.args.crop_size)
            dataset = NTInpaintDataset(
                path.join(self.args.data_dir, type), model_offset,
                training=(type == "train"), size=self.args.crop_size)
            print(f"{type:6s}: {len(dataset)} pairs")
            batch_size = self.args.batch_size
            num_samples = self.args.num_samples

        # Both loaders are built once, in initialize(), but a DataLoader without
        # this spawns its workers again for every epoch it is iterated. On
        # Windows that is a real process spawn plus a fresh `import torch` in
        # each one, paid 200 times a phase on the train loader and again on
        # every eval pass. Keeping them alive costs only their memory.
        # getattr, not attribute access: New_Trainer's own train.py does not
        # declare these flags and would otherwise crash on this line.
        keep = (self.args.num_workers > 0
                and not getattr(self.args, 'no_persistent_workers', False))

        if type == "train":
            self.sampler = dataset.create_sampler(num_samples)
            return torch.utils.data.DataLoader(
                dataset, sampler=self.sampler, batch_size=batch_size,
                shuffle=False, pin_memory=True, num_workers=self.args.num_workers,
                prefetch_factor=self.args.prefetch_factor if self.args.num_workers > 0 else None,
                persistent_workers=keep,
                drop_last=True)
        # A fixed, evenly spaced subset. Eval must measure the SAME material
        # every time or the curve moves for reasons that have nothing to do with
        # the model; evenly spaced (rather than random) also keeps it
        # representative of the whole eval set, and the sorted clip order means
        # the same indices mean the same clips on every run.
        want = getattr(self.args, "eval_samples", 0) or 0
        if want and want < len(dataset):
            idx = torch.linspace(0, len(dataset) - 1, want).round().long().tolist()
            dataset = torch.utils.data.Subset(dataset, sorted(set(idx)))
            print(f"eval  : {len(dataset)} of the available samples "
                  f"(--eval-samples {want})")
        elif want:
            print(f"eval  : all {len(dataset)} samples (fewer than --eval-samples {want})")

        if len(dataset) < batch_size:
            raise RuntimeError(
                f"eval set has {len(dataset)} "
                f"{'clips' if self.args.video else 'pairs'} but the eval batch size is "
                f"{batch_size}; the eval loader drops the last partial batch, so it would "
                f"be empty. Lower --batch-size or add eval data.")
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
            num_workers=self.args.num_workers,
            prefetch_factor=self.args.prefetch_factor if self.args.num_workers > 0 else None,
            persistent_workers=keep,
            drop_last=True)

    def save_epoch_model(self):
        """`--save-epoch-step N` -- keep a numbered snapshot every N epochs.

        nunif's own flag is all-or-nothing, which on a 600 epoch run means 600
        copies of a 85 MB model. The best model is written separately and is
        never affected by this.
        """
        step = getattr(self.args, "save_epoch_step", 0) or 0
        if step and (self.epoch % step) != 0:
            return
        super().save_epoch_model()

    def load_initial_parameters(self, checkpoint_filename):
        """--checkpoint-file: allow seeding the video model from a trained image
        model. nunif's version calls load_model(strict=True), which refuses the
        image state_dict outright (the temporal blocks have no counterpart)."""
        with self._plain_model():
            if not self.args.video:
                return super().load_initial_parameters(checkpoint_filename)
            data = torch.load(checkpoint_filename, map_location="cpu", weights_only=False)
            name = data.get("name", "")
            if name == self.model.name:
                return super().load_initial_parameters(checkpoint_filename)
            print(f"init   : seeding {self.model.name} from {name}")
            transfer_image_weights(self.model, data["state_dict"])
            return data

    def create_criterion(self):  # noqa: C901
        """The temporal losses are assembled here rather than inherited, to use
        the corrected TemporalSmoothingPenalty. Weights are nunif's."""
        if not self.args.loss.startswith("temporal_"):
            return None
        from nunif.modules.weighted_loss import WeightedLoss
        from nunif.modules.clamp_loss import ClampLoss
        from nunif.modules.dct_loss import DCTLoss
        from nunif.modules.lpips import LPIPSWith
        from iw3.training.inpaint.trainer import TemporalGradientLoss

        if self.args.loss == "temporal_l1lpips":
            base = WeightedLoss(
                (ClampLoss(torch.nn.L1Loss()), TemporalGradientLoss(), TemporalSmoothingPenalty()),
                weights=(0.8, 0.2, 0.01))
        elif self.args.loss == "temporal_l1dctlpips":
            base = WeightedLoss(
                (ClampLoss(torch.nn.L1Loss()),
                 DCTLoss(window_size=32, clamp=True, overlap=True),
                 TemporalGradientLoss(), TemporalSmoothingPenalty()),
                weights=(0.4, 0.4, 0.2, 0.01))
        else:
            return None
        frames = int(getattr(self.args, "lpips_frames", 0) or 0)
        if self.args.video and frames > 0:
            print(f"loss   : LPIPS on {frames} of {VIDEO_SEQ} frames per step")
            return SampledLPIPSWith(base, weight=0.2, frames=frames)
        return LPIPSWith(base, weight=0.2)

    def create_env(self):
        criterion = self.create_criterion()
        if criterion is None:
            criterion = super().create_env().criterion
        env = NTInpaintEnv(self.model, criterion=criterion, sampler=self.sampler,
                           discriminator=self.discriminator)
        if self.discriminator is not None and getattr(self.args, "compile_model", False):
            # Only the env's reference is swapped for the compiled critic.
            # self.discriminator stays the plain Model -- it is what gets saved,
            # loaded and handed to the optimizer, and upstream's InpaintEnv
            # starts with `if discriminator:`, which a compiled module answers
            # with "does not support len()". Both share the same parameters, so
            # the optimizer still trains what the env runs.
            env.discriminator = self._try_compile_disc(self.discriminator)
        return env

    def _try_compile_disc(self, disc):
        """Compile the critic too. nunif only ever compiles the generator.

        On a GAN step the critic runs three forwards and its own backward, and
        the generator's adversarial loss backpropagates through it as well --
        all at full clip resolution. Against a *compiled* generator that is the
        larger half of the step: measured here, 8 it/s with compile and no
        critic became 3.2 it/s with one.

        Probed the same way as the generator -- real shapes, real backward -- and
        in both forms the training loop uses: mask=None (the generator's path,
        where the gradient flows back into the image) and mask given (the
        critic's own path, which returns a second tensor). Anything not probed
        here compiles silently inside epoch 1 instead.

        A failure costs the critic's compile only; the generator keeps its.
        """
        crop = int(getattr(self.args, "crop_size", 256) or 256)
        off = int(getattr(self.model, "i2i_offset", 0) or 0)
        side = crop - off * 2
        seq = getattr(self.model, "seq_len", 0) or 0
        n = seq if (getattr(self.args, "video", False) and seq) else \
            int(getattr(self.args, "batch_size", 1) or 1)
        print(f"note   : compiling the critic for {n}x{side}x{side}. Another few minutes,")
        print("note   : also once. Training has not stalled.")
        sys.stdout.flush()
        t0 = time.time()
        try:
            compiled = torch.compile(disc)
            g = torch.Generator(device=self.device).manual_seed(0)
            fake = torch.rand((n, 3, side, side), device=self.device, generator=g,
                              requires_grad=True)
            cond = torch.rand((n, 3, side, side), device=self.device, generator=g)
            # The mask goes in at crop size, not output size -- the critic fits it
            # to the image itself. Probing the wrong size would compile a graph
            # the run never uses.
            mask = torch.zeros((n, 1, crop, crop), device=self.device)
            mask[:, :, :, crop // 3:crop // 3 + crop // 8] = 1
            with self.autocast_for_probe():
                disc.requires_grad_(False)
                out = compiled(fake, cond, mask=None)
                out.float().mean().backward()
                disc.requires_grad_(True)
                z, _ = compiled(fake.detach(), cond, mask=mask)
                z.float().mean().backward()
            disc.zero_grad(set_to_none=True)
            print(f"note   : critic compiled in {time.time() - t0:.0f}s")
            sys.stdout.flush()
            return compiled
        except Exception as e:                                      # noqa: BLE001
            print(f"note   : the critic would not compile "
                  f"({type(e).__name__}: {str(e).splitlines()[0][:160]})")
            print("note   : running it uncompiled -- the generator keeps its compile")
            sys.stdout.flush()
            try:
                # No torch._dynamo.reset() here: it would throw away the
                # generator's compiled graphs too, and they would then rebuild
                # silently inside epoch 1.
                disc.requires_grad_(True)
                disc.zero_grad(set_to_none=True)
            except Exception:                                       # noqa: BLE001
                pass
            return disc

    # ---- optimizer -------------------------------------------------------
    # ---- the critic has to survive a phase boundary ----------------------
    # nunif's checkpoint holds the generator, the optimizers, the schedulers and
    # the grad scalers -- and not one word about the discriminator (grep it).
    # So across `--resume --reset-state` phases the critic was thrown away and
    # rebuilt from random weights each time, and a brand new critic judging an
    # already-good generator is noise for its first few hundred steps. That made
    # "critic on every run" strictly worse than it sounds: not one critic
    # learning for 600 epochs, but three or four restarts.
    #
    # Saved beside the checkpoint under a name that publish_model and
    # find_models both already skip (".checkpoint" in it, and it does not start
    # with "inpaint."), so it can never be mistaken for a trained model.
    DISC_FILE = "discriminator.checkpoint.pth"
    SAMPLER_FILE = "sampler.checkpoint.pth"

    def _disc_path(self):
        return path.join(self.args.model_dir, self.DISC_FILE)

    def _sampler_path(self):
        return path.join(self.args.model_dir, self.SAMPLER_FILE)

    # ---- the hard-example sampler has to survive a resume too --------------
    # HardExampleSampler starts every sample at loss_sma = inf and weight 1, and
    # nunif saves none of it. So the first epochs after any resume sample the
    # dataset UNIFORMLY, while the epochs before it were oversampling the
    # hardest 10% by up to 4x. The mean training loss over a uniform sample is
    # lower than over a hard-biased one, so the train curve drops for a few
    # epochs and then climbs back as the weights rebuild -- visible on every
    # resume, and eval never moves because eval has no sampler. It also means a
    # few epochs of mining thrown away each time.
    #
    # loss_sma is the whole state that matters; the weights are derived from it
    # by update_weights().
    def _save_sampler(self):
        sampler = getattr(self, "sampler", None)
        sma = getattr(sampler, "loss_sma", None)
        if sma is None:
            return
        try:
            torch.save({"loss_sma": sma.cpu(), "size": int(sma.numel())},
                       self._sampler_path())
        except Exception as e:                                      # noqa: BLE001
            print(f"warning: could not save the sampler state: {e}")

    def _load_sampler(self):
        sampler = getattr(self, "sampler", None)
        sma = getattr(sampler, "loss_sma", None)
        if sma is None:
            return
        fp = self._sampler_path()
        if not path.isfile(fp):
            return
        try:
            blob = torch.load(fp, map_location="cpu", weights_only=False)
            saved = blob["loss_sma"]
            if saved.numel() != sma.numel():
                print(f"note   : saved sampler state is for {saved.numel()} samples, "
                      f"this dataset has {sma.numel()}; starting it fresh")
                return
            sampler.loss_sma = saved.to(sma.dtype)
            sampler.update_weights()
            known = int((sampler.loss_sma < float("inf")).sum())
            print(f"note   : carried the hard-example weights over "
                  f"({known} of {sampler.loss_sma.numel()} samples already scored)")
        except Exception as e:                                      # noqa: BLE001
            print(f"note   : could not reuse the sampler state ({e}); starting it fresh")

    def save_checkpoint(self, **kwargs):
        with self._plain_model():
            super().save_checkpoint(**kwargs)
        if not getattr(self.args, "disable_hard_example", False):
            self._save_sampler()
        # getattr: setup_model() sets this, but a save that somehow runs
        # before it should not take the run down with an AttributeError.
        if getattr(self, "discriminator", None) is None:
            return
        try:
            # _bare: with --compile the critic is an OptimizedModule, whose
            # state_dict keys all carry an "_orig_mod." prefix. Saving those
            # would make the file unreadable by an uncompiled run.
            torch.save({"name": getattr(self.discriminator, "name", ""),
                        "state_dict": self._bare(self.discriminator).state_dict()},
                       self._disc_path())
        except Exception as e:                                      # noqa: BLE001
            print(f"warning: could not save the critic: {e}")

    def resume(self):
        with self._plain_model():
            super().resume()
        if not getattr(self.args, "disable_hard_example", False):
            self._load_sampler()
        if getattr(self, "discriminator", None) is None:
            return
        fp = self._disc_path()
        if not path.isfile(fp):
            print("note   : no saved critic found, starting it from scratch")
            return
        try:
            blob = torch.load(fp, map_location=self.device, weights_only=False)
            want = getattr(self.discriminator, "name", "")
            if blob.get("name") and want and blob["name"] != want:
                print(f"note   : saved critic is {blob['name']}, this run uses {want}; "
                      f"starting it from scratch")
                return
            self._bare(self.discriminator).load_state_dict(blob["state_dict"])
            print("note   : carried the critic over from the previous run")
        except Exception as e:                                      # noqa: BLE001
            print(f"note   : could not reuse the saved critic ({e}); "
                  f"starting it from scratch")

    def create_optimizers(self):
        """Let the discriminator have its own optimizer and learning rate.

        Upstream builds it with `self.create_optimizer(self.discriminator)`, so
        it inherits whatever the generator uses -- here that is CAME with a
        1000-step warmup, which is tuned for the generator. A critic that spends
        its first thousand steps at a fraction of its learning rate is a weak
        critic, and a weak critic early is exactly when a GAN drifts. AdamW is
        the standard choice for the discriminator, so it is the default; pass
        --discriminator-optimizer came to go back to sharing.
        """
        if self.discriminator is None:
            return super().create_optimizers()
        g_opt = self.create_optimizer(self.model)
        d_opt = self.create_optimizer(
            # _bare: the weight-decay grouping walks named_modules(), and the
            # compiled wrapper renames every one of them. Same tensors either
            # way, so the optimizer still drives the compiled critic.
            self._bare(self.discriminator),
            optimizer_type=getattr(self.args, "discriminator_optimizer", None) or "adamw",
            lr=getattr(self.args, "discriminator_lr", None) or self.args.learning_rate)
        return g_opt, d_opt

    def create_optimizer(self, model, optimizer_type=None, lr=None, weight_decay=None, adam_beta1=None):
        optimizer_type = optimizer_type or self.args.optimizer
        if optimizer_type != "came":
            return super().create_optimizer(model, optimizer_type, lr, weight_decay, adam_beta1)

        lr = lr if lr is not None else self.args.learning_rate
        weight_decay = weight_decay if weight_decay is not None else self.args.weight_decay
        beta1 = adam_beta1 if adam_beta1 is not None else self.args.adam_beta1
        betas = (beta1, self.args.came_beta2, self.args.came_beta3)
        groups = configure_optim_groups(model, weight_decay=weight_decay)
        warmup = getattr(self.args, "came_warmup_steps", 0)
        print(f"optim  : CAME lr={lr:g} betas={betas} wd={weight_decay:g} "
              f"clip={self.args.came_clip_threshold:g} warmup={warmup} steps "
              f"({len(groups[0]['params'])} decayed / {len(groups[1]['params'])} not)")
        return CAME(groups, lr=lr, betas=betas,
                    eps=(self.args.came_eps1, self.args.came_eps2),
                    clip_threshold=self.args.came_clip_threshold,
                    weight_decay=weight_decay, warmup_steps=warmup)
