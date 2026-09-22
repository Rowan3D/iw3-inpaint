r"""
The GUI's web server. Standard library only, so it runs on any Python 3.8+ --
including one that has never heard of torch. All the heavy work happens in
subprocesses driven by nunif's own interpreter, which is the one the user points
us at once and we remember.

  launch_gui.bat            ->  http://localhost:8090

Routes
  GET  /                      the page
  GET  /static/<file>         css / js
  GET  /api/state             config + derived paths + every job's status
  POST /api/config            merge a partial config and save it
  POST /api/detect            guess where nunif is
  POST /api/browse            list subfolders, for the folder pickers
  POST /api/inspect           count usable media files in a folder
  POST /api/clean             delete 0-byte / truncated media files
  POST /api/dataset           check a prepared dataset folder
  POST /api/runs              training runs that have a progress.csv
  POST /api/progress          one run's loss rows and preview list
  POST /api/models            trained models, plus iw3 install status
  POST /api/inspect_model     identify one .pth
  POST /api/install           add a model to iw3 and install the hook
  GET  /api/preview/<run>/<f> an eval preview image
  POST /api/job/<slot>/start  begin a job
  POST /api/job/<slot>/stop   end it
"""
from __future__ import annotations

import csv
import json
import re
import os
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from os import path
from urllib.parse import unquote

_APP = path.dirname(path.abspath(__file__))
_GUI = path.dirname(_APP)
for _p in (_APP, _GUI):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as cfgmod                                            # noqa: E402
import jobs as jobsmod                                             # noqa: E402


STATIC = path.join(_APP, "static")
MANAGER = jobsmod.JobManager()

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# One expression covers all three downloaders: they all end their status line
# with "<done>/<total> ... <size> ". Parsing what the tools already print keeps
# the bundled copies byte-identical to New_Trainer's.
DOWNLOAD_RE = r"(?P<done>\d+)/(?P<total>\d+)(?:\s+saved)?,\s*(?P<extra>[\d.]+\s*[KMGTP]?B)"

# The closing summary: "50 videos, 1.8 GB on disk, ..." / "20000 images, ...".
FINAL_RE = r"^\s*(?P<done>\d+)\s+(?:videos|images),\s*(?P<extra>[\d.]+\s*[KMGTP]?B)\s+on disk"

# What the tools need on top of a working nunif install. Checked against every
# third-party import in tools\ and ntrainer\:
#
#   huggingface_hub, pyarrow  the image downloaders (pd12m / 4klsdb)
#   requests                  the video downloaders
#   lpips                     the perceptual term in the training loss
#   dctorch                   DCTLoss, used by the *dct* loss variants
#   tqdm, schedulefree        nunif's training loop and its optimizer choices
#
# `av` is deliberately NOT here even though ntrainer/video_io.py imports it:
# nunif's own requirements.txt PINS it (av==17.1.0), and this list is installed
# with --upgrade, which would drag it past that pin. It arrives with nunif.
# tqdm and requests are in nunif's requirements too but unpinned, so upgrading
# them is safe.
PIP_PACKAGES = ["huggingface_hub", "pyarrow", "lpips", "dctorch", "tqdm",
                "schedulefree", "requests"]


def _flag(argv, name, value, default=None):
    """Append `--name value`, skipping it when it matches the tool's default."""
    if value in (None, "", []):
        return
    if default is not None:
        try:                                  # 5 and 5.0 are the same default
            if float(value) == float(default):
                return
        except (TypeError, ValueError):
            if str(value) == str(default):
                return
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    argv += [name, str(value)]


# ---- job builders --------------------------------------------------------

def build_install_deps(p, _opts):
    return dict(
        argv=[p["python_exe"], "-m", "pip", "install", "--upgrade"] + PIP_PACKAGES,
        cwd=p["nunif_dir"], label="Installing dependencies")


def build_compile_support(p, _opts):
    """Fetch the C headers torch.compile needs. Not part of the dependency step:
    it is not a PyPI package, it is only needed for an optional speed setting,
    and it writes into the Python install rather than site-packages."""
    return dict(argv=[p["python_exe"], path.join(p["tools_dir"], "install_compile_support.py")],
                cwd=p["nunif_dir"], label="Enabling compile support")


def build_probe(p, opts):
    argv = [p["python_exe"], path.join(p["tools_dir"], "fetch_videos.py"),
            "--output", p["download_dir"], "--probe"]
    return dict(argv=argv, cwd=p["nunif_dir"], env=_keys_env(opts),
                label="Testing API keys")


def _keys_env(opts):
    env = {}
    # keys go through the environment, never argv, so they never land in the
    # log pane or in another user's process list
    if opts.get("pexels_key"):
        env["PEXELS_API_KEY"] = opts["pexels_key"]
    if opts.get("pixabay_key"):
        env["PIXABAY_API_KEY"] = opts["pixabay_key"]
    return env


def build_fetch_videos(p, opts):
    out = opts.get("download_dir") or p["download_dir"]
    os.makedirs(out, exist_ok=True)
    pre = _sweep_note(out)
    argv = [p["python_exe"], path.join(p["tools_dir"], "fetch_videos.py"),
            "--output", out, "--count", str(int(opts.get("count") or 400))]
    sources = [s for s in ("pexels", "pixabay") if opts.get("src_" + s)]
    if sources and len(sources) < 2:
        argv += ["--source"] + sources
    _flag(argv, "--min-width", opts.get("min_width"), 1920)
    _flag(argv, "--max-width", opts.get("max_width"), 3840)
    _flag(argv, "--min-duration", opts.get("min_duration"), 5.0)
    _flag(argv, "--max-duration", opts.get("max_duration"), 60.0)
    _flag(argv, "--max-disk-gb", opts.get("max_disk_gb"), 0)
    _flag(argv, "--workers", opts.get("workers"), 6)
    _flag(argv, "--mix", opts.get("mix"))
    if opts.get("allow_ai_generated"):
        argv.append("--allow-ai-generated")
    if opts.get("ignore_no_ai_training"):
        argv.append("--ignore-no-ai-training")
    if opts.get("allow_low_quality"):
        argv.append("--allow-low-quality")
    return dict(argv=argv, cwd=p["nunif_dir"], env=_keys_env(opts),
                progress_re=DOWNLOAD_RE, final_re=FINAL_RE,
                total=int(opts.get("count") or 400),
                counter=jobsmod.count_files(out, VIDEO_EXT),
                preamble=pre, label="Downloading videos")


def _sweep_note(out):
    """Clear failed transfers before starting, so a resumed run does not count
    them as already-downloaded and leave them on disk forever."""
    removed, failed = sweep_incomplete(out)
    note = []
    if removed:
        note.append(f"removed {len(removed)} empty or incomplete file(s) "
                    f"left by an earlier run")
    if failed:
        note.append(f"could not remove {len(failed)} file(s) -- in use?")
    return note


def build_fetch_images(p, opts):
    out = opts.get("download_dir") or p["download_dir"]
    os.makedirs(out, exist_ok=True)
    pre = _sweep_note(out)
    argv = [p["python_exe"], path.join(p["tools_dir"], "fetch_images.py"),
            "--output", out, "--source", opts.get("image_source") or "pd12m",
            "--count", str(int(opts.get("count") or 20000))]
    _flag(argv, "--min-width", opts.get("min_width"), 1920)
    _flag(argv, "--max-disk-gb", opts.get("max_disk_gb"), 0)
    _flag(argv, "--workers", opts.get("workers"), 16)
    _flag(argv, "--mix", opts.get("mix"))
    return dict(argv=argv, cwd=p["nunif_dir"], env=None,
                progress_re=DOWNLOAD_RE, final_re=FINAL_RE,
                total=int(opts.get("count") or 20000),
                counter=jobsmod.count_files(out, IMAGE_EXT),
                preamble=pre, label="Downloading images")


# make_dataset.py and make_video_dataset.py both print "  [i/N] ..." per batch
# of source files, which is the only progress either of them reports.
PREP_RE = r"\[(?P<done>\d+)/(?P<total>\d+)\]\s+(?P<extra>\d+ (?:clips|done))"


def _prep_paths(p, opts):
    src = opts.get("prep_src") or p["download_dir"]
    out = opts.get("prep_out") or p["dataset_dir"]
    os.makedirs(out, exist_ok=True)
    return src, out


def build_scan(p, opts):
    src, out = _prep_paths(p, opts)
    video = (opts.get("media") or "video") != "image"
    tool = "make_video_dataset.py" if video else "make_dataset.py"
    argv = [p["python_exe"], path.join(p["tools_dir"], tool),
            "--input", src, "--output", out, "--scan"]
    if video:
        _flag(argv, "--seq", opts.get("prep_seq"), 12)
        _flag(argv, "--fps", opts.get("prep_fps"), 30)
        _flag(argv, "--stride", opts.get("prep_stride"), 1)
        _flag(argv, "--skip", opts.get("prep_skip"), 24)
        _flag(argv, "--clips-per-video", opts.get("prep_clips"), 4)
        _flag(argv, "--min-width", opts.get("prep_min_width"), 1280)
    return dict(argv=argv, cwd=p["nunif_dir"], label="Checking footage")


def build_prep(p, opts):
    src, out = _prep_paths(p, opts)
    video = (opts.get("media") or "video") != "image"
    tool = "make_video_dataset.py" if video else "make_dataset.py"
    argv = [p["python_exe"], path.join(p["tools_dir"], tool),
            "--input", src, "--output", out]

    # shared with the image tool
    _flag(argv, "--size", opts.get("prep_size"), 640)
    _flag(argv, "--eval-ratio", opts.get("prep_eval_ratio"), 0.03 if video else 0.02)
    _flag(argv, "--model-type", opts.get("prep_model_type"), "Any_V3_Mono")
    _flag(argv, "--resolution", opts.get("prep_resolution"), 784)
    _flag(argv, "--gpu", opts.get("prep_gpu"), 0)
    _flag(argv, "--workers", opts.get("prep_workers"))
    _flag(argv, "--min-detail", opts.get("prep_min_detail"), 0)
    _flag(argv, "--mapper-set", opts.get("prep_mapper"), "safe")
    _flag(argv, "--limit", opts.get("prep_limit"), 0)
    lo, hi = opts.get("prep_div_lo"), opts.get("prep_div_hi")
    if lo not in (None, "") and hi not in (None, ""):
        argv += ["--divergence", str(lo), str(hi)]
    lo, hi = opts.get("prep_conv_lo"), opts.get("prep_conv_hi")
    if lo not in (None, "") and hi not in (None, ""):
        argv += ["--convergence", str(lo), str(hi)]
    widths = str(opts.get("prep_frame_width") or "").replace(",", " ").split()
    if widths:
        argv += ["--frame-width"] + widths
    if opts.get("prep_no_mirror"):
        argv.append("--no-mirror")
    if opts.get("prep_no_refine"):
        argv.append("--no-refine")
    if not opts.get("prep_depth_aa", True):
        argv.append("--no-depth-aa")
    if opts.get("prep_overwrite"):
        argv.append("--overwrite")

    if video:
        _flag(argv, "--seq", opts.get("prep_seq"), 12)
        _flag(argv, "--stride", opts.get("prep_stride"), 1)
        _flag(argv, "--skip", opts.get("prep_skip"), 24)
        _flag(argv, "--fps", opts.get("prep_fps"), 30)
        _flag(argv, "--clips-per-video", opts.get("prep_clips"), 4)
        _flag(argv, "--min-width", opts.get("prep_min_width"), 1280)
        _flag(argv, "--min-mask", opts.get("prep_min_mask"), 300)
        _flag(argv, "--png-level", opts.get("prep_png_level"), 1)
        _flag(argv, "--depth-batch", opts.get("prep_depth_batch"), 4)
        _flag(argv, "--mask-batch", opts.get("prep_mask_batch"), 4)
        mode = opts.get("prep_start_mode") or "0"
        if mode == "random":
            argv += ["--start", "random"]
        elif mode == "seconds" and opts.get("prep_start_seconds"):
            argv += ["--start", str(opts["prep_start_seconds"])]
    else:
        _flag(argv, "--num-samples", opts.get("prep_num_samples"), 2)
        _flag(argv, "--rotate-prob", opts.get("prep_rotate_prob"), 0.25)
        _flag(argv, "--min-parallax", opts.get("prep_min_parallax"), 0)

    total = count_media(src, VIDEO_EXT if video else IMAGE_EXT)
    return dict(argv=argv, cwd=p["nunif_dir"], progress_re=PREP_RE,
                total=total or None,
                label="Preparing clips" if video else "Preparing images")


def count_media(folder, exts):
    if not path.isdir(folder):
        return 0
    n = 0
    for dirpath, _, names in os.walk(folder):
        for f in names:
            if path.splitext(f)[1].lower() in exts:
                n += 1
    return n


def last_logged_epoch(model_dir):
    """Progress for a training run comes from progress.csv, which the trainer
    writes one row per epoch and which already counts continuously across
    phases. Parsing tqdm would be guesswork; this is exact."""
    def _read():
        fp = path.join(model_dir, "progress.csv")
        if not path.exists(fp):
            return 0
        try:
            with open(fp, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if (r.get("epoch") or "").strip()]
            return max((int(float(r["epoch"])) for r in rows), default=0)
        except Exception:                                          # noqa: BLE001
            return 0
    return _read


def _progress_rows(model_dir):
    fp = path.join(model_dir, "progress.csv")
    if not path.isfile(fp):
        return []
    try:
        with open(fp, newline="", encoding="utf-8") as f:
            return [r for r in csv.DictReader(f) if (r.get("epoch") or "").strip()]
    except OSError:
        return []


def _int(row, key, default=0):
    try:
        return int(float(row.get(key) or default))
    except (TypeError, ValueError):
        return default


def phase_progress(model_dir):
    """(last epoch logged, highest phase seen, epochs logged inside that phase).

    Phases are nunif's 200-epoch chunks; the trainer numbers them in
    progress.csv. Everything that needs to say "which run is this" reads it from
    here, because the chain's own step number only counts the steps in the
    chain that happens to be running now.
    """
    rows = _progress_rows(model_dir)
    if not rows:
        return 0, 0, 0
    last = max(_int(r, "epoch") for r in rows)
    phase = max(_int(r, "phase", 1) for r in rows)
    in_phase = sum(1 for r in rows if _int(r, "phase", 1) == phase)
    return last, phase, in_phase


def phase_bests(model_dir):
    """{phase: (best eval loss, the epoch it happened)}."""
    best = {}
    for r in _progress_rows(model_dir):
        ev = (r.get("eval_loss") or "").strip()
        if not ev:
            continue
        try:
            v = float(ev)
        except ValueError:
            continue
        ph = _int(r, "phase", 1)
        if ph not in best or v < best[ph][0]:
            best[ph] = (v, _int(r, "epoch"))
    return best


def _model_dir(p, opts):
    name = (opts.get("train_name") or "model").strip()
    safe = "".join(c for c in name if c.isalnum() or c in "-_ .").strip() or "model"
    return path.join(p["models_dir"], safe), safe


# A stop is a click away and each extra phase is hours, so this is only a
# backstop against a chain that somehow never gets stopped.
MAX_EXTRA_PHASES = 90


def build_train(p, opts):
    data = opts.get("train_data") or p["dataset_dir"]
    model_dir, safe = _model_dir(p, opts)
    os.makedirs(model_dir, exist_ok=True)
    video = (opts.get("media") or "video") != "image"

    epochs = int(opts.get("train_epochs") or 200)
    runs = max(1, int(opts.get("train_runs") or 3))
    reset_every = int(opts.get("train_lr_reset") or 40)
    cycles = max(1, round(epochs / reset_every)) if reset_every else 1

    base = [p["python_exe"], path.join(p["tools_dir"], "train.py"), data, model_dir]
    if video:
        base.append("--video")
    _flag(base, "--arch", opts.get("train_arch"), "inpaint.nt_inpaint_v2_b")
    _flag(base, "--crop-size", opts.get("train_crop"), 256)
    _flag(base, "--batch-size", opts.get("train_batch"), 8)
    _flag(base, "--backward-step", opts.get("train_backward"), 4)
    _flag(base, "--num-samples", opts.get("train_num_samples"), 20000)
    _flag(base, "--max-epoch", epochs, 200)
    _flag(base, "--learning-rate-cycles", cycles, 5)
    _flag(base, "--eval-step", opts.get("train_eval_step"), 4)
    _flag(base, "--eval-samples", opts.get("train_eval_samples"), 256)
    _flag(base, "--eval-previews", opts.get("train_previews"), 8)
    _flag(base, "--eval-preview-rows", opts.get("train_preview_rows"), 4)
    _flag(base, "--optimizer", opts.get("train_optimizer"), "came")
    _flag(base, "--learning-rate", opts.get("train_lr"))
    _flag(base, "--num-workers", opts.get("train_workers"))
    _flag(base, "--gpu", opts.get("train_gpu"), 0)
    _flag(base, "--loss", opts.get("train_loss"))
    if opts.get("train_disable_hard"):
        base.append("--disable-hard-example")
    # nunif has had EMA all along -- it evaluates and saves the averaged weights
    # when it is on -- and nothing in this project was using it.
    if opts.get("train_ema", True):
        base.append("--ema-model")
        _flag(base, "--ema-decay", opts.get("train_ema_decay"), 0.999)
    if opts.get("train_channels_last"):
        base.append("--channels-last")
    if opts.get("train_compile"):
        base.append("--compile")
    _flag(base, "--lpips-frames", opts.get("train_lpips_frames"), 4)
    _flag(base, "--amp-float", opts.get("train_amp_float"))

    # The critic is not added to `base`: it is applied per run below, so it can
    # start at a later run than the first.
    critic = []
    if opts.get("train_discriminator"):
        _flag(critic, "--discriminator", opts.get("train_discriminator"))
        _flag(critic, "--discriminator-weight", opts.get("train_disc_weight"), 0.2)
        _flag(critic, "--generator-warmup-iteration", opts.get("train_disc_warmup"), 500)

    keep = opts.get("train_keep") or "best"
    if keep == "all":
        base.append("--save-epoch")
    elif keep == "every":
        base += ["--save-epoch", "--save-epoch-step", str(int(opts.get("train_keep_step") or 20))]

    # nunif trims a cosine run so it ends at the bottom of a cycle:
    #   max_epoch -= (max_epoch % cycles) + 1
    # so a "200 epoch" run really does 199. Counting the nominal number would
    # leave the bar and the ETA permanently short of the end.
    per_run = max(1, epochs - (epochs % cycles) - 1)

    # Where the training actually is, as opposed to where this chain starts.
    # Restarting a chain part-way is normal -- you stop for the night, you press
    # Start again -- and the old code then relabelled phase 2 as "run 1 of 3"
    # and kept the total at 3 x 199, while the chain really had three MORE
    # phases queued behind the one already running. Both numbers now come from
    # progress.csv.
    last_ep, cur_phase, in_phase = phase_progress(model_dir)
    resuming = bool(opts.get("train_resume")) and cur_phase > 0
    # step 0 continues the current phase when resuming; otherwise the trainer
    # starts a new phase on top of whatever is already logged.
    #
    # `runs` is the size of the PLAN -- "3 runs of 200" -- not a number of runs
    # to add. Resuming used to set last_phase = first_phase + runs - 1, so
    # stopping during run 2 of 3 and pressing Start again queued three MORE
    # phases on top of the two already done and moved the target from 597 to
    # 796, then 995 after the next stop. That was not only the label: those
    # phases were really queued and would really have run. Resuming now
    # finishes the plan it is already inside.
    if resuming:
        first_phase = cur_phase
        last_phase = max(runs, cur_phase)      # already past the plan? finish where you are
    else:
        first_phase = cur_phase + 1
        last_phase = cur_phase + runs
    n_steps = max(1, last_phase - first_phase + 1)

    # Which run the critic joins on. A GAN is the one setting here that can lose
    # a run, and the reconstruction loss does most of the work anyway, so the
    # safe order is to bank a good model first and let the critic sharpen it:
    # by the time it starts, the previous run's best is already archived as
    # .phaseN.pth and cannot be overwritten.
    # Clamped to the last run: asking for the critic at run 3 of a 2-run chain
    # otherwise picked a critic, showed the warning, and then quietly trained
    # without one.
    critic_start = min(max(1, int(opts.get("train_disc_start") or 1)),
                       last_phase) if critic else 0

    def phase_step(phase_no, first):
        argv = list(base)
        if critic and phase_no >= critic_start:
            argv += critic
        if not first:
            argv += ["--resume", "--reset-state"]
        elif opts.get("train_resume"):
            argv.append("--resume")
        label = "training"
        if last_phase > 1 or phase_no > 1:
            label = (f"run {phase_no} of {last_phase}" if phase_no <= last_phase
                     else f"run {phase_no} (extra)")
        if critic and phase_no >= critic_start:
            label += " + critic"
        return {"argv": argv, "label": label}

    steps = [phase_step(first_phase + i, i == 0) for i in range(n_steps)]

    final_name = f"{safe}.pth"
    steps.append({"call": lambda: publish_model(model_dir, p["models_dir"], final_name),
                  "label": f"save as {final_name}"})

    if resuming:
        # Each phase ends at the same internal epoch, so N phases is always
        # N x per_run logged epochs however many times it was stopped.
        total = per_run * last_phase
    else:
        total = last_ep + per_run * runs

    # ---- keep going past the plan, while the toggle says so ----------------
    # Read from config.json rather than from `opts` every time, so turning it on
    # (or off) from the live page part way through a run is picked up at the
    # next phase boundary instead of needing a restart.
    extra = [0]

    def extend(job):
        if extra[0] >= MAX_EXTRA_PHASES:
            return
        # only when the last TRAINING step just finished: the save step is last
        # and nothing may run after it.
        if job.step_index != len(job.steps) - 2:
            return
        if not cfgmod.load().get("train_overflow"):
            return
        extra[0] += 1
        job.steps.insert(len(job.steps) - 1,
                         phase_step(last_phase + extra[0], first=False))
        job.total = (job.total or 0) + per_run
        return (f"overflow is on -- adding run {last_phase + extra[0]}; "
                f"turn it off on the live page to stop after this one")

    cfgmod.save({"train_overflow": bool(opts.get("train_overflow"))})

    return dict(steps=steps, cwd=p["nunif_dir"],
                counter=last_logged_epoch(model_dir),
                total=total, extend=extend,
                label=f"Training {safe}")


def build_verify(p, opts):
    """Ask iw3 itself, in a fresh process, whether every installed piece is live
    and whether the named entry loads. This goes through the real registry and
    the real hook, so it is the only check that proves the install worked."""
    argv = [p["python_exe"], path.join(EXTRAS_DIR, "install.py"), "--check",
            "--nunif", p["nunif_dir"], "--python", p["python_exe"]]
    if (opts.get("name") or "").strip():
        argv += ["--name", opts["name"].strip()]
    if opts.get("kind") in ("video", "image"):
        argv += ["--kind", opts["kind"]]
    return dict(argv=argv, cwd=p["nunif_dir"], label="Checking the iw3 install")


def build_bench(p, opts):
    """Measure a real training step at the settings on the page, so the crop and
    batch are chosen from numbers rather than from a guess and an out-of-memory
    crash three hours in."""
    video = (opts.get("media") or "video") != "image"
    argv = [p["python_exe"], path.join(p["tools_dir"], "bench_model.py"), "--train"]
    _flag(argv, "--crop", opts.get("train_crop"), 256)

    if video:
        # A video model mixes across the whole clip, so its batch axis IS time
        # and it asserts on anything that is not a multiple of seq_len. That is
        # also what training really does -- one loader item is one clip -- so
        # the honest benchmark batch is a whole number of clips.
        clips = max(1, int(opts.get("bench_clips") or 1))
        batch = SEQ_LEN * clips
    else:
        batch = max(1, int(opts.get("train_batch") or 8))
    _flag(argv, "--train-batch", batch, 8)
    _flag(argv, "--gpu", opts.get("train_gpu"), 0)
    # Measure what this page is actually configured to run: the real criterion
    # (the perceptual term is the majority of the cost), and the critic if one
    # is chosen, so the extra it costs is a row you can read off.
    if video:
        argv.append("--video")
        _flag(argv, "--lpips-frames", opts.get("train_lpips_frames"), 4)
    _flag(argv, "--amp-float", opts.get("train_amp_float"))
    if opts.get("train_discriminator"):
        _flag(argv, "--discriminator", opts.get("train_discriminator"))

    family = VIDEO_ARCH_MAP if video else {}
    stock = "inpaint.light_video_inpaint_v1" if video else "inpaint.light_inpaint_v1"
    if opts.get("bench_all"):
        picked = ["inpaint.nt_inpaint_v2_s", "inpaint.nt_inpaint_v2_b",
                  "inpaint.nt_inpaint_v2_l"]
    else:
        picked = [opts.get("train_arch") or "inpaint.nt_inpaint_v2_b"]
    argv += ["--arch"] + [family.get(a, a) for a in picked] + [stock]
    return dict(argv=argv, cwd=p["nunif_dir"], label="Measuring speed and memory")


# One clip is 12 frames everywhere in this project; the model asserts on it.
SEQ_LEN = 12

# train.py does this mapping itself for --video; the benchmark takes an arch
# name directly, so it has to be done here too.
VIDEO_ARCH_MAP = {
    "inpaint.nt_inpaint_v2": "inpaint.nt_video_inpaint_v2",
    "inpaint.nt_inpaint_v2_s": "inpaint.nt_video_inpaint_v2_s",
    "inpaint.nt_inpaint_v2_b": "inpaint.nt_video_inpaint_v2_b",
    "inpaint.nt_inpaint_v2_l": "inpaint.nt_video_inpaint_v2_l",
}


PHASE_RE = re.compile(r"\.phase(\d+)\.pth$")


def publish_model(model_dir, models_dir, final_name):
    """Copy the best model out of the run folder under the name the user chose,
    so it is obvious which file to install and the run folder stays intact.

    This used to take the first .pth in sort order that was not a checkpoint or
    a phase archive, which was wrong twice over:

    * any unrelated file dropped in the folder could win -- a pinned copy named
      HQ_Inpaint_e76.pth sorted ahead of inpaint.nt_video_inpaint_v2_b.pth,
      because capitals sort before lowercase, and would have been published as
      the finished model at the end of a three-day run;
    * it ignored the .phaseN.pth archives, so if the last phase ended worse than
      an earlier one -- which happens, every phase is a fresh cosine -- the
      worse model got published.

    Now only files the trainer itself writes are candidates, and progress.csv
    decides between them.
    """
    import shutil
    bests = phase_bests(model_dir)
    live_phase = max(bests) if bests else 1

    try:
        names = os.listdir(model_dir)
    except OSError as e:
        raise RuntimeError(f"cannot read {model_dir}: {e}")

    cands = []                                   # (eval or None, phase, path)
    for f in names:
        if not f.startswith("inpaint.") or not f.endswith(".pth"):
            continue
        if ".checkpoint" in f:
            continue
        m = PHASE_RE.search(f)
        phase = int(m.group(1)) if m else live_phase
        cands.append((bests.get(phase, (None, 0))[0], phase, path.join(model_dir, f)))

    if not cands:
        raise RuntimeError(f"no finished model in {model_dir} -- did the run get far "
                           f"enough to write one?")

    scored = [c for c in cands if c[0] is not None]
    # No eval recorded at all (a run stopped before its first eval): fall back
    # to the newest phase rather than guessing.
    pick = min(scored, key=lambda c: c[0]) if scored else max(cands, key=lambda c: c[1])
    dst = path.join(models_dir, final_name)
    shutil.copy2(pick[2], dst)
    note = f", eval {pick[0]:.6g}" if pick[0] is not None else ""
    return (f"{path.basename(pick[2])} (run {pick[1]}{note}) -> {dst} "
            f"({path.getsize(dst) / 1e6:.0f} MB)")


BUILDERS = {
    "deps": build_install_deps,
    "compile_support": build_compile_support,
    "train": build_train,
    "bench": build_bench,
    "verify": build_verify,
    "scan": build_scan,
    "prep": build_prep,
    "probe": build_probe,
    "fetch_videos": build_fetch_videos,
    "fetch_images": build_fetch_images,
}


# ---- helpers -------------------------------------------------------------

# Any real clip is far bigger than this. Interrupted transfers land at 0 bytes
# or a few KB, and they are indistinguishable from real files by name alone.
MIN_VIDEO_BYTES = 64 * 1024
MIN_IMAGE_BYTES = 2 * 1024


def _incomplete(folder):
    """Media files too small to be real, plus leftover .part files."""
    bad = []
    if not folder or not path.isdir(folder):
        return bad
    try:
        names = os.listdir(folder)
    except OSError:
        return bad
    for name in names:
        fp = path.join(folder, name)
        if not path.isfile(fp):
            continue
        ext = path.splitext(name)[1].lower()
        if ext == ".part":
            bad.append(name)
            continue
        floor = (MIN_VIDEO_BYTES if ext in VIDEO_EXT
                 else MIN_IMAGE_BYTES if ext in IMAGE_EXT else None)
        if floor is None:
            continue
        try:
            if path.getsize(fp) < floor:
                bad.append(name)
        except OSError:
            pass
    return bad


def sweep_incomplete(folder):
    """Delete them. Returns (removed, failed)."""
    removed, failed = [], []
    for name in _incomplete(folder):
        try:
            os.remove(path.join(folder, name))
            removed.append(name)
        except OSError:
            failed.append(name)
    return removed, failed


def check_nunif(raw):
    """Is this a usable nunif install? Answered without saving anything.

    The setup page used to find out only when you pressed Save, which wrote the
    path to config.json first and then told you it was wrong, and left the red
    message sitting there while you typed a correct one. This says what is
    actually missing, as you type.
    """
    out = {"raw": raw or "", "ok": False, "nunif_dir": "", "python_exe": "",
           "message": "", "kind": "bad"}
    raw = (raw or "").strip().strip('"')
    if not raw:
        out["message"] = "Type or pick the folder that has nunif\\ and iw3\\ inside it."
        out["kind"] = ""
        return out
    expanded = path.normpath(path.expanduser(raw))
    if not path.isdir(expanded):
        out["message"] = "There is no folder at that path."
        return out

    resolved = cfgmod.resolve_nunif(raw)
    if not resolved:
        # Say which half is missing -- "not a nunif install" is useless when you
        # are one folder above or below the right one.
        have = []
        for sub in ("nunif", "iw3"):
            if path.isdir(path.join(expanded, sub)):
                have.append(sub)
        if have:
            missing = [s for s in ("nunif", "iw3") if s not in have]
            out["message"] = (f"Found {have[0]}\\ but not {missing[0]}\\ in that folder. "
                              f"This needs the folder that has both.")
        else:
            near = [d for d in ("nunif", "iw3", "New_Trainer")
                    if path.isdir(path.join(expanded, d))]
            out["message"] = ("That folder has neither nunif\\ nor iw3\\ in it."
                              + (f" It does contain {near[0]}\\ — try one level in."
                                 if near else ""))
        return out

    out["nunif_dir"] = resolved
    py = cfgmod.python_for(resolved)
    out["python_exe"] = py
    if not py:
        out["message"] = (f"Found nunif at {resolved}, but not its python folder beside "
                          f"it. Jobs need that interpreter, so they will not run.")
        return out
    out["ok"] = True
    out["kind"] = "ok"
    same = path.normpath(resolved) == path.normpath(expanded)
    out["message"] = (f"Looks right. Using {resolved}"
                      + ("" if same else " (one level in from what you typed)")
                      + f", with {path.basename(path.dirname(py))}\\"
                      f"{path.basename(py)}.")
    return out


def folder_stats(folder):
    """What is actually in a folder, so 'use my own files' can be checked
    before a long job rather than after one."""
    out = {"exists": False, "videos": 0, "images": 0, "bytes": 0, "path": folder,
           "empty": 0, "nested": 0, "folders": 0}
    if not folder or not path.isdir(folder):
        return out
    out["exists"] = True
    # Walk, don't list. Both dataset tools find their input with os.walk
    # (list_videos / list_images), so a folder of your own clips dropped inside
    # the downloads folder IS used -- and counting only the top level here said
    # otherwise, which is the opposite of reassuring when you have just put one
    # there.
    try:
        for dirpath, dirnames, names in os.walk(folder):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            nested = path.abspath(dirpath) != path.abspath(folder)
            if nested and any(path.splitext(n)[1].lower() in VIDEO_EXT | IMAGE_EXT
                              for n in names):
                out["folders"] += 1
            for name in names:
                fp = path.join(dirpath, name)
                if not path.isfile(fp):
                    continue
                ext = path.splitext(name)[1].lower()
                if ext in VIDEO_EXT:
                    out["videos"] += 1
                elif ext in IMAGE_EXT:
                    out["images"] += 1
                else:
                    continue
                if nested:
                    out["nested"] += 1
                try:
                    out["bytes"] += path.getsize(fp)
                except OSError:
                    pass
    except OSError as e:
        out["error"] = str(e)
    # The counts above include the duds, since they are ordinary .mp4/.jpg
    # files as far as the filesystem is concerned. Take them back out per kind
    # so "12 videos" means twelve videos you can actually train on.
    # _incomplete stays top-level only, deliberately: it exists to clear the
    # downloader's own failed transfers, and a short clip of your own is a
    # perfectly good file that happens to be small.
    bad = _incomplete(folder)
    out["empty"] = len(bad)
    for name in bad:
        ext = path.splitext(name)[1].lower()
        if ext in VIDEO_EXT:
            out["videos"] = max(0, out["videos"] - 1)
        elif ext in IMAGE_EXT:
            out["images"] = max(0, out["images"] - 1)
    return out


EVAL_RE_S = r"^epoch(\d+)_(\d+)\.png$"
EVAL_RE = re.compile(EVAL_RE_S)

PROGRESS_KEYS = ("epoch", "phase", "train_loss", "eval_loss", "lr", "weight_decay",
                 "seconds", "timestamp")


def list_runs(models_dir):
    """Every folder under the models directory that a training run has written
    a progress.csv into, newest activity first."""
    out = []
    if not path.isdir(models_dir):
        return out
    for name in os.listdir(models_dir):
        d = path.join(models_dir, name)
        fp = path.join(d, "progress.csv")
        if not path.isfile(fp):
            continue
        try:
            mtime = path.getmtime(fp)
        except OSError:
            mtime = 0
        out.append({"name": name, "path": d, "mtime": mtime})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def _num(v):
    v = (v or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def run_progress(model_dir):
    """progress.csv plus the eval previews on disk. Read fresh every poll: the
    trainer rewrites the file whole, so a half-written read just loses one tick
    rather than corrupting anything."""
    out = {"path": model_dir, "rows": [], "previews": {}, "preview_epochs": [],
           "slots": [], "exists": path.isdir(model_dir)}
    fp = path.join(model_dir, "progress.csv")
    if path.isfile(fp):
        try:
            with open(fp, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if not (r.get("epoch") or "").strip():
                        continue
                    out["rows"].append({
                        "epoch": int(float(r["epoch"])),
                        "phase": int(float(r.get("phase") or 1)),
                        "train": _num(r.get("train_loss")),
                        "eval": _num(r.get("eval_loss")),
                        "lr": _num(r.get("lr")),
                        "seconds": _num(r.get("seconds")),
                        "timestamp": (r.get("timestamp") or "").strip(),
                    })
        except Exception as e:                                     # noqa: BLE001
            out["error"] = f"progress.csv unreadable ({e})"
        out["rows"].sort(key=lambda r: r["epoch"])

    d = path.join(model_dir, "eval")
    if path.isdir(d):
        slots = set()
        by_epoch = {}
        try:
            for name in os.listdir(d):
                m = EVAL_RE.match(name)
                if not m:
                    continue
                ep, slot = int(m.group(1)), int(m.group(2))
                by_epoch.setdefault(ep, {})[slot] = name
                slots.add(slot)
        except OSError:
            pass
        out["previews"] = {str(e): {str(k): v for k, v in files.items()}
                           for e, files in by_epoch.items()}
        out["preview_epochs"] = sorted(by_epoch)
        out["slots"] = sorted(slots)
    return out


# ---- installing a model into iw3 ----------------------------------------

def _pickle_strings(raw):
    """Every BINUNICODE / SHORT_BINUNICODE string in a pickle, read by its own
    length prefix.

    A .pth is a zip with a pickle inside, and the architecture name is a plain
    string in it. Reading it this way means the GUI can identify a model in
    milliseconds without importing torch -- which would otherwise cost about
    eight seconds and a gigabyte of RAM just to show a filename in a list.
    """
    import struct
    out, i, n = [], 0, len(raw)
    while i < n:
        op = raw[i]
        if op == 0x58 and i + 5 <= n:                    # BINUNICODE
            ln = struct.unpack_from("<I", raw, i + 1)[0]
            if 0 < ln < 4096 and i + 5 + ln <= n:
                try:
                    out.append(raw[i + 5:i + 5 + ln].decode("utf-8"))
                except UnicodeDecodeError:
                    pass
                i += 5 + ln
                continue
        elif op == 0x8C and i + 2 <= n:                   # SHORT_BINUNICODE
            ln = raw[i + 1]
            if i + 2 + ln <= n:
                try:
                    out.append(raw[i + 2:i + 2 + ln].decode("utf-8"))
                except UnicodeDecodeError:
                    pass
                i += 2 + ln
                continue
        i += 1
    return out


def inspect_model(fp):
    out = {"path": fp, "exists": False, "arch": "", "kind": "", "bytes": 0,
           "saved": "", "nunif": False}
    if not fp or not path.isfile(fp):
        return out
    out["exists"] = True
    try:
        out["bytes"] = path.getsize(fp)
    except OSError:
        pass
    try:
        import zipfile
        with zipfile.ZipFile(fp) as z:
            pkl = next((n for n in z.namelist() if n.endswith("data.pkl")), None)
            if pkl:
                strings = _pickle_strings(z.read(pkl))
                out["nunif"] = "nunif_model" in strings
                for t in strings:
                    if t.startswith("inpaint.") and not out["arch"]:
                        out["arch"] = t
                    if (not out["saved"] and len(t) > 18
                            and t[4:5] == "-" and t[7:8] == "-"):
                        out["saved"] = t[:19]
    except Exception as e:                                         # noqa: BLE001
        out["error"] = str(e)
    if out["arch"]:
        out["kind"] = "video" if "video" in out["arch"] else "image"
    return out


def find_models(models_dir):
    """Every finished model the GUI knows about: the published copies sitting in
    the models folder, and the best model inside each run folder. Checkpoints
    are deliberately left out -- they carry the optimizer state and are for
    resuming, not for installing."""
    found = []
    if not path.isdir(models_dir):
        return found

    def add(fp, label, group):
        info = inspect_model(fp)
        if not info["exists"] or not info["nunif"]:
            return
        info.update(label=label, group=group)
        found.append(info)

    try:
        names = sorted(os.listdir(models_dir), key=str.lower)
    except OSError:
        return found
    for name in names:
        fp = path.join(models_dir, name)
        if path.isfile(fp) and name.lower().endswith(".pth"):
            add(fp, path.splitext(name)[0], "finished")
    for name in names:
        d = path.join(models_dir, name)
        if not path.isdir(d):
            continue
        # "installed" holds frozen copies of models already listed above, made
        # by the Install step. Showing them again would just be the same
        # weights under a second name.
        if name.lower() == "installed":
            continue
        try:
            inner = sorted(os.listdir(d), key=str.lower)
        except OSError:
            continue
        bests = phase_bests(d)
        live_phase = max(bests) if bests else 1
        for f in inner:
            if not f.lower().endswith(".pth") or ".checkpoint" in f:
                continue
            if f.endswith(".pth.bk"):
                continue
            if not f.startswith("inpaint."):
                # Not something the trainer wrote. Calling it "best so far"
                # would be a guess, and a wrong one -- name it plainly.
                add(path.join(d, f), f"{name} \u2014 {path.splitext(f)[0]}", name)
                continue
            m = PHASE_RE.search(f)
            phase = int(m.group(1)) if m else live_phase
            tag = f"run {phase}" if m else f"run {phase}, still training"
            score = bests.get(phase, (None, 0))
            if score[0] is not None:
                tag += f" \u2014 best eval {score[0]:.4g} at epoch {score[1]}"
            add(path.join(d, f), f"{name} \u2014 {tag}", name)
    return found


def yml_path(nunif_dir):
    return path.join(nunif_dir, "iw3", "inpaint_models.yml")


def read_yml_names(nunif_dir):
    """Top-level keys of inpaint_models.yml, without needing pyyaml here."""
    fp = yml_path(nunif_dir)
    names = []
    if not path.isfile(fp):
        return names
    try:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^([A-Za-z0-9_.\-]+):\s*$", line)
                if m:
                    names.append(m.group(1))
    except OSError:
        pass
    return names


# The trainer installs into iw3 with the SAME installer that is shipped to other
# people (iw3_extras\install.py, a copy of Installer\installer\), so what you
# test from here is exactly what they get. It writes <nunif>\nt_inpaint\ plus a
# site-packages hook, and this GUI keeps managing the model entries itself --
# several side by side, one per install name, each a frozen copy.
EXTRAS_DIR = path.join(path.dirname(path.dirname(path.abspath(__file__))), "iw3_extras")
EXTRAS_FOLDER = "nt_inpaint"
HOOK_FILES = ("nt_iw3.pth", "nt_iw3_register.py")
# what this GUI used to write; the installer removes it (it would register the
# older architectures over the new ones and apply the mask patch twice)
LEGACY_HOOK_FILES = ("ntrainer.pth", "ntrainer_iw3_register.py")


def hook_target(python_exe):
    """The site-packages of the interpreter iw3 runs with.

    getsitepackages() lists candidates that need not all exist -- on Debian the
    last entry is a directory that was never created -- so take the last one
    that is actually there, preferring a real "site-packages" over the prefix
    itself.
    """
    import subprocess
    try:
        out = subprocess.run(
            [python_exe, "-c", "import site;print('\\n'.join(site.getsitepackages()))"],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt" else 0)
    except Exception:                                              # noqa: BLE001
        return ""
    cands = [d.strip() for d in (out.stdout or "").splitlines() if d.strip()]
    cands = [d for d in cands if path.isdir(d)]
    if not cands:
        return ""
    named = [d for d in cands if path.basename(d).lower() in
             ("site-packages", "dist-packages")]
    return (named or cands)[-1]


def _tree_digest(folder):
    """One hash over every .py under folder, to tell whether what is installed
    in iw3 is the code this GUI ships or an older copy."""
    import hashlib
    h = hashlib.md5()
    if not path.isdir(folder):
        return ""
    for dp, dns, fns in os.walk(folder):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for fn in sorted(f for f in fns if f.endswith(".py")):
            fp = path.join(dp, fn)
            h.update(path.relpath(fp, folder).replace("\\", "/").encode())
            try:
                with open(fp, "rb") as f:
                    h.update(f.read())
            except OSError:
                pass
    return h.hexdigest()


def site_dirs(python_exe):
    """Every existing site-packages / dist-packages directory of that Python."""
    import subprocess
    try:
        r = subprocess.run(
            [python_exe, "-c", "import site;print('\\n'.join(site.getsitepackages()))"],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt" else 0)
    except Exception:                                              # noqa: BLE001
        return []
    return [d.strip() for d in (r.stdout or "").splitlines()
            if d.strip() and path.isdir(d.strip())]


def hook_status(p):
    """What is installed in iw3 right now, from the files themselves.

    Every candidate packages folder is looked in, rather than guessing which one
    the installer picked -- the two rules agree on Windows' embedded Python but
    not everywhere, and a status that says "not installed" about a working
    install is worse than none."""
    out = {"dir": "", "installed": False, "legacy": False, "current": False,
           "features": {}, "folder": ""}
    dirs = site_dirs(p["python_exe"]) if p["python_exe"] else []
    for d in dirs:
        if all(path.isfile(path.join(d, f)) for f in HOOK_FILES):
            out["installed"], out["dir"] = True, d
        if any(path.isfile(path.join(d, f)) for f in LEGACY_HOOK_FILES):
            out["legacy"] = True
    if not out["dir"] and dirs:
        out["dir"] = hook_target(p["python_exe"])
    if p["nunif_dir"]:
        root = path.join(p["nunif_dir"], EXTRAS_FOLDER)
        out["folder"] = root
        try:
            with open(path.join(root, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            out["features"] = {k: bool(cfg.get(k)) for k in
                               ("model", "mask_patch", "window", "lowres", "border")}
            out["installed_at"] = cfg.get("installed", "")
        except (OSError, ValueError):
            pass
        out["current"] = bool(out["features"]) and (
            _tree_digest(path.join(root, "nt_iw3"))
            == _tree_digest(path.join(EXTRAS_DIR, "payload", "nt_iw3")))
    return out


def run_extras(p, args, timeout=900):
    """Run iw3_extras\\install.py with nunif's own Python. -> (returncode, lines)."""
    import subprocess
    script = path.join(EXTRAS_DIR, "install.py")
    if not path.isfile(script):
        raise RuntimeError(f"the installer is missing from {EXTRAS_DIR} -- copy the "
                           f"iw3_extras folder over with the rest of the GUI")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([p["python_exe"], script, "--nunif", p["nunif_dir"],
                        "--python", p["python_exe"]] + list(args),
                       cwd=p["nunif_dir"], stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, env=env,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                       if os.name == "nt" else 0)
    return r.returncode, ((r.stdout or "") + (r.stderr or "")).splitlines()


def install_extras(p, name, kind, lowres, border):
    """The model is always on here (that is what the Install button is for);
    the other two are the same choices the standalone installer asks about."""
    rc, lines = run_extras(p, ["--only", "both" if lowres else "model",
                               "--border", "yes" if border else "no",
                               "--external-model", "--name", name, "--kind", kind])
    checks = [ln.strip() for ln in lines if ln.strip().startswith(("[ok]", "[!!]"))]
    notes = [ln.strip() for ln in lines if "old hook" in ln or "replaces" in ln]
    if rc not in (0, 2) or not checks:
        tail = "\n".join(lines[-15:])
        raise RuntimeError(f"the iw3 installer stopped (code {rc}):\n{tail}")
    return rc == 0, checks, notes


def pin_model(models_dir, model_path, name):
    """Freeze the chosen weights under `installed\\<name>.pth` and return that.

    Installing used to write the run folder's live path straight into the yml,
    and that file is not a snapshot: the trainer rewrites it every time eval
    improves, and `--reset-state` zeroes best_loss so the FIRST eval of the next
    phase overwrites it even though it is worse. An entry installed as "epoch
    199" therefore turned into a worse model within the hour, silently, with
    iw3 still showing the name the user chose.

    Copying costs ~85 MB per install and makes the entry mean what it says.
    """
    import shutil
    if path.dirname(path.abspath(model_path)) == path.abspath(path.join(models_dir,
                                                                       "installed")):
        return model_path                                  # already pinned
    dst_dir = path.join(models_dir, "installed")
    os.makedirs(dst_dir, exist_ok=True)
    dst = path.join(dst_dir, f"{name}.pth")
    if path.abspath(dst) == path.abspath(model_path):
        return dst
    shutil.copy2(model_path, dst)
    return dst


def install_model(p, model_path, name, kind):
    """Add or replace one entry in iw3's inpaint_models.yml.

    The file is rewritten rather than parsed and re-serialised: it is a config a
    person edits by hand, and round-tripping it through a YAML library would
    reformat their comments and reorder their entries.
    """
    if not path.isfile(model_path):
        raise RuntimeError("that model file does not exist")
    name = (name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,64}", name):
        raise RuntimeError("the name may only use letters, numbers, dot, dash "
                           "and underscore, and cannot be empty")
    if kind not in ("video", "image"):
        raise RuntimeError("the model must be a video or an image model")

    model_path = pin_model(p["models_dir"], model_path, name)

    fp = yml_path(p["nunif_dir"])
    os.makedirs(path.dirname(fp), exist_ok=True)
    old = ""
    if path.isfile(fp):
        with open(fp, encoding="utf-8") as f:
            old = f.read()
        backup = fp + ".backup"
        if not path.exists(backup):
            with open(backup, "w", encoding="utf-8") as f:
                f.write(old)

    # drop any previous block for this name, keeping everything else verbatim
    lines = old.splitlines()
    out, skipping = [], False
    for line in lines:
        if re.match(rf"^{re.escape(name)}:\s*$", line):
            skipping = True
            continue
        if skipping:
            if line.strip() == "" or line.startswith((" ", "\t")):
                continue
            skipping = False
        out.append(line)
    while out and not out[-1].strip():
        out.pop()

    block = [f"{name}:", f"  {kind}: {model_path}"]
    text = "\n".join(out + ([""] if out else []) + block) + "\n"
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, fp)
    return fp, model_path


def dataset_stats(folder):
    """A dataset is train/ and eval/ full of either clip folders (video) or
    _C.png/_M.png pairs (images). Saying which, and how many, catches the common
    mistake of pointing step 3 at the raw footage."""
    out = {"exists": False, "train": 0, "eval": 0, "unit": "clips", "path": folder}
    if not folder or not path.isdir(folder):
        return out
    out["exists"] = True
    for split in ("train", "eval"):
        d = path.join(folder, split)
        if not path.isdir(d):
            continue
        try:
            names = os.listdir(d)
        except OSError:
            continue
        dirs = sum(1 for n in names if path.isdir(path.join(d, n)))
        pairs = sum(1 for n in names if n.lower().endswith("_c.png"))
        if dirs:
            out[split] = dirs
        else:
            out[split] = pairs
            out["unit"] = "pairs"
    return out


def list_folders(folder):
    """Subfolders plus drive letters, for the built-in picker. Browsers cannot
    give a real folder path from <input type=file>, so the GUI walks the disk
    through this instead of asking people to paste paths."""
    out = {"path": folder or "", "parent": "", "dirs": [], "drives": []}
    if os.name == "nt":
        out["drives"] = [f"{c}:\\" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                         if path.exists(f"{c}:\\")]
    if not folder:
        out["path"] = out["drives"][0] if out["drives"] else path.abspath(os.sep)
        folder = out["path"]
    folder = path.normpath(path.expanduser(folder))
    out["path"] = folder
    parent = path.dirname(folder)
    out["parent"] = parent if parent and parent != folder else ""
    try:
        for name in sorted(os.listdir(folder), key=str.lower):
            if name.startswith("."):
                continue
            if path.isdir(path.join(folder, name)):
                out["dirs"].append(name)
    except OSError as e:
        out["error"] = str(e)
    return out


def full_state():
    cfg = cfgmod.load()
    p = cfgmod.paths(cfg)
    cfg.pop("pexels_key", None)     # never send the keys back to the page
    cfg.pop("pixabay_key", None)
    return {
        "config": cfg,
        "has_pexels_key": bool(cfgmod.load().get("pexels_key")),
        "has_pixabay_key": bool(cfgmod.load().get("pixabay_key")),
        "paths": p,
        "jobs": {n: MANAGER.state(n, tail=0) for n in BUILDERS},
        "os": os.name,
    }


# ---- http ----------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "ntrainer-gui"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype, cache=None, etag=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if etag:
            self.send_header("ETag", etag)
        # Everything the API returns is live state, so no-store is right for it.
        # Preview PNGs are the exception: one file is one epoch and one sample,
        # written once and never rewritten, so re-fetching it every time the
        # slider passes over it was pure waste -- and the refetch is what made
        # the image go black between epochs.
        self.send_header("Cache-Control", cache or "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json; charset=utf-8")

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8")) or {}
        except Exception:                                          # noqa: BLE001
            return {}

    def _static(self, name):
        safe = path.basename(unquote(name))
        fp = path.join(STATIC, safe)
        if not path.isfile(fp):
            self._send(404, "not found", "text/plain")
            return
        ctype = {".html": "text/html; charset=utf-8",
                 ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8",
                 ".svg": "image/svg+xml"}.get(path.splitext(safe)[1], "text/plain")
        with open(fp, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_GET(self):
        url = self.path.split("?", 1)[0]
        if url in ("/", "/index.html"):
            self._static("index.html")
        elif url.startswith("/static/"):
            self._static(url[len("/static/"):])
        elif url == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        elif url == "/api/state":
            self._json(full_state())
        elif url.startswith("/api/preview/"):
            self._preview(url[len("/api/preview/"):])
        elif url.startswith("/api/job/"):
            slot = url[len("/api/job/"):].strip("/")
            st = MANAGER.state(slot)
            self._json(st or {"name": slot, "running": False, "lines": []})
        else:
            self._send(404, "not found", "text/plain")

    def _preview(self, rest):
        try:
            run, name = rest.split("/", 1)
        except ValueError:
            self._send(404, "not found", "text/plain")
            return
        run, name = unquote(run), unquote(name)
        d = self._run_dir(run)
        # the filename pattern is the whole allow-list: no traversal, no other
        # files out of the run folder
        if not d or not EVAL_RE.match(name):
            self._send(404, "not found", "text/plain")
            return
        fp = path.join(d, "eval", name)
        if not path.isfile(fp):
            self._send(404, "not found", "text/plain")
            return
        # Previews are immutable for a given run, but a run REBUILT under the
        # same name writes a different epoch0004_1.png to the same URL. Marking
        # them immutable would then show the previous run's pictures until the
        # browser cache was cleared by hand. Revalidating instead costs one
        # empty 304 and still avoids re-sending the image.
        try:
            mtime = int(path.getmtime(fp))
        except OSError:
            mtime = 0
        tag = f'"{mtime:x}-{path.getsize(fp):x}"'
        if self.headers.get("If-None-Match") == tag:
            self.send_response(304)
            self.send_header("ETag", tag)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return
        with open(fp, "rb") as f:
            self._send(200, f.read(), "image/png", cache="no-cache", etag=tag)

    def do_POST(self):
        url = self.path.split("?", 1)[0]
        body = self._body()
        try:
            if url == "/api/config":
                cfgmod.save(body)
                p = cfgmod.paths()
                cfgmod.ensure_data_dirs(p)
                self._json(full_state())
            elif url == "/api/detect":
                self._json({"nunif_dir": cfgmod.detect_nunif()})
            elif url == "/api/check_nunif":
                self._json(check_nunif(body.get("path") or ""))
            elif url == "/api/browse":
                self._json(list_folders(body.get("path") or ""))
            elif url == "/api/inspect":
                self._json(folder_stats(body.get("path") or ""))
            elif url == "/api/models":
                p = cfgmod.paths()
                self._json({
                    "models": find_models(p["models_dir"]),
                    "models_dir": p["models_dir"],
                    "hook": hook_status(p),
                    "yml": {"path": yml_path(p["nunif_dir"]) if p["nunif_dir"] else "",
                            "names": read_yml_names(p["nunif_dir"]) if p["nunif_dir"] else []},
                })
            elif url == "/api/inspect_model":
                self._json(inspect_model(body.get("path") or ""))
            elif url == "/api/install":
                p = cfgmod.paths()
                if not p["ready"]:
                    raise RuntimeError("set the nunif folder first")
                done = []
                fp, pinned = install_model(p, body.get("path") or "",
                                           body.get("name") or "",
                                           body.get("kind") or "")
                done.append(f"frozen a copy of the weights as {pinned}, so further "
                            f"training cannot change what this entry means")
                done.append(f"added \u201c{body.get('name')}\u201d to {fp}")
                lowres = bool(body.get("lowres", True))
                border = bool(body.get("border", True))
                cfgmod.save({"install_lowres": lowres, "install_border": border})
                ok, checks, notes = install_extras(p, (body.get("name") or "").strip(),
                                                   body.get("kind") or "video",
                                                   lowres, border)
                self._json({"ok": ok, "did": done + notes, "checks": checks})
            elif url == "/api/runs":
                mods = cfgmod.paths()["models_dir"]
                self._json({"models_dir": mods,
                            "runs": [{"name": r["name"]} for r in list_runs(mods)]})
            elif url == "/api/progress":
                self._json(self._progress(body.get("run") or ""))
            elif url == "/api/dataset":
                self._json(dataset_stats(body.get("path") or ""))
            elif url == "/api/clean":
                removed, failed = sweep_incomplete(body.get("path") or "")
                self._json({"removed": len(removed), "failed": len(failed),
                            "names": removed[:20]})
            elif url.startswith("/api/job/") and url.endswith("/start"):
                self._start(url[len("/api/job/"):-len("/start")], body)
            elif url.startswith("/api/job/") and url.endswith("/stop"):
                slot = url[len("/api/job/"):-len("/stop")]
                self._json({"stopped": MANAGER.stop(slot)})
            else:
                self._send(404, "not found", "text/plain")
        except Exception as e:                                     # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 400)

    def _run_dir(self, run):
        """Resolve a run name to a folder, refusing anything outside the models
        directory -- the name arrives from the page and ends up in a file path."""
        mods = path.abspath(cfgmod.paths()["models_dir"])
        name = path.basename((run or "").strip())
        if not name:
            return None
        d = path.abspath(path.join(mods, name))
        if path.dirname(d) != mods:
            return None
        return d

    def _progress(self, run):
        d = self._run_dir(run)
        if not d:
            return {"exists": False, "rows": [], "previews": {}, "preview_epochs": [],
                    "slots": []}
        out = run_progress(d)
        out["name"] = path.basename(d)
        job = MANAGER.state("train", tail=0)
        # Only claim the live numbers when this is the run actually training.
        if job and job.get("running") and (job.get("label") or "").find(
                path.basename(d)) >= 0:
            out["job"] = {"running": True, "eta": job.get("eta"),
                          "done": job.get("done"), "total": job.get("total"),
                          "step": job.get("step"), "steps": job.get("steps"),
                          "percent": job.get("percent"), "sub": job.get("sub")}
        else:
            out["job"] = {"running": False,
                          "any": bool(job and job.get("running")),
                          "other": (job or {}).get("label", "") if job and job.get("running") else ""}
        return out

    def _start(self, slot, body):
        if slot not in BUILDERS:
            self._json({"error": f"unknown job {slot}"}, 400)
            return
        p = cfgmod.paths()
        if not p["ready"]:
            self._json({"error": "Set the nunif folder first: press Settings, top "
                                 "right, and give it the folder that contains "
                                 "nunif\\ and iw3\\."}, 400)
            return
        cfgmod.ensure_data_dirs(p)
        opts = dict(body or {})
        # keys live in config.json; the page only sends them when they change
        saved = cfgmod.load()
        opts.setdefault("pexels_key", saved.get("pexels_key", ""))
        opts.setdefault("pixabay_key", saved.get("pixabay_key", ""))
        spec = BUILDERS[slot](p, opts)
        cls = jobsmod.ChainJob if "steps" in spec else jobsmod.Job
        # Every tool imports `iw3` and `nunif`, and the bundled ones also import
        # `ntrainer`. Running a script by path puts the SCRIPT's folder on
        # sys.path, not the working directory, so the tools' own relative guess
        # at where nunif lives is wrong from inside GUI\tools. Say it outright
        # instead: this is the one thing the whole GUI depends on knowing.
        env = dict(spec.get("env") or {})
        parts = [p["nunif_dir"], p["gui_dir"]]
        existing = os.environ.get("PYTHONPATH", "")
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(x for x in parts if x)
        env.setdefault("NT_CWD", p["gui_dir"])
        spec["env"] = env
        try:
            job = MANAGER.start(slot, cls=cls, **spec)
        except RuntimeError as e:
            self._json({"error": str(e)}, 409)
            return
        self._json(job.state())


def main():
    port = 8090
    for a in sys.argv[1:]:
        if a.startswith("--port"):
            try:
                port = int(a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1])
            except (ValueError, IndexError):
                pass
    no_browser = "--no-browser" in sys.argv
    host = "0.0.0.0" if "--lan" in sys.argv else "127.0.0.1"

    cfgmod.ensure_data_dirs(cfgmod.paths())

    httpd = None
    for _ in range(25):
        try:
            httpd = ThreadingHTTPServer((host, port), Handler)
            break
        except OSError:
            port += 1
    if httpd is None:
        print(f"could not bind a port near {port}", file=sys.stderr)
        return 1

    url = f"http://{'localhost' if host == '127.0.0.1' else _lan_ip()}:{port}/"
    print("=" * 60)
    print(" iw3 inpaint trainer -- all-in-one GUI")
    print("=" * 60)
    print(f" open : {url}")
    print(f" data : {cfgmod.DATA_DIR}")
    print(" close this window to stop the GUI (running jobs stop with it)")
    print()
    if not no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
        for name in MANAGER.any_running():
            MANAGER.stop(name)
    return 0


def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    sys.exit(main())
