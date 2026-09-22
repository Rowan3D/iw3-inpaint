# iw3-inpaint trainer

Double-click **`launch_gui.bat`**. It opens in your browser at
`http://localhost:8090`.

That is the whole thing. Close the black window to stop it.

## What it is

A local web page served by a small Python program in `app\`. Nothing is sent
anywhere; the server only listens on this machine.

The page itself needs no libraries at all — any Python 3.8+ can run it. The
actual work (downloading, preparing, training) runs as separate processes using
**nunif's own Python**, the one that has PyTorch in it. You point the GUI at
your nunif install once, on the first run, and it remembers.

## The five steps

| | |
|---|---|
| 1 Collect | Download stock video or images, or point at footage you already have |
| 2 Prepare | Warp and mask it into training pairs |
| 3 Train | Set the model up and start training |
| 4 Live view | Loss graph and eval previews while it runs |
| 5 Install | Put the finished model into iw3 |

Each step hands its output folder to the next, so you only choose a folder once.
Every setting has a tooltip — hover it. The defaults are the recommended values
unless the tables below say otherwise.

**First run:** open **Settings**, point it at your nunif install (Auto-detect
usually finds it), then press **Install dependencies** once. It adds
huggingface_hub, pyarrow, lpips, dctorch, tqdm, schedulefree and requests to
nunif's Python.

---

### 1 · Collect

Downloads royalty-free footage, or uses a folder of your own.

* **Video clips** (recommended) — trains the temporal model, which is what iw3
  uses for video. **Still images** — for a stills-only model. **I already have
  files** — skip downloading.
* **Sources:** Pexels and Pixabay for video (free API key each — **Test keys**
  checks them without downloading). pd12m (public domain, sharp throughout —
  recommended) or 4klsdb (native 4K, but lots of blurred backgrounds) for images.
* **Resumable** — stop any time, press Start again later to add more.
* **Scene mix** — balances the download across people / urban / interior /
  transport / nature, so the model does not only see one kind of place.
  Default: people 25, urban 30, interior 10, transport 10, nature 20, other 5.
* **Filters on by default:** skips AI-generated clips (the model would learn
  their artefacts), respects contributors' "no AI training" flags, skips clips
  the site rates as low quality.
* **Check folder / Remove empty files** — finds and cleans up failed or
  half-finished downloads.

| Setting | Default | Recommended |
|---|---|---|
| How many | 400 | 400 for a first model; 1000+ for a serious one (each clip ≈ 20 MB) |
| Minimum width | 1920 | keep ≥ 1920 — wider footage makes the wide holes you are training for |
| Maximum width | 3840 | 3840 |
| Shortest / longest clip | 5 s / 60 s | leave |
| Parallel downloads | 6 | lower if downloads start failing |

### 2 · Prepare

Estimates depth, warps each frame with iw3's own warp at a random 3D strength,
and saves the masked / real pairs the model learns from.

* **Check footage first** — reports resolution, frame rate, length and how many
  clips you will get, in seconds, without writing anything.
* **Held-out eval set** — whole videos are held back, so no eval clip is ever
  trained on and the eval number is honest.
* **Detail filter** — drops clips whose holes land on flat or out-of-focus
  areas; blurred backgrounds only teach the model to paint mush.
* **Multiple working widths** — frames are warped at 1280 and 1920 wide, so the
  model sees the hole sizes iw3 produces at both.
* **Depth edge refinement** — pulls depth edges onto the real edges in the
  picture (edge ramp 4.9 px → 1.3 px), so the masks are sharp. Leave it on; it
  cannot be added later without rebuilding the dataset.
* **Mirrored copies** saved by default (free extra data).
* **Resumable** — finished clips are skipped when you start again.

| Setting | Default | Recommended |
|---|---|---|
| Frames per clip | 12 | leave — the model is built around 12 |
| Frame gap (stride) | 4 | 4; raise for slow / high-frame-rate footage |
| Sample rate | 24 fps | leave |
| Clips per video | 2 | 2; more only for long, varied videos |
| Start sampling at | after 3 s | skips titles and fades |
| Saved crop size | 640 | must be bigger than the training crop — 640 covers a 384 or 512 crop |
| Divergence range | 2 – 16 | 2 – 16; raise the top if you convert at higher strength |
| Convergence range | 0 – 1 | leave |
| Working widths | 1280 1920 | leave |
| Minimum hole pixels | 300 | leave |
| Depth variation | safe | safe |
| Depth model | Any_V3_Mono | best quality; **Any_S** is the easy speed-up for huge datasets |
| Depth resolution | 784 | leave |
| Depth / mask batch | 4 / 4 | lower if you run out of VRAM |
| Only process N videos | 0 | set ~20 for a quick trial run first |

### 3 · Train

* **Image or video model**, three sizes: Small (~9M), **Base (~16M,
  recommended)**, Large (~38M).
* **Run chains** — "3 runs of 200 epochs" queues itself: each run keeps the
  weights and restarts the learning rate. Stop and press Start again with
  **Continue where it stopped** and it finishes the plan it was in, not a new
  one. **Keep training past the last run** adds runs until you press Stop, and
  can be flipped while training.
* **Every run's best is archived** (`.phase1.pth`, `.phase2.pth` …) before the
  next starts, so a worse later run can never overwrite a better model. At the
  end it publishes the best run, not the last.
* **Measure speed and memory** — runs a few real steps at your exact settings
  and reports time per step and VRAM, in under a minute. Tick *compare all model
  sizes* to benchmark Small / Base / Large together. **Use this before any long
  run.**
* **Averaged model (EMA)** on by default — smoother, usually slightly better
  weights at no extra training time; this is what gets evaluated and installed.
* **Hard example mining** — revisits the crops it does worst on.
* **Critic (GAN)** optional — pushes for sharp fills instead of the blurry
  average. Can start at a later run, so a good model is already banked when it
  joins.
* **Faster perceptual loss** — the VGG loss runs on 4 frames per clip instead
  of all 12 (it was 62% of the compute per step), same result on average.
* **torch.compile support** — *Enable compile support* fetches the headers the
  embedded Python lacks, so *Compile the model* works on nunif-windows.

| Setting | Default | Recommended |
|---|---|---|
| Epochs per run × runs | 200 × 3 | 200 × 3 (results keep improving up to ~600 epochs) |
| Learning rate resets every | 40 | leave — the dip after each reset is expected |
| Samples per epoch | 20000 | leave; this, not dataset size, sets epoch length |
| Model size | Base | Base; Large only with lots of data and VRAM |
| Crop size | 384 | **384**; 512 if VRAM allows. Multiples of 128 only. 256 is too small for high strength — a 130 px hole is half the crop |
| Optimizer | CAME | CAME |
| Learning rate / Loss | blank | leave blank — the right one is picked for you |
| Perceptual frames per clip | 4 | 4 |
| Loader threads | 8 | ≈ your CPU cores on an SSD; fewer on a hard drive |
| Averaged copy (EMA) | on, 0.999 | leave |
| Which models to keep | best only | best only; *snapshot every 20* if you want to go back |
| Check every / samples | 4 / 256 | leave |
| Critic | none | none for a first model; then **l3c**, strength 0.2, warm-up 500 |
| Critic starts at run | 3 | 3 until you have seen one behave on your footage; 1 also worked on the released model's data |
| Fast-maths format | fp16 | **bfloat16** when using a critic (RTX 30xx / 40xx) |
| Compile | off | try it with *Measure speed* — on an RTX 4080 at 384 it measured ~8 it/s compiled, against ~3.8 for the uncompiled run |
| Channels-last | off | leave off unless the benchmark shows it helps |

**Time:** on an RTX 4080 one 200-epoch video run is roughly half a day at
crop 256 and more at 384 — read the real figure off Live view after the first
few epochs.

### 4 · Live view

* Train and eval loss on one graph (separate axes), with a divider at every
  run boundary so a restart does not read as a regression.
* Eval previews every check — masked input, mask, prediction, ground truth —
  across the clip so you see it move. Drag the **epoch** slider to watch the
  same crop improve through training; **follow latest** keeps up with new ones.
* Start / Stop / Continue from here too.

What to watch: eval loss should keep falling across runs. With a critic, a
small rise when it joins is normal; a steady climb over several checks means
lower the critic strength or turn it off. If previews look sharper while eval
flattens, trust the previews.

### 5 · Install

* Pick a trained run (its best model) or any `.pth` file; **Check file** reads
  its architecture.
* Copies the weights to a frozen file first (see below), then installs the
  model **and every iw3-inpaint feature** — mask handling, window reuse, VRAM
  fixes — with checkboxes for **low-res inpainting** and the **screen-edge
  fix**. Re-using a name replaces that entry.
* **Check the install** asks iw3 itself, in a fresh process, whether each piece
  is live and the model loads.

---

## Portable

This folder is self-contained — `tools\`, `ntrainer\` and `iw3_extras\` are
bundled, so you can move or copy the whole folder anywhere, including another
machine. On a new machine it asks for the nunif path again and nothing else.

`tools\fetch_videos.py` only renames a download once the transfer has
completed and is a plausible size, so a run stopped mid-download never leaves
0-byte or truncated `.mp4` files behind.

## Files it creates

```
config.json          your settings (including API keys - keep it to yourself)
data\downloads\      downloaded footage    (unless you chose another folder)
data\dataset\        prepared training pairs
data\models\         trained models and loss graphs
data\models\installed\   frozen copies of whatever you installed into iw3
```

### Why `installed\` exists

A training run keeps one file for its best model and **rewrites it** every time
eval improves. Worse, each new run (`--reset-state`) zeroes the best-so-far
score, so the first eval of the next run overwrites that file even though it is
worse than what the previous run reached.

So the Install step never points iw3 at the run folder. It copies the weights
you picked into `data\models\installed\<name>.pth` first and points the yml
there. An entry called `HQ_Inpaint_e199` keeps meaning epoch 199 for as long as
it exists. The cost is ~85 MB per install; delete the file and the yml entry
together when you are done with one.

Each run's best is also archived as `inpaint.<arch>.phase<N>.pth` inside the run
folder when the next run starts, so earlier runs stay installable. The "save as"
step at the end of training compares every run's best eval in `progress.csv` and
publishes the winner — not simply the last one, and not whatever file happens to
sort first in the folder.

## If it will not start

**"No Python was found"** — the GUI looks for `python.exe` beside the folder
(`<install>\python\python.exe`), then on your PATH. The easy fix is to put this
trainer folder inside your nunif install. Failing that,
install Python from python.org with "Add python.exe to PATH" ticked.

**The page loads but every job fails** — open **Settings** and check the nunif
path. It must be the folder that has `nunif\` and `iw3\` inside it, and there
must be a `python\` folder next to it.

**Port already in use** — it tries 8090 and counts upward, and prints the one it
settled on. Pass `--port=9000` to `launch_gui.bat` to choose.

Add `--lan` to reach the GUI from another machine on your network. Anyone who
can reach that address can start jobs, so only do this on a network you trust.

## Step 5 and your iw3 install

The Install step runs the same installer as the
[standalone one](../installer) (bundled here as `iw3_extras\`), pointed at the
frozen copy of the model you picked. The checkboxes under the Install button
are the installer's options: **low-res inpainting** and the **screen-edge fix**.
Everything the installer does comes with it — the mask handling, window reuse,
VRAM fixes and the `[ok]` / `[!!]` self-check.

It writes, all reversible:

* `<nunif>\nt_inpaint\` — the code, settings and uninstaller;
* one entry in `iw3\inpaint_models.yml` (original backed up beside it as
  `.before-nt-inpaint`);
* two small files in nunif's site-packages that switch it on without importing
  PyTorch until iw3 itself is loaded.

To undo it, run `<nunif>\nt_inpaint\uninstall.bat`.

[Auto 3D Strength](https://github.com/Rowan3D/iw3-auto3d) and
[low-res only](https://github.com/Rowan3D/iw3-lowres) are separate installers.
