# iw3-inpaint — better inpainting for high 3D strength

Add-ons for [iw3](https://github.com/nagadomi/nunif): a **trained inpainting model built for high 3D strength**, the mask handling it needs, **faster, lower-VRAM video inpainting**, a **fix for stretched screen edges** — and the **trainer** used to make the model, so you can train your own.

No nunif or iw3 file is modified — everything is applied at run time, in memory, and every piece can be switched off on its own.

**[⬇ Download the latest release](https://github.com/Rowan3D/iw3-inpaint/releases/latest)**

| Folder | |
|---|---|
| [`installer/`](installer) | installs the model and the patches into iw3 — most people only need this |
| [`trainer/`](trainer) | a browser GUI that downloads footage, builds a dataset, trains a model and installs it into iw3 |

---

## Installer features

### 1. Trained inpainting model
+ **21M-parameter video inpainting model trained for large divergence**, where iw3's own model has the least to work with. 3D Strength 8 and up is where it earns its keep.
+ **The mask handling it needs comes with it.** iw3 only marks pixels its warp left completely empty (~0.1% of a frame), while the band that actually *looks* stretched is about ten times wider and never gets repainted. This finds that band from the warp itself — where the source index barely advances, the picture is being smeared — and hands it to the model. This is the difference between "looks identical to stock" and "obviously better".
+ **Stock models are untouched** — the mask change only engages for models trained on it.

### 2. Window reuse — each video frame through the network once, not twice
+ iw3 inpaints video on a sliding window of 12 frames and keeps the middle 6, so every frame is computed twice. Only the middle of the network mixes frames; the rest is reused from the previous window.
+ **About a third less time on video**, bit-identical picture (zero pixels off by more than one 8-bit step).
+ Engages only for a model that supports it; falls back silently otherwise.

### 3. Low-res inpainting, full-res output *(optional)*
+ **Inpaint Max Width no longer shrinks your video.** Only the model runs small; detail is put back everywhere except inside the holes. Hole-aware downscale, max-pooled mask, soft-mask blend.
+ Works with **any** inpainting model, including iw3's own. Also available on its own as [iw3-lowres](https://github.com/Rowan3D/iw3-lowres).

### 4. Screen-edge fix *(optional)*
+ At high 3D strength iw3 smears the outermost column sideways — ~80 px per edge at strength 16 on 1920-wide video. iw3's **Preserve Screen Border** fixes this but `forward_inpaint` ignores it; now it works. Measured: **6.8% of the frame smeared → 0**.
+ (Inpainting that band instead was tested and lost on 18 of 18 frames — there's nothing beyond the edge to fill from.)

### 5. Speed and VRAM — all bit-identical output
+ Cold-start spike removed: **peak 1358 → 870 MiB, 35% below stock**.
+ No VRAM jump at scene boundaries (flush tail run in fixed groups).
+ Low-res path: no discarded full-res copy, grouped downscale/put-back — **13% lower peak** at 4K.
+ Cache stores copies not views, dead frames dropped early, no pointless per-window copying.

### Installer
+ One click, no Python needed — finds iw3 and uses its Python.
+ Pick the model, low-res patch, or both; yes/no for the screen-edge fix. Re-run to change.
+ **Self-check**: a separate process imports iw3 the way iw3 does and prints `[ok]`/`[!!]` per piece before you start a conversion.
+ No torch at startup — a lazy hook waits for iw3 to be imported.
+ Clean uninstall; `NT_INPAINT_DISABLE=1` switches everything off without removing it.

Full details, measurements and tuning variables: [installer/PATCH_NOTES.md](installer/PATCH_NOTES.md).

## Install

1. Download **`iw3-inpaint-installer.zip`** from [Releases](https://github.com/Rowan3D/iw3-inpaint/releases/latest) (it includes the model) and unzip it anywhere
2. Double-click **`install.bat`**
3. Confirm or type where iw3 is (the folder with `nunif\` and `iw3\` in it)
4. Pick what to install
5. Close iw3 completely and open it again

## Use

Set **Method** to `forward_inpaint` (or `mlbw_l2_inpaint` / `monobw_inpaint`).

| Feature | Where |
|---|---|
| Trained model | pick it in **Inpainting Model** |
| Low-res inpainting | **Inpaint Max Width** 1280 or 1920 |
| Screen-edge fix | tick **Preserve Screen Border** |
| Window reuse, mask handling | automatic with the model |

## Uninstall

`<nunif>\nt_inpaint\uninstall.bat`, or run install.bat again and pick a different set.

---

## Trainer

A local browser GUI (`trainer/launch_gui.bat` → `http://localhost:8090`) that takes you from nothing to a model installed in iw3, in five steps:

| Step | |
|---|---|
| **1 Collect** | download royalty-free stock video / images (Pexels, Pixabay — free API keys; pd12m, 4klsdb image sets), or use your own footage. Resumable, filters for resolution, length, AI-generated and no-AI-training clips |
| **2 Prepare** | warp and mask footage into training pairs with iw3's own warp — random 3D strength / convergence, multiple widths, depth reshaping, blur and tiny-hole filtering, whole-video eval hold-out |
| **3 Train** | image or video (temporal) model, resumable runs, best-eval tracking across runs, publishes the best run rather than the last |
| **4 Live view** | loss graph and eval previews while it trains |
| **5 Install** | copies the chosen checkpoint to a frozen file and installs it into iw3 **with every installer feature above** (checkboxes for low-res and screen-edge fix) |

Nothing is sent anywhere; the server only listens on your machine. Heavy work runs in nunif's own Python. See [trainer/README.md](trainer/README.md).

## Requirements

A working iw3 / nunif install with its own Python — the [nunif-windows](https://github.com/nagadomi/nunif/blob/master/windows_package/docs/README_EN.md) package, or a source install with a venv. An NVIDIA GPU for training.

---

## The iw3 tools

Three add-ons for [iw3](https://github.com/nagadomi/nunif) (the 2D → 3D converter in nagadomi's nunif). Each one installs and uninstalls **on its own** — use any one, any two, or all three.

| Tool | What it does | Download |
|---|---|---|
| [**iw3-inpaint**](https://github.com/Rowan3D/iw3-inpaint) | A trained inpainting model built for high 3D strength, the mask handling it needs, faster, lower-VRAM video inpainting, a fix for stretched screen edges — and the trainer to make your own models | [latest release](https://github.com/Rowan3D/iw3-inpaint/releases/latest) |
| [**iw3-auto3d**](https://github.com/Rowan3D/iw3-auto3d) | **Auto 3D Strength** — picks the 3D strength per scene from how close the shot is: landscapes get less, close-ups more | [latest release](https://github.com/Rowan3D/iw3-auto3d/releases/latest) |
| [**iw3-lowres**](https://github.com/Rowan3D/iw3-lowres) | **Low-res inpainting, full-res output** — iw3's "Inpaint Max Width" without shrinking your video | [latest release](https://github.com/Rowan3D/iw3-lowres/releases/latest) |

## Credits & license

MIT — see [LICENSE](LICENSE).
iw3 / nunif by [nagadomi](https://github.com/nagadomi/nunif) (MIT). The window-reuse idea started from a friend's work — thanks.
