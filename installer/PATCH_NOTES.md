# iw3 Inpainting Extras — feature list / patch notes

Everything below is applied at run time, in memory. **No file belonging to
nunif or iw3 is modified**, so updating iw3 cannot collide with any of it.
Every piece can be switched off on its own, and the installer proves each one
is live before it finishes.

---

## Installation

+ **One click.** `install.bat` runs from anywhere — Desktop, Downloads, a USB
  stick. No Python needed on the system: it finds the one iw3 already uses.
+ **Finds iw3 by itself**, in this order — `NUNIF_DIR` if set, a nunif install
  around the folder you unzipped into, any Python on PATH, and failing all that
  it asks for the path.
+ **Pick what you want.** Low-res patch, trained model, or both; then a separate
  yes/no for the screen-edge fix. Re-running with a different set replaces the
  old one cleanly. (Auto 3D Strength has its own separate installer.)
+ **Self-check.** After installing, a separate Python process imports iw3 the
  way iw3 does and prints `[ok]` or `[!!]` for each piece — architectures
  registered, mask patch live, model list entry visible, model file actually
  loads, window reuse active, low-res active, screen-edge fix active. You know
  it works before you start a conversion, not three hours in.
+ **Everything lives in one folder**, `<nunif>\nt_inpaint\` — code, the model,
  settings, and the uninstaller. Two small files go in site-packages to switch
  it on. One line is added to `iw3\inpaint_models.yml`, with the original saved
  beside it as `.before-nt-inpaint`.
+ **No torch at startup.** The hook installs a lazy import hook rather than
  importing torch, so `pip`, `python` and every other command in that install
  start as fast as they did before. Nothing loads until something imports iw3.
+ **Refuses to write nonsense.** The installer reads the model's architecture
  before writing the yml entry; if the .pth isn't an inpainting model it stops
  instead of leaving an entry iw3 can't load.
+ **Clean uninstall** — `<nunif>\nt_inpaint\uninstall.bat`, or
  `NT_INPAINT_DISABLE=1` to switch everything off without removing anything.
  Installing without the model removes the stale yml entry so iw3 never offers
  a model whose architecture is no longer registered.

---

## 1. Low-resolution inpainting, full-resolution output

> Works with **any** inpainting model, including the one iw3 ships with.

+ **"Inpaint Max Width" no longer shrinks your video.** Stock iw3 resizes the
  whole frame before the model sees it, and everything after that — including
  the output — is that smaller picture. Now the frame stays full size and only
  the model runs small: you get the speed of a 1280-wide pass with the
  resolution of the source.
+ **The detail is put back everywhere except inside the holes.** What the
  downscale threw away is added back outside the mask, so the original pixels
  return exactly; inside a hole there was nothing to restore, so that is the
  model's fill, upscaled.
+ **Hole-aware downscale.** The downscale divides by valid-pixel weight instead
  of averaging raw pixels, so the black of an unfilled hole is never smeared
  into its neighbours on the way down — the model isn't asked to match a halo
  that isn't there.
+ **Holes can't disappear on the way down.** The mask is max-pooled rather than
  sampled. A hole is three or four pixels wide, and at 3840 → 1280 a
  nearest-neighbour sample lands between the bands and drops them. Measured on
  3 px bands at 1920: sampling kept 8 of 12, max-pooling keeps 10.
+ **The seam lands where the model put it.** The blend uses the model's own
  soft-edged mask, not the hard one, so kept detail and fresh fill meet in the
  same place the model chose.
+ **Patched at the model's `infer`, not iw3's internals** — by then the eye is
  already flipped for the correct side and the mask already dilated and closed,
  so this code repeats none of that and keeps working if iw3 changes how it
  gets there.

---

## 2. Trained inpainting model

> Only if a `.pth` was shipped in the installer's `models\` folder.

+ **A 21M-parameter video inpainting model trained for large divergence**,
  where iw3's own model has the least to work with. Divergence 8 and up is
  where it earns its keep; below about 5 the holes are small enough that
  nothing has much to do.
+ **The mask handling it needs comes with it.** iw3 marks only the pixels its
  warp left completely empty — about 0.1% of a frame — while the band that
  actually *looks* stretched is roughly ten times wider and never gets
  repainted. Without this part, any inpainting model produces much the same
  picture as the stock one; this was the difference between "looks identical"
  and "obviously better".
+ **The stretched band is found from the warp itself.** The warp already
  carries, for every output pixel, the source x it came from. Where that index
  barely advances, the picture is being smeared — a free, exact damage
  detector in output coordinates. Short runs are ignored and the result is
  grown a couple of pixels so the model gets a clean edge to work from.
+ **Stock models are not touched by it.** The mask change only engages for
  models trained on this kind of mask; iw3's own models are left with the mask
  they were trained on.
+ **Video and image variants share one backbone**, so a video model is a short
  fine-tune on top of the image run rather than a second training run from
  scratch.

---

## 3. Window reuse — each frame through the network once, not twice

> Engages only for a model that advertises it, so stock iw3 models are
> unaffected. *(This one started from a friend's work — thanks.)*

+ **iw3 computes every video frame twice.** It inpaints on a sliding window of
  12 frames and keeps the middle 6, so each frame goes through the whole
  network once as itself and once as its neighbour's context.
+ **Most of that second pass isn't needed.** Only the middle of the network
  actually mixes frames together. Everything above the first temporal block and
  everything below the last sees one frame at a time — so the top half is
  carried over from the previous window and the bottom half runs for the 6
  frames being kept.
+ **About a third less time on video.** Measured on a test clip: the encoder
  went from 48 frame-passes to 26 and the decoder from 48 to 27; with the
  low-res patch as well, 16.3s → 12.4s.
+ **The picture does not change.** Two clips came out bit-identical — zero
  pixels different by more than one 8-bit step.
+ **Frames are tracked by identity, not position.** A global counter that never
  resets follows the frame queue exactly, including the repeated frames iw3
  pads with, so "the same id" always means "the same picture" and a new clip
  can never read the previous clip's cached work.
+ **Falls back silently.** If anything in the chain doesn't take the new
  arguments, it just runs the plain call — no gain, no harm.

---

## 4. Screen-edge fix *(optional)*

+ **The stretched edges at high divergence.** Before warping, iw3 pads the
  frame by repeating its first and last column, so everything the warp pulls in
  from outside the frame is a copy of that one column smeared sideways — about
  80 px per edge at divergence 16 on a 1920-wide video, twice that at 30. No
  inpainting mask covers it, because to iw3 those pixels aren't holes.
+ **iw3 already has the fix and forward_inpaint ignores it.** "Preserve Screen
  Border" fades the parallax to zero at the edges so nothing is pulled from
  outside the frame. `mlbw_l2_inpaint` and `monobw_inpaint` honour it;
  `forward_inpaint` accepts the setting and then drops it. Now it works, with
  iw3's own band width.
+ **Measured: 6.8% of the frame smeared → 0**, at divergence 8, 16 and 30
  alike. Streaks replaced by real detail right up to the edge.
+ **Nothing changes unless you tick the box.** The cost is the same as for the
  other methods — less depth right at the left and right edges.

### Tried and rejected: inpainting the edge band

Filling that band with the model was tested properly — real frames cropped
down, warped, and the band scored against the pixels that had been cut off.
**Plain stretching won on 18 of 18 frames, by 3 to 7 dB.** It is content from
beyond the edge of the frame: there is nothing on that side to fill from, and
the model has never seen a hole that touches the frame edge. Not shipped.

---

## 5. Speed and VRAM

Everything in this section is **bit-identical output** — verified with
`torch.equal`, not eyeballed.

+ **Cold-start spike removed.** The first window of a clip had nothing cached
  and ran the prefix for all 12 frames at once — one spike, at the very start,
  that set the high-water mark for the entire run and hid every saving after
  it. Now run in steady-state-sized groups.
  *Peak 1358 → 870 MiB, and 870 on every window instead of only after the
  first — **35% below stock**, where before it was slightly above.*
+ **Cache entries are copies, not views.** A cached slice kept the whole batch's
  storage alive, so one surviving frame pinned every frame computed beside it.
  *Resident cache 202 → 135 MiB.*
+ **Dead frames are dropped before the expensive part**, not at the start of the
  next call — so the temporal blocks, where the peak is, no longer run with a
  full cache behind them. *135 → 67 MiB.*
+ **The 1/4-resolution skip is built only for the frames the decoder runs**,
  not for the whole window.
+ **Scene boundaries no longer spike.** iw3's flush emits 9 frames instead of
  the usual 6, so the per-frame tail ran half as wide again and that one call
  set the peak for the whole conversion. The tail now runs in fixed groups.
  *Flush window 1444 → 1244 MiB; normal windows unchanged.*
+ **A discarded full-resolution copy of every frame is gone.** The low-res patch
  wanted only the soft mask out of the model's preprocess, which also builds an
  erased copy of the picture at source resolution — over a gigabyte at 4K,
  never read. It now takes just the mask, for the kept frames only, and verifies
  once per model that the shortcut matches exactly before trusting it.
+ **The downscale and the put-back run in groups**, and the put-back keeps one
  source-resolution tensor where it used to hold four.
  *Together: peak 1476 → 1285 MiB, **13% less**, at a 4K-source / 1280-model
  ratio.*
+ **A gigabyte of pointless copying per eye per window** removed from the window
  wrapper — the positions it was copying are exactly the ones iw3 throws away.

### Looked at and left alone

- Caching the low-res downscale between windows — about 2% of the time, and it
  would cost back more VRAM than it saves.
- Dropping `antialias` on the upsamples — changes pixels, and is slower anyway.
- Storing the frame cache in half precision — changes pixels.

---

## Using it

All of this needs an inpainting method: set **Method** to `forward_inpaint`
(or `mlbw_l2_inpaint` / `monobw_inpaint`). The Inpainting Model box and Inpaint
Max Width only appear for those.

| Feature | Where |
|---|---|
| Low-res inpainting | set **Inpaint Max Width** to 1280 or 1920 |
| Trained model | pick it by name in **Inpainting Model** |
| Screen-edge fix | tick **Preserve Screen Border** |
| Window reuse, mask handling | automatic with the model |

iw3 reads its model list once at startup — close it completely and reopen after
installing.

---

## Switches

| Variable | Effect |
|---|---|
| `NT_INPAINT_DISABLE=1` | everything off, nothing removed |
| `NT_INPAINT_VERBOSE=1` | print which pieces loaded |
| `NT_LOWRES_DISABLE=1` | low-res patch off |
| `NT_MASK_DISABLE=1` | mask patch off |
| `NT_WINDOW_DISABLE=1` | window reuse off (slower, more VRAM, same picture) |
| `NT_BORDER_DISABLE=1` | screen-edge fix off |
| `NT_MASK_GRAD=0.4` | how stretched a pixel must be before it's repainted (0.4 = 2.5×) |
| `NT_MASK_RUN=3` | ignore stretched runs narrower than this |
| `NT_MASK_GROW=2` | widen what is marked, in pixels |
| `NT_TAIL_CHUNK=6` | frames per full-resolution pass; lower if still short of VRAM, 0 disables |

---

## Requirements

A working iw3 / nunif install with its own Python — the nunif-windows package,
or a source install with a venv. Nothing is downloaded and nothing else is
installed; the installer only copies files.
