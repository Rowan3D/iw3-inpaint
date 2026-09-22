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

Each step hands its output folder to the next one, so you only ever choose a
folder once.

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
