iw3 inpainting extras   https://github.com/Rowan3D/iw3-inpaint
=====================

Upgrades for a normal iw3 install, installed by double-clicking
install.bat. This folder can sit anywhere - Desktop, Downloads, a USB stick.


What you get
------------

1. LOW-RESOLUTION INPAINTING, FULL-RESOLUTION VIDEO

   iw3's "Inpaint Max Width" makes the inpainting faster by shrinking the whole
   frame before the model sees it - and everything after that, including your
   output, is that smaller picture.

   With this installed the frame stays full size. Only the model runs small,
   and the detail lost on the way down is put back everywhere except inside
   the filled holes. You get the speed of a 1280-wide pass and the resolution
   of the source.

   Works with any inpainting model, including the one iw3 ships with.

2. A TRAINED INPAINTING MODEL  (if one was shipped in models\)

   A 21M-parameter video inpainting model trained specifically for large
   divergence, where iw3's own model has the least to work with. It comes with
   the mask handling it needs: iw3 normally marks only the pixels its warp left
   completely empty, about 0.1% of a frame, while the band that actually looks
   stretched is ten times wider and never gets repainted. Without that part,
   any inpainting model produces much the same picture.

   It also runs each frame through the network once instead of twice. iw3
   inpaints video on a sliding window of 12 frames and keeps the middle 6, so
   every frame is computed twice - once as itself, once as its neighbour's
   context. Only the middle of the network actually mixes frames together; the
   parts above and below it see one frame at a time, so those are reused from
   the window before. About a third less time and a third less VRAM on video,
   for a picture that is the same to the last 8-bit step - tested on two clips,
   zero pixels different by more than one step.

   (The stock iw3 models are untouched by it: it only engages for a model that
   says it can do it.)

3. SCREEN-EDGE FIX  (optional, asked separately)

   At high divergence iw3 fills the left and right edges of each eye by
   smearing the outermost column of the picture sideways - about 80 px at
   divergence 16 on a 1920-wide video, twice that at 30. No inpainting mask
   covers it, because to iw3 those pixels are not holes.

   Inpainting that band was tried and measured: it is worse than the smear.
   It is content from beyond the edge of the frame, with nothing on that side
   to fill from. The right fix is the one iw3 already has - "Preserve Screen
   Border", which fades the depth to the screen plane at the edges so nothing
   is pulled from outside the frame - but forward_inpaint silently ignores that
   checkbox. This makes it work, with iw3's own band width. Measured: the
   smeared pixels go from 6.8% of the frame to 0 at divergence 16.

   Nothing changes unless you tick the box. The cost is the same as for the
   other methods: less depth right at the left and right edges.

You choose the first two as a set (either, or both), then yes or no to the
third.

Also available as separate installers:
   Auto 3D Strength (per-scene 3D strength)  https://github.com/Rowan3D/iw3-auto3d
   Low-res inpainting on its own             https://github.com/Rowan3D/iw3-lowres


Installing
----------

   1. Double-click  install.bat
   2. Confirm or type where iw3 is (the folder with nunif\ and iw3\ in it)
   3. Pick what to install
   4. Close iw3 and open it again

The installer checks its own work at the end and prints [ok] or [!!] for each
piece, so you know before you start a conversion.


Using it
--------

The model and the low-res patch need an inpainting method: set Method to
forward_inpaint (or mlbw_l2_inpaint / monobw_inpaint). The Inpainting Model
box and Inpaint Max Width only appear for those.

   * Low-res: set "Inpaint Max Width" to 1280 or 1920. Higher is slower and
     sharper inside the holes; the rest of the frame is unaffected either way.
   * Model: pick it by name in the "Inpainting Model" box.
   * Screen edges: tick "Preserve Screen Border".

Divergence 8 and above is where the trained model earns its keep. Below about
5 the holes are small enough that nothing has much to do.


Where it goes
-------------

   <nunif>\nt_inpaint\            all of it: code, the model, settings
   <python>\Lib\site-packages\    two small files that switch it on
   <nunif>\iw3\inpaint_models.yml one entry added (the original is backed up
                                  beside it as .before-nt-inpaint)

No file belonging to nunif or iw3 is modified. Everything is applied at run
time, in memory, so updating iw3 cannot collide with it.


Removing it
-----------

   <nunif>\nt_inpaint\uninstall.bat

or run install.bat again and pick a different set. NT_INPAINT_DISABLE=1 in the
environment turns everything off without removing anything.


If something goes wrong
-----------------------

   * "Unknown model name" - the two files in site-packages are missing or
     point at the wrong folder. Run install.bat again.
   * The model is in the list but looks identical to the stock one - the mask
     part is not active. Run install.bat again and watch the [ok] lines.
   * Out of VRAM on video - lower "Inpaint Max Width" (1280 is a good start),
     or set NT_WINDOW_DISABLE=1 if you suspect the window reuse.
   * Nothing at all happens - iw3 reads its model list once at startup. Close
     it completely and reopen.

Set NT_INPAINT_VERBOSE=1 to have iw3 print which pieces loaded.


Tuning (optional)
-----------------

Environment variables, if you want to experiment:

   NT_MASK_GRAD=0.4     how stretched a pixel must be before it is repainted.
                        0.4 means 2.5x. Higher repaints more of the smeared
                        band; too high and flat areas go soft.
   NT_MASK_RUN=3        ignore runs narrower than this many pixels
   NT_MASK_GROW=2       widen what is marked, in pixels
   NT_LOWRES_DISABLE=1  turn off the low-res patch only
   NT_MASK_DISABLE=1    turn off the mask patch only
   NT_TAIL_CHUNK=6      how many frames go through the full-resolution stage at
                        once. 6 is what a normal window emits; the flush at a
                        scene boundary emits 9, and this stops that one call
                        from setting the VRAM peak for the whole conversion.
                        Lower it if you are still short of VRAM; 0 turns it off.
   NT_BORDER_DISABLE=1  turn off the screen-edge fix only
   NT_WINDOW_DISABLE=1  turn off the window reuse only (video gets slower and
                        uses more VRAM; the picture does not change)


Requirements
------------

A working iw3 / nunif install with its own Python (the nunif-windows package,
or a source install with a venv). Nothing is downloaded and nothing else is
installed - the installer only copies files.
