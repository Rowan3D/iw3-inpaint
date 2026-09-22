r"""
Inpaint at a lower resolution, keep the full-resolution picture.

What iw3 does on its own
------------------------
"Inpaint Max Width" in iw3 shrinks the whole frame before the inpainting model
sees it:

    x = self._resize(x, max_width)        # base_inpaint.py, infer()

and everything after that -- including the output -- is that smaller picture.
It is a speed and memory setting, and it costs you the resolution of the whole
frame to pay for a model that only ever touches the holes.

What this does instead
----------------------
The frame stays full size. Only the model runs small, and the detail that was
lost on the way down is added back outside the mask:

    low       = downscale(eye, max_width)        valid pixels only, renormalised
    filled    = model(low, mask_low)
    detail    = eye - upscale(low)               what the downscale threw away
    eye       = upscale(filled) + detail * (1 - blurred_mask)

Inside the hole there is no detail to restore, so it is the model's fill,
upscaled. Everywhere else the original pixels come back exactly. The result is
a full-resolution frame from a model that ran at, say, 1280 wide.

The downscale divides by the valid-pixel weight rather than averaging raw
pixels, so the black of an un-filled hole is not smeared into its neighbours on
the way down.

How it is applied
-----------------
No file in iw3 is edited. Two methods are wrapped:

  * `BaseImageInpaint.infer` / `BaseVideoInpaint.infer` -- remember max_width
    for this call and pass None down, which turns iw3's own pre-resize into a
    no-op, so the frame keeps its size.
  * the inpaint model's own `infer` -- where the work happens. It is wrapped
    once per component, the first time that component runs.

Wrapping the model's infer rather than iw3's `_inpaint_single` is deliberate:
by then the eye is already flipped for the correct side and the mask is already
dilated and closed, so this code does not have to know or repeat any of that,
and it keeps working if iw3 changes how it gets there.

NT_LOWRES_DISABLE=1 turns it off.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F


__all__ = ["install", "low_res_infer"]

_installed = False


def _even(n):
    n = int(n)
    return n + 1 if n % 2 else n


def _chunk():
    """Frames per full-resolution pass. 6 = what a normal window emits, so the
    ordinary case is unchanged; NT_TAIL_CHUNK=0 turns it off."""
    try:
        return int(os.environ.get("NT_TAIL_CHUNK", 6))
    except ValueError:
        return 6


def _blur_only(model, eye, mask):
    """The soft mask `model.preprocess(eye, mask)` returns, without the erased
    picture it returns beside it.

    preprocess computes `x * (1 - mask)` -- a full copy of the frames at source
    resolution -- and this code only ever wants the mask. Taking the same two
    lines directly is the same tensor, and it is checked against preprocess on
    every call the first time a model is seen."""
    m = mask if mask.is_floating_point() else mask.float()
    blur = getattr(model, "mask_blur", None)
    if blur is not None and getattr(model, "_nt_blur_ok", None) is not False:
        out = torch.clamp(blur(m) + m, 0, 1)
        if getattr(model, "_nt_blur_ok", None) is None:
            # prove the shortcut matches before trusting it for the rest of the run
            try:
                _, ref = model.preprocess(eye[0:1], mask[0:1])
                model._nt_blur_ok = bool(torch.equal(ref, out[0:1]))
            except Exception:           # noqa: BLE001
                model._nt_blur_ok = False
            if not model._nt_blur_ok:
                return model.preprocess(eye, mask)[1]
        return out
    if hasattr(model, "preprocess"):
        return model.preprocess(eye, mask)[1]
    return m


def low_res_infer(model, eye, mask, max_width, original_infer,
                  keep=None, ids=None, cache=None):
    """The model runs at max_width; the frame stays the size it came in at.

    `keep` (see window.py) = the frames the caller will actually use. The
    downscale still covers the whole window, because the model mixes every frame
    in it, but the full-resolution half -- the two bicubic upsamples and the
    detail put-back, by far the most expensive pixels here -- is done for those
    frames only."""
    new_w = _even(max_width)
    new_h = _even(round(eye.shape[-2] * (new_w / eye.shape[-1])))
    size = (max(8, new_h), max(8, new_w))

    # Weighted downscale: sum(valid pixels) / sum(valid weight). A plain
    # interpolate() would pull the black of the hole into the pixels beside it,
    # and the model would then be asked to match a halo that is not really there.
    #
    # Max-pool the mask instead of sampling it: a hole is three or four pixels
    # wide, and at 3840 -> 1280 a nearest-neighbour sample lands between the
    # bands and drops them, leaving the model asked to fill nothing while the
    # damage is still there. Any masked pixel in a block masks the block, so a
    # band can never disappear on the way down -- at worst it gets a pixel wider,
    # which is the harmless direction. Measured on 3px bands at 1920: nearest
    # kept 8 of 12, this keeps 10.
    #
    # Done a few frames at a time: `eye * valid` is a whole extra copy of the
    # window at source resolution, and at 4K that is over a gigabyte held for
    # nothing. Per-frame work, so the numbers do not change.
    def down(e, m):
        valid = 1.0 - m
        low_sum = F.interpolate(e * valid, size=size, mode="bilinear",
                                antialias=True, align_corners=False)
        low_weight = F.interpolate(valid, size=size, mode="bilinear",
                                   antialias=True, align_corners=False)
        return (low_sum / torch.clamp(low_weight, min=1e-6),
                F.adaptive_max_pool2d(m, size) > 0)

    n, step = eye.shape[0], _chunk()
    step = n if step <= 0 or step >= n else step
    parts = [down(eye[i:i + step], mask[i:i + step].float()) for i in range(0, n, step)]
    low = torch.cat([p[0] for p in parts], dim=0)
    low_mask = torch.cat([p[1] for p in parts], dim=0)
    del parts

    if keep is None:
        filled = original_infer(low, low_mask)
    else:
        filled = original_infer(low, low_mask, keep=keep, ids=ids, cache=cache)
        if filled.shape[0] == len(keep):
            idx = torch.as_tensor(list(keep), dtype=torch.long, device=eye.device)
            eye = eye.index_select(0, idx)
            low = low.index_select(0, idx)
            mask = mask.index_select(0, idx)

    full = eye.shape[-2:]

    def compose(e, lo, fi, m):
        # The model's own preprocess() gives the soft-edged mask it will actually
        # erase with, so the seam between kept detail and fresh fill lands in the
        # same place the model put it. Anything without preprocess() gets the hard
        # mask, which is what iw3's stock model works with anyway.
        #
        # _blur_only() is that same mask without preprocess's `x * (1 - mask)`,
        # which is a source-resolution copy of the picture that nothing here ever
        # reads. Taken per chunk and for the kept frames only.
        bm = _blur_only(model, e, m)
        low_up = F.interpolate(lo, size=full, mode="bicubic",
                               antialias=True, align_corners=False)
        # filled_up + (e - low_up) * (1 - bm), with the two multiplies folded
        # into the one tensor and the upscaled low frames dropped before the
        # upscaled fill is made. Same operations in the same order; at 4K each
        # source-resolution temporary is ~300 MB per six frames.
        detail = e - low_up
        del low_up
        if torch.result_type(detail, fi) == detail.dtype:
            detail.mul_(1.0 - bm)
            detail.add_(F.interpolate(fi, size=full, mode="bicubic",
                                      antialias=True, align_corners=False))
            return detail
        return F.interpolate(fi, size=full, mode="bicubic",
                             antialias=True, align_corners=False) \
            + detail * (1.0 - bm)

    # These are the largest tensors anywhere in the run -- two upsamples to the
    # source resolution, per frame -- so how many frames are in flight here is
    # the single biggest memory decision. A normal window has 6; the flush at a
    # scene boundary has 9, and that one call would otherwise set the peak for
    # the whole conversion. Per-frame work, so the result does not change.
    n, step = eye.shape[0], _chunk()
    if step <= 0 or step >= n:
        return compose(eye, low, filled, mask)
    out = None
    for i in range(0, n, step):
        part = compose(eye[i:i + step], low[i:i + step],
                       filled[i:i + step], mask[i:i + step])
        if out is None:
            out = part.new_empty((n,) + part.shape[1:])
        out[i:i + step] = part
    return out


def _wrap_model(component):
    """Wrap this component's inpaint model once, in place."""
    model = getattr(component, "model", None)
    if model is None or getattr(model, "_nt_lowres_wrapped", False):
        return
    original_infer = model.infer

    def infer(eye, mask, *args, **kwargs):
        max_width = getattr(component, "_nt_max_width", None)
        if (max_width and torch.is_tensor(eye) and eye.ndim >= 3
                and eye.shape[-1] > max_width):
            # the window arguments (window.py) belong to the model, not to iw3:
            # take them out of the call and hand them on explicitly
            window = {k: kwargs.pop(k) for k in ("keep", "ids", "cache") if k in kwargs}
            return low_res_infer(model, eye, mask, max_width, original_infer, **window)
        return original_infer(eye, mask, *args, **kwargs)

    model.infer = infer
    model._nt_lowres_wrapped = True
    model._nt_lowres_original = original_infer


def install() -> bool:
    """Patch iw3 in place. Safe to call more than once."""
    global _installed
    if _installed:
        return True
    if os.environ.get("NT_LOWRES_DISABLE"):
        return False

    from iw3.base_inpaint import BaseImageInpaint, BaseVideoInpaint

    def wrap(cls):
        if getattr(cls.infer, "_nt_lowres", False):
            return
        original = cls.infer

        def infer(self, x, depth, *args, **kwargs):
            # Take max_width out of the call so iw3's own pre-resize does
            # nothing, and hand it to the model wrapper instead.
            self._nt_max_width = kwargs.get("max_width", None)
            kwargs["max_width"] = None
            _wrap_model(self)
            return original(self, x, depth, *args, **kwargs)

        infer._nt_lowres = True
        cls.infer = infer

    for cls in (BaseImageInpaint, BaseVideoInpaint):
        wrap(cls)

    _installed = True
    return True
