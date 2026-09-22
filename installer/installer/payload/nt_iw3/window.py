r"""
Run each window's per-frame work once instead of twice.

The cost this removes
---------------------
iw3 runs the video inpaint on a sliding window of `model_seq` = 12 frames and
keeps only the middle 6 (`pre_padding` = `post_padding` = 3): the outer frames
are context, their output is thrown away, and the window then advances by 6. So
every frame goes through the whole network twice -- once as the frame being
inpainted, once as its neighbour's context. That is the 2x in
`model 15 ms/frame (30 per emitted frame)`.

Most of that second pass is not needed, because most of the network never looks
at another frame:

    stem, enc1, down1, the 1/8 blocks below the first temporal block
        ... a PER-FRAME function -> cache it, the next window reuses it
    [the temporal blocks, and everything between them]
        ... needs the whole window -- see below
    dec2, dec1, to_image, and the full-resolution put-back in lowres.py
        ... a PER-FRAME function -> run it for the 6 frames we keep

Measured at 1280x768 on a 4090: 26% of the network is above the first temporal
block, 46% between the mixes, 28% below the last one. Halving the first and the
last gives ~30% off the model's time per emitted frame end to end (30 -> 21 ms
at `Inpaint Max Width` 1280, 57 -> 43 at 1920) and a third off the VRAM. The
pictures are the same to within one 8-bit step -- the decoder runs at batch 6
instead of 12, which is a different reduction order, not a different sum.

Why the middle cannot be shared
-------------------------------
A temporal block mixes the window with `proj_spatial`, a learned
`seq_len x seq_len` matrix, so the output at slot p is a weighted sum over the
whole window with the weights of THAT slot. A frame sitting at slot p in one
window and at slot p+-6 in the next therefore has two different values, and both
are needed: one becomes its own output, the other is what its neighbours mix
against. No cache can remove that; only a smaller `--inpaint-pre/post-padding`
can, and that is a quality setting, not a free one.

How it is applied
-----------------
No file in iw3 is edited. Three wrappers, in iw3's own terms:

  * `FrameQueue.add/fill/remove/clear` -- carry a list of frame ids in lockstep
    with the queue's buffers, so "the same id" always means "the same picture"
    (the end padding repeats a frame, and gets that frame's id).
  * `BaseVideoInpaint.forward` / `_inpaint_single` -- note which slice this call
    is about to keep, and which eye it is.
  * the model's `infer` -- hand it that slice and those ids, then scatter the
    frames it returns back into a full-length tensor. iw3's own slicing, and
    every other line of it, is untouched.

It does nothing unless the loaded model advertises `WINDOW_CAPABLE` (the
nt_inpaint_v2 pair does), so the stock iw3 models are unaffected, and it falls
back to the plain call if anything in the chain ignores the new arguments.

NT_WINDOW_DISABLE=1 turns it off.
"""
from __future__ import annotations

import itertools
import os

import torch


__all__ = ["install"]

_installed = False
_FRAME_ID = itertools.count()     # never resets: see _wrap_queue


def _keep(component, n):
    """The frame positions this call will keep -- exactly the slice
    `BaseVideoInpaint.forward` is about to take. None when it keeps everything,
    which is when there is nothing to save."""
    pre = int(getattr(component, "pre_padding", 0) or 0)
    post = int(getattr(component, "post_padding", 0) or 0)
    flush = bool(getattr(component, "_nt_flush", False))
    keep = list(range(pre, n)) if (flush or post == 0) else list(range(pre, n - post))
    if len(keep) >= n or not keep:
        return None
    return keep


def _wrap_model(model):
    """Wrap this model's `infer` once, in place. The window it should use is
    read from the model at call time (`_nt_window`), never captured: iw3 builds
    a fresh component for every run and keeps the model, so a wrapper holding on
    to one component would go on reading a frame queue nobody fills."""
    if model is None or getattr(model, "_nt_window_wrapped", False):
        return
    if not getattr(model, "WINDOW_CAPABLE", False):
        return
    original_infer = model.infer

    def infer(eye, mask, *args, **kwargs):
        window = getattr(model, "_nt_window", None)
        if window is None or not torch.is_tensor(eye) or "keep" in kwargs:
            return original_infer(eye, mask, *args, **kwargs)
        keep, ids, is_left = window
        if keep is None or len(ids) != eye.shape[0]:
            return original_infer(eye, mask, *args, **kwargs)

        # One cache per eye: the left eye is mirrored before it reaches the
        # model, so the two eyes of a frame are two different pictures.
        caches = model.__dict__.setdefault("_nt_caches", {})
        cache = caches.setdefault(bool(is_left), {})
        oldest = min(ids)
        for c in caches.values():
            for fid in [f for f in c if f < oldest]:
                del c[fid]          # the queue can never ask for it again

        try:
            out = original_infer(eye, mask, *args, keep=keep, ids=ids, cache=cache, **kwargs)
        except TypeError:
            # something in the chain does not take a window (an older lowres
            # patch, a wrapper of our own that got in first): no harm, no gain
            return original_infer(eye, mask, *args, **kwargs)
        if out.shape[0] != len(keep):
            return out              # it ignored the window; it is already whole
        # Give iw3 back the length it handed us, so its own slicing -- and the
        # frame queue's arithmetic -- carry on untouched. The positions we did
        # not compute are context, and are dropped by the caller: `keep` is the
        # same slice `BaseVideoInpaint.forward` is about to take, so nothing
        # reads them. empty_like rather than clone for that reason -- at 4K the
        # copy it skips is a gigabyte of traffic per eye per window.
        full = torch.empty_like(eye)
        full[keep] = out.to(full.dtype)
        return full

    model.infer = infer
    model._nt_window_wrapped = True
    model._nt_window_original = original_infer


def _wrap_queue(FrameQueue):
    """Frame ids that follow the queue's buffers exactly.

    The counter is global and never resets: a new queue means a new clip, and
    an id that came round again would let its first frames read the PREVIOUS
    clip's cached prefix -- the same picture in the same slot, and wrong."""
    if getattr(FrameQueue.add, "_nt_window", False):
        return
    orig_init, orig_add = FrameQueue.__init__, FrameQueue.add
    orig_fill, orig_remove, orig_clear = FrameQueue.fill, FrameQueue.remove, FrameQueue.clear

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.nt_ids = []

    def add(self, *args, **kwargs):
        orig_add(self, *args, **kwargs)
        self.nt_ids.append(next(_FRAME_ID))

    def fill(self):
        n = len(self.nt_ids)
        pad = orig_fill(self)
        if pad:
            # fill() repeats the last frame: those copies ARE that frame
            self.nt_ids[n:] = [self.nt_ids[n - 1]] * pad
        return pad

    def remove(self, n):
        orig_remove(self, n)
        del self.nt_ids[:n]

    def clear(self):
        orig_clear(self)
        self.nt_ids.clear()

    add._nt_window = True
    FrameQueue.__init__, FrameQueue.add, FrameQueue.fill = __init__, add, fill
    FrameQueue.remove, FrameQueue.clear = remove, clear


def install() -> bool:
    """Patch iw3 in place. Safe to call more than once."""
    global _installed
    if _installed:
        return True
    if os.environ.get("NT_WINDOW_DISABLE"):
        return False

    from iw3.base_inpaint import FrameQueue, BaseVideoInpaint

    _wrap_queue(FrameQueue)

    if not getattr(BaseVideoInpaint.forward, "_nt_window", False):
        orig_forward = BaseVideoInpaint.forward

        def forward(self, flush=False, *args, **kwargs):
            # the last window of a clip keeps its tail as well
            self._nt_flush = bool(flush)
            try:
                return orig_forward(self, flush, *args, **kwargs)
            finally:
                self._nt_flush = False

        forward._nt_window = True
        BaseVideoInpaint.forward = forward

    if not getattr(BaseVideoInpaint._inpaint_single, "_nt_window", False):
        orig_single = BaseVideoInpaint._inpaint_single

        def _inpaint_single(self, eye, mask, is_left, *args, **kwargs):
            model = getattr(self, "model", None)
            if getattr(model, "WINDOW_CAPABLE", False) and torch.is_tensor(eye):
                queue = getattr(self, "frame_queue", None)
                model._nt_window = (_keep(self, eye.shape[0]),
                                    list(getattr(queue, "nt_ids", ()) or ()),
                                    bool(is_left))
                _wrap_model(model)
            try:
                return orig_single(self, eye, mask, is_left, *args, **kwargs)
            finally:
                if model is not None:
                    model._nt_window = None

        _inpaint_single._nt_window = True
        BaseVideoInpaint._inpaint_single = _inpaint_single

    _installed = True
    return True
