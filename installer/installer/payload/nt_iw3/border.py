r"""
Make iw3's "Preserve Screen Border" work with forward_inpaint.

The stretched band at the screen edges
--------------------------------------
Before the forward warp, iw3 pads the frame by repeating its first and last
column (`ReplicationPad2d`, `padding_size = width * divergence * 0.01`). Every
pixel the warp pulls in from outside the frame is one of those copies, so the
edge of each eye becomes a single source column smeared sideways -- about
0.4 * divergence percent of the width on each side, 77 px at divergence 16 on a
1920-wide frame. Those pixels count as "defined", so no inpainting mask ever
covers them.

Inpainting that band was measured and is worse than the smear: cropping real
frames and scoring the band against the pixels that were cut off, the model
lost to plain stretching on 18 of 18 frames (-3 to -7 dB). The band is content
from beyond the frame -- there is nothing on that side to inpaint from, and the
model was never trained on a hole that touches the frame edge.

What this does instead
----------------------
iw3 already has the right fix, "Preserve Screen Border": fade the parallax to
zero over a band at the left and right edges, so nothing is ever pulled from
outside the frame. mlbw_l2_inpaint and monobw_inpaint honour it; forward_inpaint
accepts the option and then drops it (`ForwardInpaint*.apply_warp` never passes
it on). This wraps those two methods and applies the same fade, with iw3's own
band width (divergence * 0.75 % of the width), so the checkbox does what it says
for every method.

The fade is applied to the depth, which is exactly equivalent:

    shift = shift_size * (depth - convergence)
    depth' = convergence + (depth - convergence) * w(x),   w: 0 at the edge -> 1

Only active when "Preserve Screen Border" is ticked. NT_BORDER_DISABLE=1 turns
it off.
"""
from __future__ import annotations

import os

import torch


__all__ = ["install", "fade_depth"]

_installed = False


def fade_depth(depth, divergence, convergence, synthetic_view="both"):
    """Pull depth toward the convergence plane over a band at each side edge.

    Same band as iw3's backward warp (`make_input_tensor`):
    divergence * 0.75 % of the width. A single-view warp moves one eye twice as
    far, so its band is twice as wide -- the same thing
    depth_order_bilinear_forward_warp does to the divergence itself."""
    if not torch.is_tensor(depth) or depth.ndim < 2:
        return depth
    H, W = depth.shape[-2:]
    div = float(divergence) * (1 if synthetic_view == "both" else 2)
    # forward_inpaint warps with width_base=False: the shift is a fraction of
    # the LONGER side, so the band is too, expressed here in depth-map pixels
    band = int(round(div * 0.75 * 0.01 * max(H, W)))
    band = min(band, W // 2)
    if band <= 0:
        return depth

    w = torch.ones(W, dtype=depth.dtype, device=depth.device)
    ramp = torch.linspace(0.0, 1.0, band, dtype=depth.dtype, device=depth.device)
    w[:band] = ramp
    w[-band:] = torch.minimum(w[-band:], ramp.flip(0))
    w = w.view(*([1] * (depth.ndim - 1)), W)

    if torch.is_tensor(convergence):
        conv = convergence.to(depth.device, depth.dtype)
        if conv.numel() == 1:
            conv = conv.reshape(())
        else:
            conv = conv.reshape(-1, *([1] * (depth.ndim - 1)))
    else:
        conv = float(convergence)
    return conv + (depth - conv) * w


def _wrap(cls):
    if getattr(cls.apply_warp, "_nt_border", False):
        return
    original = cls.apply_warp

    def apply_warp(self, x, depth, divergence, convergence, synthetic_view,
                   preserve_screen_border=False, *args, **kwargs):
        if preserve_screen_border:
            depth = fade_depth(depth, divergence, convergence, synthetic_view)
        return original(self, x, depth, divergence, convergence, synthetic_view,
                        preserve_screen_border, *args, **kwargs)

    apply_warp._nt_border = True
    cls.apply_warp = apply_warp


def install() -> bool:
    """Patch iw3 in place. Safe to call more than once."""
    global _installed
    if _installed:
        return True
    if os.environ.get("NT_BORDER_DISABLE"):
        return False
    from iw3.forward_inpaint import ForwardInpaintImage, ForwardInpaintVideo
    for cls in (ForwardInpaintImage, ForwardInpaintVideo):
        _wrap(cls)
    _installed = True
    return True
