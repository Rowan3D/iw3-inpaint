r"""
Give iw3 the mask our models were trained on.

The problem, measured rather than assumed
-----------------------------------------
iw3 builds the inpaint mask in `forward_warp.gen_mask2()`, which marks only the
pixels the forward splat left *completely unwritten* (-1), plus whatever
`fix_layered_holes()` flagged (-2). On real footage the splat almost never
leaves a pixel unwritten: at a depth edge it **stretches** the background across
the disocclusion instead. Those pixels are written, so they are not in the mask.

On one frame of the eval set, at the same disparity as divergence 11.5 at 1920:

    iw3's mask            0.12% of the frame, bands 1px wide (p90 3px)
    our training mask    16.18% of the frame, bands 11px wide (p90 25px)

and the overlap between them is 0.3%. So the model is handed a task it was never
trained on -- a few 1px specks -- while the part that actually looks wrong, the
stretched band, is never marked. Every inpaint model therefore produces the same
picture, which is exactly what it looks like in practice.

Widening iw3's mask does not fix it (outer dilation 24 reaches 2.1% coverage):
the specks are not where the damage is.

What this does
--------------
The warp carries a second channel: for every target pixel, the source x it came
from. That index is the damage detector, in target coordinates, for free:

    * undamaged  -- the index advances by 1 per pixel
    * stretched  -- it advances by less than 1 (one source pixel covers several)
    * a filled hole -- it does not advance at all

So a run of pixels whose index gradient is below 1 *is* the disocclusion band.
No depth pipeline, no second warp, no geometry of our own that could drift from
what iw3 actually did.

`fix_layered_holes()` is the one function that sees both the eye and its index,
and it runs immediately before `gen_mask2()`. Patching it to also mark stretched
runs as -2 puts our band into the mask that iw3 itself builds, and every path
downstream -- image, video, both eyes, the max_width branch -- picks it up
unchanged.

Safety
------
The patch is inert unless one of our `inpaint.nt_*` models is the one being run,
so selecting a stock model in the same session behaves exactly as before. It is
also skipped for any model whose own training masks came from `gen_mask2` --
which is every model but ours.

Set NTRAINER_NO_MASK_PATCH=1 to disable it, NTRAINER_MASK_DEBUG=1 to print what
it marks on the first few frames.
"""
from __future__ import annotations

import os
import threading

import torch
import torch.nn.functional as F


__all__ = ["install", "stretch_mask", "MaskParams"]


def _env(name, default, cast=float):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return cast(value)
    except ValueError:
        return default


class MaskParams:
    """Defaults measured against the training-mask band distribution.

    On a real eval frame at the same disparity as divergence 11.5 at 1920, the
    training masks are median 11px wide (p90 21, max 54). These settings put the
    patched mask at median 11px (p90 22, max 44) -- the same job the model was
    trained to do -- while iw3's own mask is median 1px (p90 3).

    max_grad is the meaningful knob: the stretch threshold. 0.4 marks a pixel
    once the warp stretched its source 2.5x or more -- once well over half of
    what you see there was invented by resampling. Measured on the frame above:
    0.30 and 0.40 remove the tearing and keep the field's texture, 0.55 starts
    flattening it. Raise it to repaint more, lower it to be conservative.
    """

    # a target pixel whose index advances by less than this came from a stretched
    # source pixel. 1.0 would mark every rounding wobble; the band we care about
    # is where several target pixels share one source pixel.
    max_grad = _env("NTRAINER_MASK_GRAD", 0.4)
    # ignore runs narrower than this: 1px stretch is resampling, not damage.
    min_run = _env("NTRAINER_MASK_RUN", 3, int)
    # the training masks include the depth ramp either side of the tear; the
    # index gradient alone stops at the geometric edge.
    grow = _env("NTRAINER_MASK_GROW", 2, int)


_state = threading.local()
_installed = False


def _active() -> bool:
    return getattr(_state, "on", False)


def _run_filter(mask: torch.Tensor, min_run: int) -> torch.Tensor:
    """Drop horizontal runs shorter than min_run (erode then dilate along x)."""
    if min_run <= 1:
        return mask
    k = min_run
    pad = k // 2
    f = mask.float()
    eroded = -F.max_pool2d(-f, (1, k), stride=1, padding=(0, pad))
    kept = F.max_pool2d(eroded, (1, k), stride=1, padding=(0, pad))
    if kept.shape[-1] != mask.shape[-1]:                     # even k loses a column
        kept = kept[..., :mask.shape[-1]]
    return (kept > 0) & mask


def _grow(mask: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 0:
        return mask
    k = 2 * n + 1
    return F.max_pool2d(mask.float(), (1, k), stride=1, padding=(0, n)) > 0


def stretch_mask(index: torch.Tensor, params: type[MaskParams] = MaskParams) -> torch.Tensor:
    """Target pixels that were stretched or invented by the warp.

    `index` is (B, 1, H, W): the source x each target pixel was taken from.
    """
    index = index.float()
    grad = index[..., 1:] - index[..., :-1]
    stretched = grad < params.max_grad
    # the gradient at x describes the step from x-1 to x; mark both endpoints so
    # a run is not one pixel short at its left edge.
    left = F.pad(stretched, (1, 0), value=False)
    right = F.pad(stretched, (0, 1), value=False)
    mask = left | right
    # a jump backwards is an occlusion boundary, not a stretch -- the pixels are
    # real, they just come from a different surface.
    mask &= ~F.pad(grad < -1.0, (1, 0), value=False)
    mask = _run_filter(mask, params.min_run)
    mask = _grow(mask, params.grow)
    return mask


def _is_ours(model) -> bool:
    name = getattr(model, "name", "") or ""
    if not name:
        inner = getattr(model, "_orig_mod", None)      # torch.compile wrapper
        name = getattr(inner, "name", "") or ""
    return name.startswith("inpaint.nt_")


def install() -> bool:
    """Patch iw3 in place. Safe to call more than once."""
    global _installed
    if _installed:
        return True
    if os.environ.get("NTRAINER_NO_MASK_PATCH"):
        return False

    import iw3.forward_warp as forward_warp
    from iw3.forward_inpaint import ForwardInpaintImage, ForwardInpaintVideo

    debug = bool(os.environ.get("NTRAINER_MASK_DEBUG"))
    seen = [0]

    original_fix = forward_warp.fix_layered_holes

    def fix_layered_holes(side_image, index_image, sign, max_tries=100):
        original_fix(side_image, index_image, sign, max_tries=max_tries)
        if not _active():
            return
        mask = stretch_mask(index_image)
        if debug and seen[0] < 4:
            seen[0] += 1
            before = (side_image[:, 0:1] < 0).float().mean().item() * 100
            print(f"note   : nt mask -- iw3 marked {before:.3f}%, "
                  f"stretch adds {mask.float().mean().item() * 100:.2f}%", flush=True)
        side_image[mask.expand_as(side_image)] = -2

    forward_warp.fix_layered_holes = fix_layered_holes

    def wrap(cls):
        original_apply = cls.apply_warp

        def apply_warp(self, *args, **kwargs):
            previous = getattr(_state, "on", False)
            _state.on = _is_ours(self.model)
            try:
                return original_apply(self, *args, **kwargs)
            finally:
                _state.on = previous

        apply_warp.__name__ = "apply_warp"
        apply_warp._ntrainer_wrapped = True
        cls.apply_warp = apply_warp

    for cls in (ForwardInpaintImage, ForwardInpaintVideo):
        if not getattr(cls.apply_warp, "_ntrainer_wrapped", False):
            wrap(cls)

    _installed = True
    return True
