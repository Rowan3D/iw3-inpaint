"""
ntrainer.depth_refine -- make depth edges land on *object* edges.

Monocular depth models (DA / DA3 / ZoeD) put a soft 3-10px ramp across every
depth discontinuity.  Any warp turns that ramp into a smeared, misplaced hole
boundary: the mask edge ends up somewhere inside the ramp instead of on the
silhouette, and the ramp itself gets stretched into a streaked halo.  That is
the second half of the "masks aren't aligned to the object edge" problem (the
first half being iw3's iterative warp -- see ntrainer.geometry).

Two cheap local steps fix it:

  1. `joint_bilateral` -- re-filter depth using *colour* similarity from the
     RGB image as the weight.  Depth stops crossing a colour boundary, so the
     transition snaps onto the silhouette.  Unlike a guided filter this cannot
     inject the guide's texture into the depth: in a flat-depth region every
     neighbour carries the same depth, so any weighting returns that depth.
  2. `toggle_contrast` -- morphological toggle (each pixel snaps to whichever
     of the local min / local max it is closer to).  Collapses what is left of
     the ramp into a step, so the gap the warp computes is generated at a
     single column instead of smeared over several.  A no-op on flat depth
     (local min == local max).

Everything is shifted-tensor / max-pool arithmetic: no iteration caps, no
per-pixel Python, ~10-30 ms at 1920x1080 on a modern GPU.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


__all__ = ["box_mean", "guided_filter", "joint_bilateral", "toggle_contrast",
           "depth_edge_mask", "refine_depth"]


def box_mean(x, radius):
    k = radius * 2 + 1
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=radius, count_include_pad=False)


# --------------------------------------------------------------------------- #
def joint_bilateral(guide, src, radius=4, sigma_color=0.06, sigma_space=None,
                    n_iter=1, separable=True):
    """
    Colour-guided bilateral filter of `src` -- i.e. joint bilateral upsampling
    when `src` is a low-resolution depth map that has been stretched to the
    guide's size, which is exactly the situation here: iw3 runs DA3 at
    prep_lower_bound=392 and interpolates the result up to full resolution, so
    every depth pixel is a ~3px block and the silhouette is a staircase with a
    multi-pixel ramp before any mask code sees it.

    separable=True runs a 1D horizontal pass then a 1D vertical pass:
    2*(2r+1) taps instead of (2r+1)^2, so radius 6 costs 26 shifted-tensor ops
    instead of 169.  Not mathematically identical to the 2D filter, but for
    snapping a depth edge onto a colour edge the difference is invisible and it
    is ~6x faster, which matters over a whole dataset.

    guide: (B,3,H,W) or (B,1,H,W) in [0,1]
    src:   (B,1,H,W)
    """
    guide = guide.float()
    src = src.float()
    if guide.shape[-2:] != src.shape[-2:]:
        guide = F.interpolate(guide, size=src.shape[-2:], mode="bilinear",
                              align_corners=False, antialias=True)
    if sigma_space is None:
        sigma_space = max(radius, 1) / 2.0

    inv_c = 1.0 / (2.0 * sigma_color * sigma_color)
    inv_s = 1.0 / (2.0 * sigma_space * sigma_space)
    r = int(radius)

    H, W = src.shape[-2:]

    def _pass(cur, offsets):
        num = torch.zeros_like(cur)
        den = torch.zeros_like(cur)
        for dy, dx in offsets:
            gp = F.pad(guide, (r, r, r, r), mode="replicate")
            sp = F.pad(cur, (r, r, r, r), mode="replicate")
            ys, xs = dy + r, dx + r
            g_sh = gp[..., ys:ys + H, xs:xs + W]
            s_sh = sp[..., ys:ys + H, xs:xs + W]
            diff = (g_sh - guide).pow(2).sum(dim=1, keepdim=True)
            w = torch.exp(-diff * inv_c - (dx * dx + dy * dy) * inv_s)
            num = num + w * s_sh
            den = den + w
        return num / den.clamp_min(1e-8)

    out = src
    for _ in range(max(1, int(n_iter))):
        if separable:
            out = _pass(out, [(0, dx) for dx in range(-r, r + 1)])
            out = _pass(out, [(dy, 0) for dy in range(-r, r + 1)])
        else:
            out = _pass(out, [(dy, dx) for dy in range(-r, r + 1)
                              for dx in range(-r, r + 1)])
    return out


# --------------------------------------------------------------------------- #
def guided_filter(guide, src, radius=8, eps=1e-4, color=True):
    """He et al. guided filter. Faster than the bilateral but its linear model
    can leak guide texture into flat-depth regions -- prefer joint_bilateral
    for depth unless you know the background is smooth."""
    guide = guide.float()
    src = src.float()
    if guide.shape[-2:] != src.shape[-2:]:
        guide = F.interpolate(guide, size=src.shape[-2:], mode="bilinear",
                              align_corners=False, antialias=True)

    if (not color) or guide.shape[1] == 1:
        I = guide if guide.shape[1] == 1 else guide.mean(dim=1, keepdim=True)
        mI, mp = box_mean(I, radius), box_mean(src, radius)
        cov = box_mean(I * src, radius) - mI * mp
        var = box_mean(I * I, radius) - mI * mI
        a = cov / (var + eps)
        b = mp - a * mI
        return box_mean(a, radius) * I + box_mean(b, radius)

    r, g, b_ = guide[:, 0:1], guide[:, 1:2], guide[:, 2:3]
    mr, mg, mb = box_mean(r, radius), box_mean(g, radius), box_mean(b_, radius)
    mp = box_mean(src, radius)
    crp = box_mean(r * src, radius) - mr * mp
    cgp = box_mean(g * src, radius) - mg * mp
    cbp = box_mean(b_ * src, radius) - mb * mp
    vrr = box_mean(r * r, radius) - mr * mr + eps
    vrg = box_mean(r * g, radius) - mr * mg
    vrb = box_mean(r * b_, radius) - mr * mb
    vgg = box_mean(g * g, radius) - mg * mg + eps
    vgb = box_mean(g * b_, radius) - mg * mb
    vbb = box_mean(b_ * b_, radius) - mb * mb + eps
    c00 = vgg * vbb - vgb * vgb
    c01 = vgb * vrb - vrg * vbb
    c02 = vrg * vgb - vgg * vrb
    c11 = vrr * vbb - vrb * vrb
    c12 = vrb * vrg - vrr * vgb
    c22 = vrr * vgg - vrg * vrg
    det = vrr * c00 + vrg * c01 + vrb * c02
    det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
    ar = (c00 * crp + c01 * cgp + c02 * cbp) / det
    ag = (c01 * crp + c11 * cgp + c12 * cbp) / det
    ab = (c02 * crp + c12 * cgp + c22 * cbp) / det
    bb = mp - ar * mr - ag * mg - ab * mb
    return (box_mean(ar, radius) * r + box_mean(ag, radius) * g
            + box_mean(ab, radius) * b_ + box_mean(bb, radius))


# --------------------------------------------------------------------------- #
def toggle_contrast(depth, kernel_size=3, n_iter=1, min_range=0.01):
    """Snap depth ramps into steps; flat regions untouched."""
    d = depth.float()
    p = kernel_size // 2
    for _ in range(int(n_iter)):
        mx = F.max_pool2d(d, kernel_size, stride=1, padding=p)
        mn = -F.max_pool2d(-d, kernel_size, stride=1, padding=p)
        snapped = torch.where((d - mn) < (mx - d), mn, mx)
        if min_range > 0.0:
            snapped = torch.where((mx - mn) >= min_range, snapped, d)
        d = snapped
    return d


def depth_edge_mask(depth, threshold=0.02, radius=4):
    """1 near a depth discontinuity, 0 elsewhere (soft, dilated)."""
    mx = F.max_pool2d(depth, 3, stride=1, padding=1)
    mn = -F.max_pool2d(-depth, 3, stride=1, padding=1)
    e = ((mx - mn) >= threshold).float()
    k = radius * 2 + 1
    e = F.max_pool2d(e, k, stride=1, padding=radius)
    return e


# --------------------------------------------------------------------------- #
def refine_depth(rgb, depth, *, method="bilateral", radius=6, sigma_color=0.08,
                 n_iter=2, separable=True, guided_radius=8, guided_eps=1e-4,
                 snap_iter=0, snap_min_range=0.01,
                 edge_only=True, edge_threshold=0.02, renormalize=False):
    """
    Full edge-alignment pass.  Input/output depth is normalised disparity
    in [0,1] (1 == near).

    method:      "bilateral" (default, safe on textured backgrounds),
                 "guided", or "none".
    edge_only:   restrict the whole refinement to a band around depth
                 discontinuities, so flat regions are bit-exact unchanged.
    snap_iter:   toggle-contrast passes (0 to disable).
    renormalize: re-stretch to [0,1] afterwards.  Off by default -- it would
                 change the divergence scale relative to the raw depth.
    """
    d0 = depth.float()
    d = d0
    if method == "bilateral":
        d = joint_bilateral(rgb, d, radius=radius, sigma_color=sigma_color,
                            n_iter=n_iter, separable=separable)
    elif method == "guided":
        d = guided_filter(rgb, d, radius=guided_radius, eps=guided_eps, color=True)
    elif method not in ("none", None):
        raise ValueError(f"unknown method {method!r}")

    if snap_iter and snap_iter > 0:
        d = toggle_contrast(d, kernel_size=3, n_iter=snap_iter, min_range=snap_min_range)

    if edge_only:
        e = depth_edge_mask(d0, threshold=edge_threshold, radius=max(radius, 3) + 2)
        d = d * e + d0 * (1.0 - e)

    if renormalize:
        lo = d.amin(dim=[1, 2, 3], keepdim=True)
        hi = d.amax(dim=[1, 2, 3], keepdim=True)
        d = (d - lo) / (hi - lo + 1e-8)
    return d.clamp(0.0, 1.0)
