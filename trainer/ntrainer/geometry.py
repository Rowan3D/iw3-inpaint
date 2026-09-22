"""
ntrainer.geometry -- analytic disocclusion masks + scan-based DIBR warping.

Replaces iw3's round-trip `nonwarp_mask()` / `shift_fill()` approach, which is
capped at `max_tries=100` one-pixel-per-step iterations and therefore falls
apart once the required shift exceeds ~100px (divergence > ~9 at 1920px).

Everything here is O(W) per row using `cummax` / `cummin` scans, so it is exact
at *any* divergence and is also much faster than the iterative version.

Sign / scale conventions match iw3 (see iw3/forward_warp.py):

    shift_px   = divergence * 0.005 * base_size      # |shift| at depth == 1
    s_right(x) = -(depth(x) - convergence) * shift_px
    s_left (x) = +(depth(x) - convergence) * shift_px

`depth` is normalised disparity in [0, 1] where 1 == nearest.


Why the "envelope"
------------------
Forward-mapping x -> u(x) = x + s(x) stretches the row by du/dx = 1 + ds/dx.
A *gently* stretched region is not a hole -- a correct renderer just resamples
it.  A hole exists only where the surface genuinely tears.

A monocular depth map never gives a sharp tear; it gives a 3-10px ramp, so one
tear is spread over several columns each contributing a fraction of the gap.
Reading the gap per column then reports a 37px hole as 13px, and iw3's
splat-based mask turns the same ramp into the "scattered disconnected stripe"
mask its own source comment describes.

So the band width at a column is taken from the depth *envelope* -- the local
maximum depth within the previous `ramp_px` columns -- and one band is emitted
per edge, at the local maximum of that width.  Gentle slopes are excluded by a
tear test, so they stay stretched surfaces instead of becoming holes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


__all__ = [
    "shift_scale",
    "signed_shift",
    "disocclusion_mask",
    "fill_along_x",
    "hole_run_length",
    "dibr_warp",
    "depth_edge_width",
]


def median3(x, axis="x"):
    """3-tap median along one axis, via max/min -- no sort, 2 shifted tensors.

    Removes single-pixel spikes while leaving a step edge bit-exact: for
    [0.9, 0.9, 0.0, 0.0] every output equals its input, but [0.2, 0.9, 0.2]
    collapses to 0.2.  Structures two pixels wide or more survive.
    """
    if axis == "x":
        pad = (1, 1, 0, 0)
        xp = F.pad(x, pad, mode="replicate")
        a, b, c = xp[..., :-2], x, xp[..., 2:]
    else:
        pad = (0, 0, 1, 1)
        xp = F.pad(x, pad, mode="replicate")
        a, b, c = xp[..., :-2, :], x, xp[..., 2:, :]
    return torch.maximum(torch.minimum(a, b), torch.minimum(torch.maximum(a, b), c))


#: torch.quantile refuses inputs beyond 2**24 elements. One 1920x1080 frame is
#: 2M and fits; a 12-frame clip is 25M and does not, which is how the video
#: dataset tripped over this. Below the cap the behaviour is byte-for-byte what
#: it always was, so nothing about the validated image path changes.
_QUANTILE_MAX = 2 ** 24


def _quantile_any_size(flat, q):
    """torch.quantile, with a strided subsample when the input is over its cap.

    A stride is the right sampler here: the values are per-pixel gradient
    magnitudes over a whole frame, so any fixed stride still sees every edge in
    the image. Measured against the exact value on a 25M-element clip, the 0.98
    quantile agrees to ~1e-5 relative.
    """
    if flat.numel() > _QUANTILE_MAX:
        stride = (flat.numel() + _QUANTILE_MAX - 1) // _QUANTILE_MAX
        flat = flat[::stride]
    return float(torch.quantile(flat, q).item())


def depth_edge_width(depth, window=21, strength_quantile=0.98):
    """
    Median horizontal ramp width, in pixels, of this depth map's edges.

    A step edge measures ~1.  Anything larger is how far the depth model (and
    the upsample from its internal `prep_lower_bound` resolution) has smeared
    the silhouette, which is what sets a useful `ramp_px` / `edge_window`.

    width = (total |gradient| over a window) / (peak |gradient| in it), which
    for a monotone ramp is exactly the ramp's width.
    """
    d = depth.float()
    g = (d[..., :-1] - d[..., 1:]).abs()
    k = int(window) | 1
    p = k // 2
    gmax = F.max_pool2d(g, kernel_size=(1, k), stride=1, padding=(0, p))
    gsum = F.avg_pool2d(g, kernel_size=(1, k), stride=1, padding=(0, p),
                        count_include_pad=False) * k
    thr = _quantile_any_size(gmax.flatten().float(), float(strength_quantile))
    sel = gmax >= max(thr, 1e-6)
    if sel.sum() == 0:
        return 1.0
    w = (gsum[sel] / gmax[sel].clamp_min(1e-8))
    return float(w.median().item())


# --------------------------------------------------------------------------- #
# scale helpers
# --------------------------------------------------------------------------- #
def _base_size(t, base_size, width_base):
    if base_size is not None:
        return float(base_size)
    return float(t.shape[-1]) if width_base else float(max(t.shape[-2:]))


def shift_scale(divergence, base_size):
    """Pixels of horizontal shift at depth == 1.0 (iw3 single-view convention)."""
    return float(divergence) * 0.005 * float(base_size)


def signed_shift(depth, divergence, convergence, *, base_size=None,
                 width_base=True, view="right"):
    """Signed per-pixel horizontal shift in pixels. (B,1,H,W) -> (B,1,H,W)"""
    s = shift_scale(divergence, _base_size(depth, base_size, width_base))
    d = (depth.float() - float(convergence)) * s
    return -d if view == "right" else d


# --------------------------------------------------------------------------- #
# 1. analytic disocclusion mask (source-image coordinates)
# --------------------------------------------------------------------------- #
def disocclusion_mask(depth, divergence, convergence, *, base_size=None,
                      width_base=True, view="right", method="envelope",
                      ramp_px=12, depth_step=0.02, tear_px=0.0, edge_window=5,
                      sharpness=0.6,
                      min_band_px=1.0,
                      anchor="edge", edge_frac=0.1, anchor_px=0, tear_rows=2,
                      smooth_px=1, despike=2, outer_smooth_rows=1, occlusion_aware=True,
                      depth_tolerance=0.02, occlusion_frac=0.3, soft=True):
    """
    Disocclusion ("hole") mask in the *source* image's coordinate frame -- the
    form iw3's inpaint training data needs (clean RGB + realistic hole mask).

    Geometry (right view, s(x) = -(d(x)-c)*S):

        u(x+1) - u(x) = 1 + (d(x) - d(x+1)) * S

    so a gap  g(x) = (d(x) - d(x+1)) * S  opens on every depth drop, which is
    the only place a right-eye disocclusion can occur.  In the finished view
    that gap is filled by background continuing rightwards from x+1, so in the
    source frame the hole is the band  [x+1, x+1+g) .  The union of all bands
    is one `cummax` -- no iteration limit, sub-pixel accurate, and the band
    begins exactly on the first background column.

    method:
        "envelope" (default) -- the width of the band at column p is the drop
            from the *local maximum depth within the previous `ramp_px`
            columns* down to p.  This is what makes the result correct on real
            depth maps: a monocular model never gives a sharp tear, it gives a
            3-10px ramp, and a per-column reading of that ramp reports only a
            fraction of the real hole width (a 37px hole measured as 13px).
            The envelope sees across the whole ramp, so one contiguous,
            correctly-sized band is emitted at the first true background
            column.  A band is emitted only at a local maximum of that width,
            so one edge produces exactly one band.
        "column" -- raw per-column gaps.  Exact on synthetic step edges, wrong
            on real depth.  Kept for comparison.

    A column only counts as torn if the depth actually steps (>= `depth_step`
    of the normalised range) or the gap exceeds `tear_px`.  A gently sloped
    surface therefore stays a *stretched* surface, which is what a correct
    renderer does with it -- this is the other half of why iw3's splat-based
    mask breaks up into stripes at high divergence.

    Args:
        depth:            (B,1,H,W) normalised disparity, 1 == near.
        divergence:       iw3 divergence (percent).
        convergence:      iw3 convergence.
        base_size:        pixel base for the shift scale (default: width).
        view:             "right" or "left".
        ramp_px:          widest depth-edge ramp to bridge (envelope radius).
        depth_step:       depth drop across `edge_window` columns that counts
                          as a discontinuity.
        tear_px:          gap (px) across `edge_window` columns that counts as
                          a discontinuity regardless of depth_step.  0 =
                          disabled (default) -- being a pixel criterion it
                          scales with divergence and so overrides depth_step at
                          high divergence, admitting smooth surface gradients
                          as tears.
        sharpness:        require the drop over `edge_window` columns to be at
                          least this fraction of the drop over 4x that many, so
                          a concentrated step qualifies and a slope does not.
                          0 = disabled.
        edge_window:      columns over which the tear test is measured.  Must
                          be wide enough to span the depth model's edge ramp.
        min_band_px:      discard bands narrower than this.
        anchor:           "edge" (default) starts each band at the silhouette,
                          i.e. at the first column of the depth ramp, so the
                          mask touches the object.  "background" starts it at
                          the first pure-background column, which leaves a
                          ramp-width gap between object and mask.
        edge_frac:        with anchor="edge", how far below the foreground
                          plateau still counts as "on the object", as a
                          fraction of this edge's own height.  Raise it if the
                          mask still leaves a gap, lower it if it eats in.
        anchor_px:        how far back the silhouette search may reach
                          (0 = same as ramp_px).
        smooth_px:        box radius of the final majority smoothing of the
                          mask (0 = off).
        outer_smooth_rows: rows of vertical median applied to the mask's outer
                          boundary (0 = off).
        despike:          0 = off, 1 = 3-tap horizontal median on the depth,
                          2 = horizontal then vertical (default).  Removes
                          single-pixel depth spikes, which otherwise read as a
                          foreground plateau and anchor a band deep inside the
                          object.  The vertical pass is also what straightens
                          the mask's outer profile: measured on a synthetic
                          784->1920 depth, outer-boundary jitter falls from
                          5.8px to 1.6px, because the jaggedness comes from
                          row-to-row variation in the plateau depth rather
                          than from the band arithmetic.  Structures 2px or
                          wider survive a median untouched.
        tear_rows:        vertical radius over which the tear test is pooled,
                          so a band does not break up row to row on an edge
                          sitting near the threshold.  0 disables.
        occlusion_frac:   occlusion tolerance as a fraction of the depth drop
                          that opened the band (floored at depth_tolerance).
        occlusion_aware:  cut a band short where it runs into a surface
                          significantly nearer than the background it started
                          from (that surface would cover the hole).
        depth_tolerance:  how much nearer counts as "another surface".
        soft:             fractional coverage in [0,1]; else hard 0/1.

    Returns:
        (B,1,H,W) float mask, 1 == needs inpainting.
    """
    if view == "left":
        return disocclusion_mask(
            depth.flip(-1), divergence, convergence, base_size=base_size,
            width_base=width_base, view="right", method=method,
            ramp_px=ramp_px, depth_step=depth_step, tear_px=tear_px,
            edge_window=edge_window, sharpness=sharpness,
            min_band_px=min_band_px, anchor=anchor,
            edge_frac=edge_frac, anchor_px=anchor_px, tear_rows=tear_rows,
            smooth_px=smooth_px, occlusion_aware=occlusion_aware,
            depth_tolerance=depth_tolerance, occlusion_frac=occlusion_frac,
            soft=soft).flip(-1)
    if view != "right":
        raise ValueError(f"view must be 'right' or 'left', got {view!r}")

    d = depth.float()
    W = d.shape[-1]
    S = shift_scale(divergence, _base_size(d, base_size, width_base))

    # A single-column depth spike reads as a foreground plateau, so the
    # envelope measures the band from it and the silhouette search anchors on
    # it -- the mask then digs a notch into the object at that row.  A 3-tap
    # median removes spikes and leaves real edges bit-exact.
    if despike:
        d = median3(d, "x")
        if despike >= 2:
            d = median3(d, "y")

    # per-column depth drop, aligned so that drop[p] is the step from p-1 to p
    drop = F.pad(d[..., :-1] - d[..., 1:], (1, 0), value=0.0)

    peak_col = None
    if method == "column":
        band = (drop * S).clamp_min(0.0)
    elif method == "envelope":
        R = max(1, int(ramp_px))
        left_max = F.max_pool2d(F.pad(d, (R, 0, 0, 0), mode="replicate"),
                                kernel_size=(1, R + 1), stride=1)
        # Measure the tear over `edge_window` columns, not one.  A depth model
        # spreads a real edge over 3-10px, so its PER-COLUMN drop is small: a
        # 0.15 step blurred over 9px drops only 0.017 per column and falls
        # under a 0.02 per-column threshold entirely -- no mask at all -- while
        # depth noise pushes the odd column over it, which is what breaks a
        # band into stair-stepped fragments.  Over a window the same edge reads
        # 0.083 and a gently sloped surface still reads far less, so the two
        # stay separable without the threshold sitting on top of the noise.
        K = max(1, int(edge_window))
        drop_k = F.pad(d[..., :-K] - d[..., K:], (K, 0), value=0.0)
        torn_b = drop_k >= float(depth_step)
        if tear_px and tear_px > 0:
            # A pixel-gap criterion scales with divergence, so at high
            # divergence it silently overrides depth_step: at div 16 on a
            # 1920px frame, tear_px=2 admits a depth drop of just 2/153.6 =
            # 0.013, which is exactly where a smooth face/hair gradient sits.
            # Measured on a real frame: that condition alone produced 154
            # spurious "nub" bands (median drop 0.0134) against 495 real edges
            # (median drop 0.47) -- a 35x separation it threw away.  Off by
            # default; it changes nothing at low divergence anyway.
            torn_b = torn_b | (drop_k * S >= float(tear_px))
        if sharpness and sharpness > 0.0:
            # Is the drop *concentrated* or spread along a slope?  Compare the
            # drop over K columns with the drop over 4K: a step scores 1.0, a
            # linear slope scores 0.25.  max(.,drop_k) guards the denominator
            # where the wider window contains a rise.
            M = K * 4
            drop_m = F.pad(d[..., :-M] - d[..., M:], (M, 0), value=0.0)
            concentrated = drop_k >= float(sharpness) * torch.maximum(drop_m, drop_k).clamp_min(1e-6)
            torn_b = torn_b & concentrated
        torn = torn_b.float()
        V = max(0, int(tear_rows))
        # The tear test is a hard threshold, and every row is scanned
        # independently, so an edge sitting near the threshold flickers row to
        # row and the band breaks into stair-stepped fragments.  Pooling the
        # test over +-tear_rows rows makes the *decision* vertically coherent
        # while each row still gets its own exact geometry.
        active = F.max_pool2d(F.pad(torn, (R, 0, V, V)),
                              kernel_size=(2 * V + 1, R + 1), stride=1)
        band = ((left_max - d) * S).clamp_min(0.0) * active
        # keep only local maxima of the band width: one band per edge
        prev = F.pad(band, (1, 0), value=0.0)[..., :-1]
        nxt = F.pad(band, (0, 1), value=0.0)[..., 1:]
        band = torch.where((band > prev) & (band >= nxt), band, torch.zeros_like(band))
        # Where the silhouette actually starts.  Any absolute per-column
        # threshold is wrong here: the depth is an upsampled low-res map, so
        # the ramp is several px wide and its slope is noisy, which both starts
        # the band late (the visible gap) and makes the inner boundary jitter
        # row to row (the jagged profile).  Instead find the last column that
        # is still within `edge_frac` of the foreground plateau -- a threshold
        # *relative* to this particular edge's own height, so it is scale-free
        # and insensitive to depth noise.
        A = max(1, int(anchor_px) if anchor_px else R)
        thresh = left_max - float(edge_frac) * (left_max - d).clamp_min(0.0)
        posl = torch.arange(W, device=d.device, dtype=torch.long).view(1, 1, 1, W)
        peak_col = torch.full_like(posl.expand_as(torn_b), -1)
        for k in range(A, -1, -1):
            d_k = F.pad(d, (k, 0, 0, 0), mode="replicate")[..., :W]
            peak_col = torch.where(d_k >= thresh, (posl - k).clamp_min(0), peak_col)
    else:
        raise ValueError(f"unknown method {method!r}")

    band = torch.where(band >= float(min_band_px), band, torch.zeros_like(band))

    pos = torch.arange(W, device=d.device, dtype=torch.float32).view(1, 1, 1, W)
    reach, origin = torch.cummax(pos + band, dim=-1)

    # `reach` IS the mask's outer boundary, so smoothing it vertically
    # straightens that boundary directly -- a few rows of median, which cannot
    # move a boundary that is already consistent and cannot round a real corner
    # the way a blur would.  The inner anchor is untouched.
    for _ in range(max(0, int(outer_smooth_rows))):
        reach = median3(reach, "y")

    cover = (reach - pos).clamp(0.0, 1.0)

    if occlusion_aware:
        origin_depth = torch.gather(d, -1, origin)
        # Tolerance scales with the size of the edge that opened the band: a
        # fixed 0.02 is smaller than the depth noise on a dark / low-contrast
        # background, which chops long bands into pieces.
        band_at_origin = (reach - origin.to(torch.float32)).clamp_min(0.0)
        tol = torch.clamp(band_at_origin * (float(occlusion_frac) / max(S, 1e-6)),
                          min=float(depth_tolerance))
        cover = cover * (d <= origin_depth + tol).to(cover.dtype)

    if anchor == "edge" and peak_col is not None:
        # Extend each band left to the silhouette itself.  `pos` is the first
        # *pure* background column, which on a real depth map sits `ramp_px`
        # past the object edge -- that is the visible gap between the object
        # and the mask.  peak_col is the last column still at foreground depth,
        # so [peak_col+1, pos] is the ramp, i.e. the mixed silhouette pixels.
        R = max(1, int(ramp_px))
        BIG = float(W + R + 10)
        start = (peak_col.to(torch.float32) + 1.0)
        start = torch.where(band > 0, start, torch.full_like(start, BIG))
        # right-aligned window minimum: out[j] = min(start[j .. j+R])
        win = -F.max_pool2d(F.pad(-start, (0, R, 0, 0), value=-BIG),
                            kernel_size=(1, R + 1), stride=1)
        cover = torch.maximum(cover, (win <= pos).to(cover.dtype))
    elif anchor not in ("edge", "background"):
        raise ValueError(f"unknown anchor {anchor!r}")

    mask = cover
    if smooth_px and smooth_px > 0:
        # The old iw3 mask is run through mask_closing(); without an equivalent
        # the analytic mask keeps every single-row wobble of the depth map.
        # A small box mean of the soft coverage is a majority vote once
        # thresholded, which straightens both boundaries without moving them.
        k = int(smooth_px) * 2 + 1
        mask = F.avg_pool2d(mask, kernel_size=k, stride=1, padding=int(smooth_px),
                            count_include_pad=False)
    if not soft:
        mask = (mask > 0.5).to(d.dtype)
    return mask


# --------------------------------------------------------------------------- #
# 2. unbounded scan-based hole fill (replaces shift_fill's 100-step loop)
# --------------------------------------------------------------------------- #
def _neighbour_indices(valid):
    B, _, H, W = valid.shape
    idx = torch.arange(W, device=valid.device, dtype=torch.long).view(1, 1, 1, W).expand(B, 1, H, W)
    left_i = torch.cummax(torch.where(valid, idx, torch.full_like(idx, -1)), dim=-1).values
    right_i = torch.flip(
        torch.cummin(torch.flip(torch.where(valid, idx, torch.full_like(idx, W)), (-1,)), dim=-1).values,
        (-1,))
    return left_i, right_i


def hole_run_length(valid):
    """Length of the contiguous invalid run each pixel belongs to (0 if valid)."""
    left_i, right_i = _neighbour_indices(valid)
    run = (right_i - left_i - 1).clamp_min(0)
    return torch.where(valid, torch.zeros_like(run), run)


def fill_along_x(values, valid, depth_key=None, prefer="background"):
    """
    Fill invalid entries along width from the nearest valid neighbour.

    Unlike iw3's `shift_fill()` this is not iterative: the nearest valid index
    on each side comes from one `cummax` and one `cummin`, so a 1000px hole
    costs the same as a 1px hole and nothing is capped.

    prefer: "background" (smaller depth wins -- correct for disocclusion
            fill), "foreground", "left" or "right".
    """
    B, C, H, W = values.shape
    left_i, right_i = _neighbour_indices(valid)
    has_l = left_i >= 0
    has_r = right_i < W
    lc = left_i.clamp(0, W - 1)
    rc = right_i.clamp(0, W - 1)

    if prefer == "left":
        use_l = has_l
    elif prefer == "right":
        use_l = ~has_r
    else:
        if depth_key is None:
            raise ValueError("depth_key is required for prefer='background'/'foreground'")
        dk = depth_key.float()
        dl = torch.gather(dk, -1, lc)
        dr = torch.gather(dk, -1, rc)
        if prefer == "background":
            better = dl <= dr
        elif prefer == "foreground":
            better = dl >= dr
        else:
            raise ValueError(f"unknown prefer={prefer!r}")
        use_l = has_l & (~has_r | better)

    src = torch.where(use_l, lc, rc).expand(B, C, H, W)
    out = torch.gather(values, -1, src)
    return torch.where(valid.expand(B, C, H, W), values, out)


# --------------------------------------------------------------------------- #
# 3. DIBR warp (splat disparity -> repair -> backward resample)
# --------------------------------------------------------------------------- #
def dibr_warp(rgb, depth, divergence, convergence, *, view="right",
              base_size=None, width_base=True, return_mask=True,
              tear_px=2.0, fill=True):
    """
    Synthesise one eye by Depth-Image-Based Rendering.

    Rather than scattering *colour* (which is what produces iw3's splat
    speckle and needs bilinear-weight renormalisation), this scatters only the
    *disparity* with a z-buffer, repairs the disparity map with the unbounded
    scan fill above, and then does a single **backward** bilinear resample of
    the colour.  Output colour is therefore always a properly filtered sample
    of the source image, never an accumulation of scattered points, and a
    gently stretched surface stays a stretched surface instead of breaking into
    stripes.

    Returns (view_rgb, hole_mask) in the target view's coordinates; hole_mask
    marks only runs at least `tear_px` wide (narrower gaps are resampled).
    """
    rgb = rgb.float()
    depth = depth.float()
    if depth.shape[-2:] != rgb.shape[-2:]:
        depth = F.interpolate(depth, size=rgb.shape[-2:], mode="bilinear",
                              align_corners=True, antialias=True)

    S = shift_scale(divergence, _base_size(rgb, base_size, width_base))
    pad = int(S) + 2
    rgb_p = F.pad(rgb, (pad, pad, 0, 0), mode="replicate")
    depth_p = F.pad(depth, (pad, pad, 0, 0), mode="replicate")

    B, C, H, W = rgb_p.shape
    dev = rgb_p.device
    sign = -1.0 if view == "right" else 1.0

    s = sign * (depth_p - float(convergence)) * S
    x = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, 1, W)
    u = x + s

    # z-buffered splat of disparity (nearest column and its neighbour)
    tgt = torch.full((B, 1, H, W), -1.0, device=dev, dtype=torch.float32)
    t0 = u.floor().to(torch.long).clamp(0, W - 1).expand(B, 1, H, W)
    t1 = (t0 + 1).clamp(0, W - 1)
    tgt.scatter_reduce_(-1, t0, depth_p, reduce="amax", include_self=True)
    tgt.scatter_reduce_(-1, t1, depth_p, reduce="amax", include_self=True)

    valid = tgt > -0.5
    run = hole_run_length(valid)
    hole = ((~valid) & (run >= max(1, int(round(tear_px))))).to(torch.float32)

    if fill:
        depth_key = torch.where(valid, tgt, torch.full_like(tgt, 2.0))
        tgt = fill_along_x(tgt, valid, depth_key=depth_key, prefer="background")

    # backward resample:  u = x + sign*(d-c)*S   =>   x = u - sign*(d_t-c)*S
    src_x = x - sign * (tgt - float(convergence)) * S
    gx = (src_x / (W - 1)) * 2.0 - 1.0
    gy = (torch.arange(H, device=dev, dtype=torch.float32).view(1, 1, H, 1)
          / max(H - 1, 1)) * 2.0 - 1.0
    grid = torch.cat([gx.expand(B, 1, H, W), gy.expand(B, 1, H, W)], dim=1).permute(0, 2, 3, 1)
    out = F.grid_sample(rgb_p, grid, mode="bilinear", padding_mode="border", align_corners=True)

    out = out[..., pad:W - pad].contiguous()
    if return_mask:
        return out, hole[..., pad:W - pad].contiguous()
    return out, None
