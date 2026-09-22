"""
ntrainer.pipeline -- the one place the depth->mask settings live.

Every tool (compare_masks, make_dataset, verify_format) goes through this.  It
exists because they previously each had their own copy of the sequence, drifted
apart, and `make_dataset` silently produced worse masks than the ones that had
been validated -- different bilateral settings and a randomised depth mapper
that was never in the validated path.

`MaskConfig` holds the validated defaults.  Change them here, everything moves
together.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from .depth_refine import refine_depth
from .geometry import disocclusion_mask, depth_edge_width


__all__ = ["MaskConfig", "MAPPER_SETS", "prepare_depth", "make_mask"]


# Depth mappers, grouped by measured effect on mask quality.  Numbers are from a
# real 1920px frame at divergence 16 (inner-boundary jitter, lower is better):
#
#   none         1.00      inv_mul_1  1.00      mul_1  2.12
#   inv_mul_2    1.00      inv_mul_3  1.00      mul_2  2.11   mul_3  2.10
#
# The mul_* family is softplus01 with bias 0.34-0.69 and scale 12: it crushes
# everything below the bias to zero, destroying background depth structure, and
# steepens the knee, amplifying noise into the mask boundary.  Renormalising
# rescues inv_mul_* (their output spans only 0.41-0.66 of the range) but cannot
# rescue mul_* (already ~0.9).  So mul_* is not in the default set.
MAPPER_SETS = {
    "none": ["none"],
    "safe": ["none", "none", "inv_mul_1", "inv_mul_2", "inv_mul_3"],
    "all": ["none", "none", "none", "inv_mul_1", "inv_mul_2", "inv_mul_3",
            "mul_1", "mul_2", "mul_3"],
}


@dataclass
class MaskConfig:
    """Validated Step 1 settings. One source of truth for every tool."""
    # depth refinement (joint bilateral upsampling of the stretched depth)
    refine: bool = True
    bilateral_radius: int = 3
    bilateral_iter: int = 1
    renormalize_after_mapper: bool = True
    # how ramp_px / edge_window are derived from the measured depth edge ramp
    ramp_scale: float = 3.0
    window_scale: float = 1.5
    # tear detection
    depth_step: float = 0.02
    tear_px: float = 0.0          # off: it overrides depth_step at high divergence
    sharpness: float = 0.6
    min_band_px: float = 1.0
    tear_rows: int = 2
    # silhouette anchoring and cleanup
    anchor: str = "edge"
    edge_frac: float = 0.1
    despike: int = 2
    smooth_px: int = 1
    outer_smooth_rows: int = 1
    occlusion_frac: float = 0.3

    @classmethod
    def from_args(cls, args):
        """Pick up any matching attribute from an argparse Namespace."""
        kw = {}
        for f in fields(cls):
            if hasattr(args, f.name):
                v = getattr(args, f.name)
                if v is not None:
                    kw[f.name] = v
        if getattr(args, "no_refine", False):
            kw["refine"] = False
        return cls(**kw)

    def describe(self):
        return (f"refine={self.refine}(r{self.bilateral_radius}x{self.bilateral_iter}) "
                f"depth_step={self.depth_step} sharpness={self.sharpness} "
                f"tear_px={self.tear_px} edge_frac={self.edge_frac} "
                f"despike={self.despike} smooth={self.smooth_px}")


def prepare_depth(rgb, depth, cfg: MaskConfig, mapper_fn=None):
    """
    Raw normalised disparity -> the depth the mask is computed from.

    Order matters: the mapper is applied first, then renormalised (the inv_mul_*
    mappers compress the output into 0.41-0.66 of the range, which shrinks every
    depth drop and pushes real edges under `depth_step`), and the RGB-guided
    refinement runs last so it sharpens the depth actually used.
    """
    d = depth.float()
    if mapper_fn is not None:
        d = mapper_fn(d)
        if cfg.renormalize_after_mapper:
            lo, hi = d.amin(), d.amax()
            d = (d - lo) / (hi - lo + 1e-8)
    if cfg.refine:
        d = refine_depth(rgb, d, radius=cfg.bilateral_radius, n_iter=cfg.bilateral_iter)
    return d


def make_mask(depth, divergence, convergence, cfg: MaskConfig, *,
              base_size=None, view="right", soft=False, ramp=None):
    """Prepared depth -> disocclusion mask, with ramp_px/edge_window sized from
    the depth's own measured edge ramp."""
    if ramp is None:
        ramp = depth_edge_width(depth)
    ramp_px = max(8, int(round(ramp * cfg.ramp_scale)))
    return disocclusion_mask(
        depth, divergence, convergence,
        base_size=base_size, view=view,
        ramp_px=ramp_px,
        edge_window=max(3, int(round(ramp * cfg.window_scale))),
        anchor_px=ramp_px,
        depth_step=cfg.depth_step,
        tear_px=cfg.tear_px,
        sharpness=cfg.sharpness,
        min_band_px=cfg.min_band_px,
        tear_rows=cfg.tear_rows,
        anchor=cfg.anchor,
        edge_frac=cfg.edge_frac,
        despike=cfg.despike,
        smooth_px=cfg.smooth_px,
        outer_smooth_rows=cfg.outer_smooth_rows,
        occlusion_frac=cfg.occlusion_frac,
        soft=soft,
    )
