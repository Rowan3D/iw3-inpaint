r"""
nt_inpaint_v2 -- a larger disocclusion inpainting backbone for iw3.

Why a new arch rather than just widening light_inpaint_v1
---------------------------------------------------------
The holes this model has to fill are *vertical bands*, and their width is set
by the divergence:

    hole_px = depth_drop * divergence * 0.005 * frame_width

At divergence 16 on a 1920-wide frame that is up to ~130px (measured on the
4KLSDB run: p50 22px, p90 63px, p99 103px, max 177px). To fill a 130px band
plausibly the network needs context from *both* sides of it, i.e. a receptive
field of a few hundred pixels.

light_inpaint_v1 has two scales: 1/4 (window 16 -> 64px of context) and 1/8
(window 8 -> 64px). Even counting the shifted windows and the 3x3 convs, its
usable context is around 100-150px. That is exactly where it starts failing,
which matches what divergence>5 looks like in the app.

So the change that matters is not width, it is a *third scale*. At 1/16
resolution a window of 8 spans 128px of the source image, and six shifted
blocks there push the effective receptive field past 400px. Width is increased
too (that is where the capacity for texture detail comes from), but the depth
of the pyramid is what makes high divergence tractable at all.

Everything else is deliberately the same as light_inpaint_v1 -- same gMLP
block, same pixel-unshuffle stem, same mask_bias trick, same offset/blend so
iw3's tiling code needs no changes.

Differences from light_inpaint_v1, all intentional:

  * Three scales (1/4, 1/8, 1/16) instead of two, with a decoder at each.
  * Configurable width/depth via presets (see PRESETS).
  * The mask is fed into the stem as extra channels, not only as `mask_bias`.
    mask_bias only marks 4x4 cells that are *entirely* hole (`> 0.99` after
    unshuffle+amax); a cell that is half hole is indistinguishable from a cell
    that happens to be dark. Passing the unshuffled mask costs one 1x1 conv's
    worth of input channels and removes that ambiguity. Disable with
    `mask_channel=False`.
  * Padding is `(-size) % mod` rather than `mod - size % mod`, so an already
    aligned input is not padded by a whole extra `mod`. Upstream always pads,
    which at a 256 crop means processing 384px -- 2.25x the pixels for nothing.

Registered as `inpaint.nt_inpaint_v2`; the preset variants register their own
names so `--arch` alone is enough to reproduce a run.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nunif.models import I2IBaseModel, register_model
from nunif.modules.permute import pixel_shuffle, pixel_unshuffle
from nunif.modules.replication_pad2d import replication_pad2d_naive, ReplicationPad2dNaive
from nunif.modules.init import basic_module_init, icnr_init
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.norm import FastLayerNorm
from nunif.modules.attention import WindowGMLP2d
from nunif.modules.gaussian_filter import SeparableGaussianFilter2d
from iw3.dilation import mask_closing, dilate_inner, dilate_outer


__all__ = ["NTInpaintV2", "PRESETS", "preset_kwargs"]


class GLUConvMLP(nn.Module):
    """Same as light_inpaint_v1's, kept verbatim so behaviour is identical."""

    def __init__(self, in_channels, out_channels, kernel_size=3, mlp_ratio=2, padding=True):
        super().__init__()
        mid = int(out_channels * mlp_ratio)
        if padding:
            self.pad = ReplicationPad2dNaive(((kernel_size - 1) // 2,) * 4, detach=True)
        else:
            self.pad = nn.Identity()
        self.w1 = nn.Conv2d(in_channels, mid, kernel_size=1, stride=1, padding=0)
        self.w2 = nn.Conv2d(mid // 2, out_channels, kernel_size=kernel_size, stride=1, padding=0)
        basic_module_init(self)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x):
        x = self.w1(x)
        x = F.glu(x, dim=1)
        x = self.pad(x)
        x = self.w2(x)
        return x


class GMLPBlock(nn.Module):
    """
    NOT quite light_inpaint_v1's block: `nunif.modules.attention.GMLP.forward`
    already ends with `x = x + shortcut`, so upstream's

        x = x + self.gmlp(x, norm1, norm2)

    adds the shortcut a second time and the block computes `2x + f(x)`. The
    activation scale therefore doubles at every block. Upstream has 6 blocks and
    gets away with it (output std ~24 at init, measured); this model has 10-19
    and it does not -- output std at init was ~50000, which trains badly and
    overflows fp16 (max 65504) outright.

    So the shortcut is added once, and the output projection of each residual
    branch is scaled at init by `residual_init` (see NTInpaintV2). Measured
    output std at init with the default 0.5: 0.6 (preset s) / 1.1 (preset l).
    """

    def __init__(self, in_channels, window_size, mlp_ratio=2, shift=False, kernel_size=3):
        super().__init__()
        self.gmlp = WindowGMLP2d(in_channels, window_size=window_size, shift=shift, mlp_ratio=mlp_ratio)
        self.norm1 = FastLayerNorm(in_channels, bias=False)
        self.norm2 = FastLayerNorm(in_channels * mlp_ratio, bias=False)
        self.glu_conv = GLUConvMLP(in_channels, in_channels, mlp_ratio=1, kernel_size=kernel_size)

    def scale_residual_init(self, scale):
        with torch.no_grad():
            for m in (self.gmlp.gmlp.proj_out, self.glu_conv.w2):
                m.weight.mul_(scale)
                if m.bias is not None:
                    m.bias.mul_(scale)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x):
        x = self.gmlp(x, self.norm1, self.norm2)   # residual is inside GMLP
        x = x + self.glu_conv(x)
        return x


def _stack(n, channels, window, mlp_ratio, shift_first=False):
    """n blocks with the shift flag alternating -- shifted windows are what let
    information cross window borders, so they must not all be the same."""
    return nn.Sequential(*[
        GMLPBlock(channels, window_size=window, mlp_ratio=mlp_ratio,
                  shift=((i % 2 == 0) == shift_first))
        for i in range(n)
    ])


# name -> kwargs. `params` in the comment is measured, see tools/bench_model.py.
PRESETS = {
    # ~7.6M params. A drop-in bigger-than-stock option if speed is critical.
    "s": dict(dims=(112, 224, 320), blocks=(1, 4, 4), dec_blocks=(1, 2)),
    # ~16M params. The default.
    "b": dict(dims=(128, 256, 384), blocks=(1, 4, 6), dec_blocks=(1, 2)),
    # ~34M params.
    "l": dict(dims=(160, 320, 512), blocks=(2, 6, 8), dec_blocks=(2, 3)),
}


def preset_kwargs(name):
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")
    return dict(PRESETS[name])


@register_model
class NTInpaintV2(I2IBaseModel):
    name = "inpaint.nt_inpaint_v2"

    def __init__(self, dims=(128, 256, 384), blocks=(1, 4, 6), dec_blocks=(1, 2),
                 windows=(16, 8, 8), mlp_ratio=2, mask_channel=True, residual_init=0.5):
        super().__init__(locals(), scale=1, offset=16, in_channels=3, blend_size=8)
        assert len(dims) == 3 and len(blocks) == 3 and len(dec_blocks) == 2
        self.downscaling_factor = 4
        pack = self.downscaling_factor ** 2
        C1, C2, C3 = dims
        w1, w2, w3 = (w if isinstance(w, (tuple, list)) else (w, w) for w in windows)
        self.mask_channel = mask_channel

        # Input must be a multiple of this for every window to tile exactly:
        #   level 1 is 1/4  of input, needs a multiple of w1
        #   level 2 is 1/8            needs a multiple of w2
        #   level 3 is 1/16           needs a multiple of w3
        self.mod = max(w1[0], w1[1], 2 * w2[0], 2 * w2[1], 4 * w3[0], 4 * w3[1])
        self.align = self.mod * self.downscaling_factor

        in_ch = 3 * pack + (pack if mask_channel else 0)
        self.patch = nn.Sequential(
            nn.Conv2d(in_ch, C1, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mask_bias = nn.Parameter(torch.zeros(1, C1, 1, 1))

        self.enc1 = _stack(blocks[0], C1, w1, mlp_ratio, shift_first=True)
        self.down1 = nn.Conv2d(C1, C2, kernel_size=2, stride=2, padding=0)
        self.enc2 = _stack(blocks[1], C2, w2, mlp_ratio)
        self.down2 = nn.Conv2d(C2, C3, kernel_size=2, stride=2, padding=0)
        self.enc3 = _stack(blocks[2], C3, w3, mlp_ratio)

        self.up2 = nn.Conv2d(C3, C2 * 4, kernel_size=1, stride=1, padding=0)
        self.dec2 = _stack(dec_blocks[1], C2, w2, mlp_ratio)
        self.up1 = nn.Conv2d(C2, C1 * 4, kernel_size=1, stride=1, padding=0)
        self.dec1 = _stack(dec_blocks[0], C1, w1, mlp_ratio)

        self.to_image = nn.Sequential(
            ReplicationPad2dNaive((1,) * 4, detach=True),
            nn.Conv2d(C1, 3 * pack, kernel_size=3, stride=1, padding=0),
        )
        basic_module_init(self.patch)
        basic_module_init(self.down1)
        basic_module_init(self.down2)
        basic_module_init(self.up1)
        basic_module_init(self.up2)
        icnr_init(self.to_image[-1], scale_factor=self.downscaling_factor)
        nn.init.trunc_normal_(self.mask_bias, 0, 0.01)
        if residual_init != 1.0:
            for m in self.modules():
                if isinstance(m, GMLPBlock):
                    m.scale_residual_init(residual_init)

        self.mask_blur = SeparableGaussianFilter2d(1, kernel_size=15, padding=15 // 2)

    # -- identical to light_inpaint_v1 so iw3's call sites work unchanged ----
    def preprocess(self, x, mask, closing=False, inner_dilation=0, outer_dilation=0, base_width=None):
        if closing:
            mask = mask_closing(mask)
        else:
            mask = mask.float()
        mask = dilate_inner(mask, n_iter=inner_dilation, base_width=base_width)
        mask = dilate_outer(mask, n_iter=outer_dilation, base_width=base_width)
        x = x * (1 - mask)
        mask = torch.clamp(self.mask_blur(mask) + mask, 0, 1)
        return x, mask

    def infer(self, x, mask, closing=False, inner_dilation=0, outer_dilation=0, base_width=None):
        x, mask = self.preprocess(x, mask, closing=closing,
                                  inner_dilation=inner_dilation, outer_dilation=outer_dilation,
                                  base_width=base_width)
        return self.forward(x, mask, skip_i2i_offset=True)

    def _forward(self, x, mask):
        x = pixel_unshuffle(x, self.downscaling_factor)
        m = pixel_unshuffle(mask, self.downscaling_factor)
        if self.mask_channel:
            x = torch.cat([x, m], dim=1)
        x = self.patch(x)

        hole = m.amax(dim=1, keepdim=True) > 0.99
        x = torch.where(hole, self.mask_bias.to(x.dtype), x)

        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))

        x2 = self.dec2(x2 + F.pixel_shuffle(self.up2(x3), 2))
        x1 = self.dec1(x1 + F.pixel_shuffle(self.up1(x2), 2))

        x = self.to_image(x1)
        return pixel_shuffle(x, self.downscaling_factor)

    def forward(self, x, mask, skip_i2i_offset=False):
        src = x
        x = (x - 0.5) / 0.5

        input_height, input_width = x.shape[2:]
        # `(-n) % a` is 0 when already aligned, unlike `a - n % a`.
        pad1 = (-input_width) % self.align
        pad2 = (-input_height) % self.align
        if pad1 or pad2:
            padding = (0, pad1, 0, pad2)
            x = replication_pad2d_naive(x, padding, detach=True)
            mask = replication_pad2d_naive(mask, padding, detach=True)

        x = self._forward(x, mask)

        if pad1 or pad2:
            x = F.pad(x, (0, -pad1, 0, -pad2))
            mask = F.pad(mask, (0, -pad1, 0, -pad2))

        if not skip_i2i_offset:
            src = F.pad(src.to(x.dtype), (-self.i2i_offset,) * 4)
            mask = F.pad(mask, (-self.i2i_offset,) * 4)
            x = F.pad(x, (-self.i2i_offset,) * 4)

        mask = mask.expand_as(src)
        src = src * (1 - mask) + x * mask

        if not self.training:
            src = src.clamp(0, 1)
        return src


def _make_preset_class(preset):
    kw = preset_kwargs(preset)

    @register_model
    class _Preset(NTInpaintV2):
        name = f"inpaint.nt_inpaint_v2_{preset}"

        def __init__(self, **kwargs):
            merged = dict(kw)
            merged.update(kwargs)
            super().__init__(**merged)

    _Preset.__name__ = f"NTInpaintV2{preset.upper()}"
    _Preset.__qualname__ = _Preset.__name__
    return _Preset


NTInpaintV2S = _make_preset_class("s")
NTInpaintV2B = _make_preset_class("b")
NTInpaintV2L = _make_preset_class("l")
