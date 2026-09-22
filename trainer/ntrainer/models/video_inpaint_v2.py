r"""
nt_video_inpaint_v2 -- the temporal variant of nt_inpaint_v2.

It is the *same network* as the image model with temporal blocks inserted, and
that is deliberate: every 2D module keeps its exact name and shape, so

    video_model.load_state_dict(image_model.state_dict(), strict=False)

transfers the whole trained image backbone and leaves only the new temporal
blocks randomly initialised. Passing the trained image .pth to `train.bat --video
--checkpoint-file` does exactly this. A video model is therefore a short
fine-tune on top of the image run, not a second 600-epoch run from scratch. (iw3's own pair cannot do this -- light_inpaint_v1
and light_video_inpaint_v1 differ in the stem, the block config and the module
layout, so nothing transfers.)

What the temporal block does
----------------------------
`GMLP3DBlock` reshapes BCHW -> 1,C,B,H,W and runs `WindowGMLP3d` with window
`(seq_len, 1, 1)`. That window is 1x1 spatially, so it mixes *only along time*,
independently per feature position: every location sees its own value across all
`seq_len` frames and nothing else. All spatial modelling stays in the 2D blocks.

They are inserted at the 1/8 and 1/16 levels only. The 1/4 level is by far the
most expensive (16x the tokens of 1/16) and temporal mixing there buys little --
nagadomi puts them at 1/8 alone; we add 1/16 as well because that is where this
model's long-range context lives.

Note this model alone does not make output temporally stable -- the loss has to
ask for it. `--video` selects `temporal_l1lpips`, whose `TemporalGradientLoss`
compares consecutive-frame differences between prediction and truth. Without it
the temporal blocks have no reason to do anything useful.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nunif.models import register_model
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.norm import FastLayerNorm
from nunif.modules.attention import WindowGMLP3d

from .inpaint_v2 import NTInpaintV2, GLUConvMLP, PRESETS, preset_kwargs


__all__ = ["NTVideoInpaintV2", "SEQ_LEN"]

SEQ_LEN = 12   # iw3's video wrappers hardcode model_seq = 12


class GMLP3DBlock(nn.Module):
    """Temporal-only gMLP block. Same shortcut fix as the 2D block: nunif's GMLP
    already adds its own residual, so adding it again doubles the activation
    scale at every block."""

    def __init__(self, in_channels, seq_len=SEQ_LEN, mlp_ratio=2):
        super().__init__()
        self.gmlp = WindowGMLP3d(in_channels, window_size=(seq_len, 1, 1),
                                 mlp_ratio=mlp_ratio, shift=False)
        self.norm1 = FastLayerNorm(in_channels, bias=False)
        self.norm2 = FastLayerNorm(in_channels * mlp_ratio, bias=False)
        self.glu_conv = GLUConvMLP(in_channels, in_channels, mlp_ratio=1)

    def scale_residual_init(self, scale):
        with torch.no_grad():
            for m in (self.gmlp.gmlp.proj_out, self.glu_conv.w2):
                m.weight.mul_(scale)
                if m.bias is not None:
                    m.bias.mul_(scale)

    @conditional_compile(["NUNIF_TRAIN"])
    def forward(self, x):
        B, C, H, W = x.shape
        # BCHW -> 1,C,D=B,H,W : the batch axis is time
        x = x.permute(1, 0, 2, 3).reshape(1, C, B, H, W)
        x = self.gmlp(x, self.norm1, self.norm2)
        x = x.permute(0, 2, 1, 3, 4).reshape(B, C, H, W)
        x = x + self.glu_conv(x)
        return x


def _run_level(blocks2d, temporal, x):
    """2D blocks in order with the temporal blocks spread evenly between them."""
    if temporal is None or len(temporal) == 0:
        return blocks2d(x)
    n, t = len(blocks2d), len(temporal)
    slots = {int(round((i + 1) * n / t)) - 1 for i in range(t)}
    ti = 0
    for i, blk in enumerate(blocks2d):
        x = blk(x)
        if i in slots and ti < t:
            x = temporal[ti](x)
            ti += 1
    return x


@register_model
class NTVideoInpaintV2(NTInpaintV2):
    name = "inpaint.nt_video_inpaint_v2"

    def __init__(self, dims=(128, 256, 384), blocks=(1, 4, 6), dec_blocks=(1, 2),
                 windows=(16, 8, 8), mlp_ratio=2, mask_channel=True, residual_init=0.5,
                 seq_len=SEQ_LEN, temporal=(2, 2)):
        super().__init__(dims=dims, blocks=blocks, dec_blocks=dec_blocks, windows=windows,
                         mlp_ratio=mlp_ratio, mask_channel=mask_channel,
                         residual_init=residual_init)
        # re-register kwargs so save/load reconstructs the video model, not the base
        self.kwargs = {}
        self.register_kwargs(dict(dims=dims, blocks=blocks, dec_blocks=dec_blocks,
                                  windows=windows, mlp_ratio=mlp_ratio,
                                  mask_channel=mask_channel, residual_init=residual_init,
                                  seq_len=seq_len, temporal=temporal))
        self.seq_len = seq_len
        self.sequence_offset = 0      # read by the video dataset via the trainer
        t2, t3 = temporal
        self.temporal2 = nn.ModuleList([GMLP3DBlock(dims[1], seq_len=seq_len, mlp_ratio=mlp_ratio)
                                        for _ in range(t2)])
        self.temporal3 = nn.ModuleList([GMLP3DBlock(dims[2], seq_len=seq_len, mlp_ratio=mlp_ratio)
                                        for _ in range(t3)])
        if residual_init != 1.0:
            for m in list(self.temporal2) + list(self.temporal3):
                m.scale_residual_init(residual_init)

    def _forward(self, x, mask):
        from nunif.modules.permute import pixel_shuffle, pixel_unshuffle
        import torch.nn.functional as F

        x = pixel_unshuffle(x, self.downscaling_factor)
        m = pixel_unshuffle(mask, self.downscaling_factor)
        if self.mask_channel:
            x = torch.cat([x, m], dim=1)
        x = self.patch(x)

        hole = m.amax(dim=1, keepdim=True) > 0.99
        x = torch.where(hole, self.mask_bias.to(x.dtype), x)

        x1 = self.enc1(x)
        x2 = _run_level(self.enc2, self.temporal2, self.down1(x1))
        x3 = _run_level(self.enc3, self.temporal3, self.down2(x2))

        x2 = self.dec2(x2 + F.pixel_shuffle(self.up2(x3), 2))
        x1 = self.dec1(x1 + F.pixel_shuffle(self.up1(x2), 2))

        x = self.to_image(x1)
        return pixel_shuffle(x, self.downscaling_factor)

    def forward(self, x, mask, skip_i2i_offset=False):
        # The temporal window is exactly seq_len, so the batch axis must be too.
        assert x.shape[0] % self.seq_len == 0, (
            f"{self.name}: got {x.shape[0]} frames, needs a multiple of seq_len="
            f"{self.seq_len}. Use infer() outside training, it pads for you.")
        return super().forward(x, mask, skip_i2i_offset=skip_i2i_offset)

    def infer(self, x, mask, closing=False, inner_dilation=0, outer_dilation=0, base_width=None):
        """Pad the sequence to a multiple of seq_len by repeating the end frames,
        then trim. iw3's video wrapper feeds whatever the frame queue holds."""
        n = x.shape[0]
        pad = (-n) % self.seq_len
        pad1 = pad // 2
        pad2 = pad - pad1
        if pad:
            x = torch.cat([x[0:1].detach()] * pad1 + [x] + [x[-1:].detach()] * pad2, dim=0)
            mask = torch.cat([mask[0:1].detach()] * pad1 + [mask] + [mask[-1:].detach()] * pad2,
                             dim=0)
        x, mask = self.preprocess(x, mask, closing=closing,
                                  inner_dilation=inner_dilation, outer_dilation=outer_dilation,
                                  base_width=base_width)
        out = self.forward(x, mask, skip_i2i_offset=True)
        if pad1:
            out = out[pad1:]
        if pad2:
            out = out[:-pad2]
        return out


def _make_preset_class(preset):
    kw = preset_kwargs(preset)

    @register_model
    class _Preset(NTVideoInpaintV2):
        name = f"inpaint.nt_video_inpaint_v2_{preset}"

        def __init__(self, **kwargs):
            merged = dict(kw)
            merged.update(kwargs)
            super().__init__(**merged)

    _Preset.__name__ = f"NTVideoInpaintV2{preset.upper()}"
    _Preset.__qualname__ = _Preset.__name__
    return _Preset


NTVideoInpaintV2S = _make_preset_class("s")
NTVideoInpaintV2B = _make_preset_class("b")
NTVideoInpaintV2L = _make_preset_class("l")


def transfer_image_weights(video_model, image_state_dict, verbose=True):
    """Load a trained nt_inpaint_v2 into nt_video_inpaint_v2.

    Returns (loaded, skipped). Everything except temporal2/temporal3 must load;
    anything else missing means the two archs were built with different kwargs,
    which is worth failing loudly over rather than silently training from noise.
    """
    own = video_model.state_dict()
    loadable = {k: v for k, v in image_state_dict.items()
                if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in loadable]
    unexpected = [k for k in image_state_dict if k not in own]
    video_model.load_state_dict(loadable, strict=False)
    non_temporal_missing = [k for k in missing if not k.startswith(("temporal2.", "temporal3."))]
    if verbose:
        print(f"transferred {len(loadable)}/{len(own)} tensors from the image model")
        print(f"  new (temporal) tensors left at init: "
              f"{len([k for k in missing if k.startswith(('temporal2.', 'temporal3.'))])}")
        if unexpected:
            print(f"  ignored {len(unexpected)} tensors not present in the video model")
    if non_temporal_missing:
        raise RuntimeError(
            "these non-temporal tensors did not transfer, so the two models were built "
            f"with different kwargs: {non_temporal_missing[:8]}"
            + (" ..." if len(non_temporal_missing) > 8 else ""))
    return len(loadable), len(missing)


PRESETS = PRESETS  # re-exported for callers that only import this module
