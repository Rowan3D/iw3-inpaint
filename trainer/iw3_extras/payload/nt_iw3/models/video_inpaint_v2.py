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

import os

import torch
import torch.nn as nn

from nunif.models import register_model
from nunif.modules.compile_wrapper import conditional_compile
from nunif.modules.norm import FastLayerNorm
from nunif.modules.attention import WindowGMLP3d

from .inpaint_v2 import NTInpaintV2, GLUConvMLP, PRESETS, preset_kwargs, take_frames


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


def tail_chunk():
    """How many frames the per-frame parts run at once.

    6 is what a normal window emits (seq_len 12, pre/post padding 3), so in the
    ordinary case this changes nothing at all; it only stops the wider calls --
    the flush at a scene boundary or the end of a clip, which emit 9 -- from
    costing half as much again. The maths is per-frame either way.
    NT_TAIL_CHUNK=0 turns it off."""
    try:
        return int(os.environ.get("NT_TAIL_CHUNK", 6))
    except ValueError:
        return 6


def compute_dtype(module, x):
    """The dtype a fresh forward would produce here: the autocast dtype when it
    is on, the weights' otherwise. A cached tensor of any other dtype was made
    under a different precision and must not be reused -- it would silently pull
    the whole window into fp32."""
    if torch.is_autocast_enabled(x.device.type):
        try:
            return torch.get_autocast_dtype(x.device.type)
        except AttributeError:          # torch < 2.4
            return torch.get_autocast_gpu_dtype()
    return next(module.parameters()).dtype


def _level_slots(blocks2d, temporal):
    """The 2D block index each temporal block fires after, in order."""
    n, t = len(blocks2d), (0 if temporal is None else len(temporal))
    if t == 0:
        return []
    return sorted({int(round((i + 1) * n / t)) - 1 for i in range(t)})


def level_prefix(blocks2d, temporal):
    """How many of a level's 2D blocks run BEFORE its first temporal block.

    Those blocks -- and everything above them -- see one frame at a time, which
    is what lets a window reuse them from the window before it."""
    slots = _level_slots(blocks2d, temporal)
    return len(blocks2d) if not slots else slots[0] + 1


def _run_level(blocks2d, temporal, x, resume=0):
    """2D blocks in order with the temporal blocks spread evenly between them.
    `resume` = the caller has already run blocks [0, resume) and none of the
    temporal ones, so the level picks up at the mix that follows block
    resume - 1. That is how a cached prefix rejoins the network."""
    slots = _level_slots(blocks2d, temporal)
    if not slots:
        for i in range(resume, len(blocks2d)):
            x = blocks2d[i](x)
        return x
    for ti, s in enumerate(slots):
        start = 0 if ti == 0 else slots[ti - 1] + 1
        for i in range(max(start, resume), s + 1):
            x = blocks2d[i](x)
        x = temporal[ti](x)
    for i in range(slots[-1] + 1, len(blocks2d)):
        x = blocks2d[i](x)
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

    #: this class takes `keep` / `ids` / `cache`; a hook can test for it before
    #: trying to hand a window's emit list to whatever model iw3 loaded.
    WINDOW_CAPABLE = True

    def _prefix(self, x, mask):
        """The per-frame half of the encoder: the stem, enc1, down1 and the 1/8
        blocks below the first temporal block. -> (the 1/4 skip, the 1/8 tensor
        the first mix takes). Nothing in here reads another frame."""
        from nunif.modules.permute import pixel_unshuffle

        x = pixel_unshuffle(x, self.downscaling_factor)
        m = pixel_unshuffle(mask, self.downscaling_factor)
        if self.mask_channel:
            x = torch.cat([x, m], dim=1)
        x = self.patch(x)

        hole = m.amax(dim=1, keepdim=True) > 0.99
        x = torch.where(hole, self.mask_bias.to(x.dtype), x)

        x1 = self.enc1(x)
        d = self.down1(x1)
        for i in range(level_prefix(self.enc2, self.temporal2)):
            d = self.enc2[i](d)
        return x1, d

    def _cached_prefix(self, x, mask, ids, cache, keep=None):
        """`_prefix` for a window whose frames are named by `ids`, computing only
        the ones `cache` does not already hold. Consecutive windows overlap by
        the stride, and the queue's end padding repeats one frame several times,
        so most of a window is usually already there.

        Three things here are about memory rather than arithmetic, because a
        cache that costs more VRAM than the pass it saves is not worth having:
        entries are copies rather than views, `x1` is built for the frames the
        decoder will actually run, and dead entries go before the temporal
        blocks rather than after them."""
        want = compute_dtype(self, x)
        for fid in [f for f, (_, d) in cache.items() if d.dtype != want]:
            del cache[fid]              # made under a different precision
        todo, seen = [], set()
        for i, fid in enumerate(ids):
            if fid not in cache and fid not in seen:
                todo.append((i, fid))
                seen.add(fid)
        # The first window of a clip has nothing cached and would otherwise run
        # the prefix for all 12 frames at once -- one spike, at the start, that
        # sets the high-water mark for the whole run and hides every saving
        # after it. Run it in steady-state-sized groups instead; the prefix is a
        # per-frame function, so the numbers are the same either way.
        step = max(1, min(tail_chunk() or len(todo), len(todo)))
        for k in range(0, len(todo), step):
            group = todo[k:k + step]
            idx = torch.as_tensor([i for i, _ in group], device=x.device, dtype=torch.long)
            x1c, dc = self._prefix(x.index_select(0, idx), mask.index_select(0, idx))
            for j, (_, fid) in enumerate(group):
                # clone, not a slice: a view keeps the whole batch's storage
                # alive, so one surviving id would pin every frame computed
                # beside it -- half again as much memory as the cache needs.
                cache[fid] = (x1c[j:j + 1].clone(), dc[j:j + 1].clone())
            del x1c, dc
        d = torch.cat([cache[f][1] for f in ids], dim=0)
        # x1 is the 1/4 skip, read only by the decoder, and the decoder runs for
        # `keep` alone -- building it for the whole window would be twice the
        # tensor for no use.
        x1 = torch.cat([cache[f][0] for f in (ids if keep is None else [ids[k] for k in keep])],
                       dim=0)
        # Everything needed is copied out now. The next window is this one
        # advanced by len(keep) (iw3 removes exactly that many frames), so only
        # the tail can ever be asked for again. Dropping the rest here -- before
        # the temporal blocks, which is where the peak is -- halves what the
        # cache is holding at the one moment it matters.
        if keep is not None:
            live = set(ids[len(keep):])
            for fid in [f for f in cache if f not in live]:
                del cache[fid]
        return x1, d

    def _forward(self, x, mask, keep=None, ids=None, cache=None):
        x1_selected = False
        if cache is not None:
            if ids is None or len(ids) != x.shape[0]:
                raise ValueError(f"{self.name}: a prefix cache needs one id per frame")
            x1, d = self._cached_prefix(x, mask, ids, cache, keep=keep)
            x1_selected = keep is not None
        else:
            x1, d = self._prefix(x, mask)

        # The whole window from here to the end of enc3: the temporal blocks mix
        # every frame in it, and a mix is a learned seq_len x seq_len matrix over
        # the window (`proj_spatial`), so a frame's value depends on the SLOT it
        # sits in -- which is why this part cannot be shared between windows.
        x2 = _run_level(self.enc2, self.temporal2, d,
                        resume=level_prefix(self.enc2, self.temporal2))
        x3 = _run_level(self.enc3, self.temporal3, self.down2(x2))

        # `_run_level` puts the temporal blocks at the END of both levels, so
        # everything below is a per-frame function again: run it for the frames
        # the caller keeps, not for the ones the window only holds as context.
        if keep is not None:
            if not x1_selected:         # _cached_prefix already built it for keep
                x1 = take_frames(x1, keep)
            x2, x3 = take_frames(x2, keep), take_frames(x3, keep)

        # The decoder is per-frame, so how many frames go through it at once is
        # a memory choice and nothing else. A normal window keeps 6; the flush
        # at a scene boundary (and at the end of a clip) keeps 9, and without a
        # cap that one call would set the high-water mark for the whole run.
        n, step = x1.shape[0], tail_chunk()
        if step <= 0 or step >= n:
            return self._tail(x1, x2, x3)
        out = None
        for i in range(0, n, step):
            part = self._tail(x1[i:i + step], x2[i:i + step], x3[i:i + step])
            if out is None:
                out = part.new_empty((n,) + part.shape[1:])
            out[i:i + step] = part
        return out

    def _tail(self, x1, x2, x3):
        import torch.nn.functional as F
        from nunif.modules.permute import pixel_shuffle

        x2 = self.dec2(x2 + F.pixel_shuffle(self.up2(x3), 2))
        x1 = self.dec1(x1 + F.pixel_shuffle(self.up1(x2), 2))
        x = self.to_image(x1)
        return pixel_shuffle(x, self.downscaling_factor)

    def forward(self, x, mask, skip_i2i_offset=False, keep=None, ids=None, cache=None):
        # The temporal window is exactly seq_len, so the batch axis must be too.
        assert x.shape[0] % self.seq_len == 0, (
            f"{self.name}: got {x.shape[0]} frames, needs a multiple of seq_len="
            f"{self.seq_len}. Use infer() outside training, it pads for you.")
        return super().forward(x, mask, skip_i2i_offset=skip_i2i_offset,
                               keep=keep, ids=ids, cache=cache)

    def infer(self, x, mask, closing=False, inner_dilation=0, outer_dilation=0, base_width=None,
              keep=None, ids=None, cache=None):
        """Pad the sequence to a multiple of seq_len by repeating the end frames,
        then trim. iw3's video wrapper feeds whatever the frame queue holds.

        `keep` (positions in the sequence AS GIVEN) is what comes back, and the
        padding never reaches the caller. `ids` names each frame for `cache`, the
        per-frame prefix carried from one window to the overlapping next: two
        frames with the same id must be the same picture."""
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
        if ids is not None and pad:
            ids = [ids[0]] * pad1 + list(ids) + [ids[-1]] * pad2
        out = self.forward(x, mask, skip_i2i_offset=True, ids=ids, cache=cache,
                           keep=None if keep is None else [k + pad1 for k in keep])
        if keep is not None:
            return out                     # already the frames that were asked for
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
