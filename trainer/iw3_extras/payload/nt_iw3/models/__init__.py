"""
Importing this package registers ntrainer's models with nunif's model registry,
which is what makes `--arch inpaint.nt_inpaint_v2_b` and `load_model()` work.
"""
from .inpaint_v2 import NTInpaintV2, PRESETS, preset_kwargs  # noqa: F401
from .video_inpaint_v2 import (  # noqa: F401
    NTVideoInpaintV2, SEQ_LEN, transfer_image_weights,
)

__all__ = ["NTInpaintV2", "NTVideoInpaintV2", "PRESETS", "preset_kwargs",
           "SEQ_LEN", "transfer_image_weights"]
