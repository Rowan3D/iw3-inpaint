r"""
Clip dataset for the video model.

Built on iw3's `VideoInpaintDataset`. One sample is `seq` consecutive frames from
one clip folder, cropped with the *same* rectangle in every frame -- that shared
rectangle is what makes the frames correspond, so it must not be re-randomised
per frame.

Only two things change from upstream:

* **Crop size is configurable** (upstream hardcodes `dataset_video.SIZE = 256`),
  for the same reason as the image dataset: a 130px hole at divergence 16 is half
  of a 256 crop.
* **Clips shorter than `seq` frames fail with a useful message** instead of an
  `IOError` naming only the folder.

Upstream's crop selection is already mask-aware here (best of 4 by mask sum over
the whole clip), unlike the image dataset's colour-variance pick, so it is kept.

Layout, as iw3's loader requires it:

    <data-dir>/train/<clip-name>/0000_C.png
                                0000_M.png
                                0001_C.png ...
    <data-dir>/eval/<clip-name>/...

Frames are read in sorted filename order, so zero-padded numbering is required.
"""
from __future__ import annotations

import os
from os import path

import torch
from torchvision.transforms import functional as TF

from iw3.training.inpaint.dataset_video import (
    VideoInpaintDataset, random_hard_example_crop, fixed_hard_example_crop,
)
from nunif.utils.image_loader import ImageLoader


__all__ = ["NTVideoInpaintDataset", "check_video_dataset"]


class NTVideoInpaintDataset(VideoInpaintDataset):
    def __init__(self, input_dir, model_offset, model_sequence_offset, training,
                 size=256, seq=12):
        self.size = size
        self.seq = seq
        super().__init__(input_dir, model_offset, model_sequence_offset, training)

    def __getitem__(self, index):
        folder = self.folders[index]
        x, mask = self.load_files(folder, self.seq, self.training)

        from nunif.utils.pil_io import load_image_simple
        x = torch.stack([TF.to_tensor(load_image_simple(fn, color="rgb")[0]) for fn in x])
        mask = torch.stack([TF.to_tensor(load_image_simple(fn, color="gray")[0]) for fn in mask])

        if self.training:
            x = self.color_jitter(x)
            mask = self.right_dilate(mask)
            mask = self.left_dilate(mask)
            # dataset_video's version returns the crops directly (unlike the
            # same-named helper in create_training_data_video.py, which also
            # returns the mask sum)
            x, mask = random_hard_example_crop(self.size, 4, x, mask)
        else:
            x, mask = fixed_hard_example_crop(self.size, x, mask)

        y = x.clone()
        y = y[:, :,
              self.model_offset: self.model_offset + (y.shape[-2] - self.model_offset * 2),
              self.model_offset: self.model_offset + (y.shape[-1] - self.model_offset * 2)]
        if self.model_sequence_offset > 0:
            y = y[self.model_sequence_offset:-self.model_sequence_offset]

        mask = mask > 0
        return x, mask, y, index


def check_video_dataset(input_dir, size, seq):
    """Fail before the first epoch rather than at a random index during it."""
    for split in ("train", "eval"):
        d = path.join(input_dir, split)
        if not path.isdir(d):
            raise RuntimeError(
                f"{d} does not exist. The video dataset needs <data-dir>/train and "
                f"<data-dir>/eval, each holding one folder per clip")
        clips = [f for f in sorted(os.listdir(d)) if path.isdir(path.join(d, f))]
        if not clips:
            raise RuntimeError(
                f"{d} has no clip folders. For video, frames live in per-clip "
                f"subfolders, not loose in train/ -- use make_video_dataset.bat")
        short, unpaired = [], []
        for c in clips:
            files = ImageLoader.listdir(path.join(d, c))
            masks = [f for f in files if f.endswith("_M.png")]
            if len(masks) < seq:
                short.append(f"{c} ({len(masks)})")
            for f in masks:
                if not path.exists(f.replace("_M.png", "_C.png")):
                    unpaired.append(f)
                    break
        if short:
            raise RuntimeError(
                f"{d}: {len(short)} clip(s) have fewer than seq={seq} frames, "
                f"e.g. {', '.join(short[:5])}")
        if unpaired:
            raise RuntimeError(f"{d}: masks without a _C.png partner, e.g. {unpaired[0]}")
        print(f"{split:6s}: {len(clips)} clips")
    return True
