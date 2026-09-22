r"""
Dataset for ntrainer, built on iw3's InpaintDataset.

Two things are changed, both because our pairs on disk are large (512-640px)
crops rather than nagadomi's tight ones:

1. **Crop size is configurable.** `iw3.training.inpaint.dataset.SIZE` is a module
   constant of 256. Training at 320/384 matters here: at divergence 16 a hole can
   be 130px wide on a 1920px frame, which is half of a 256 crop. A model never
   shown more than half a crop of context around a hole cannot learn to close one.

2. **Training crops are chosen by mask content, not colour variance.**
   Upstream picks the best of 4 random crops by `rect.std()` of the RGB, i.e. the
   most colourful crop -- with no guarantee the crop contains any hole at all.
   On a 640px pair whose mask covers a few percent of the area, most random crops
   are empty, so most samples would teach nothing. Here the score is
   `mask_sum * (colour_std + eps)`: a crop with no hole scores 0 and loses,
   and among crops that do contain a hole the more detailed one wins (detail
   under the mask is what teaches reconstruction rather than blur).

Eval is left deterministic (upstream's sliding-window `fixed_hard_example_crop`),
which is what makes the eval loss and the preview images comparable across
epochs -- the monitor's "same image, different epoch" view depends on it.
"""
from __future__ import annotations

from torchvision.transforms import functional as TF
from torchvision import transforms as T

from iw3.training.inpaint.dataset import InpaintDataset, fixed_hard_example_crop


__all__ = ["NTInpaintDataset"]


class MaskedHardExampleCrop:
    def __init__(self, size, samples=8, eps=1e-3):
        self.size = (size, size)
        self.samples = samples
        self.eps = eps

    def __call__(self, mask, x):
        best = None
        for _ in range(self.samples):
            i, j, h, w = T.RandomCrop.get_params(mask, self.size)
            m = TF.crop(mask, i, j, h, w)
            mask_sum = m.float().sum().item()
            if best is not None and mask_sum == 0:
                continue
            rgb = TF.crop(x, i, j, h, w)
            score = mask_sum * (rgb.std(dim=[1, 2]).sum().item() + self.eps)
            if best is None or score > best[0]:
                best = (score, m, rgb)
        return best[1], best[2]


class NTInpaintDataset(InpaintDataset):
    def __init__(self, input_dir, model_offset, training, size=256, crop_samples=8):
        self.size = size
        super().__init__(input_dir, model_offset, training)
        self.random_crop = MaskedHardExampleCrop(size, samples=crop_samples)

    def __getitem__(self, index):
        from nunif.utils.pil_io import load_image_simple

        im, _ = load_image_simple(self.files[index], color="rgb")
        mask, _ = load_image_simple(self.masks[index], color="gray")
        x = TF.to_tensor(im)
        mask = TF.to_tensor(mask)

        if self.training:
            mask = self.right_dilate(mask)
            mask = self.left_dilate(mask)
            mask, x = self.random_crop(mask, x)
        else:
            x, mask = fixed_hard_example_crop(self.size, x, mask)

        y = x.clone()
        y = TF.crop(y, self.model_offset, self.model_offset,
                    y.shape[-2] - self.model_offset * 2,
                    y.shape[-1] - self.model_offset * 2)
        mask = mask > 0
        return x, mask, y, index

    def describe(self):
        sizes = set()
        for fn in self.files[:16]:
            from nunif.utils.pil_io import load_image_simple
            im, _ = load_image_simple(fn, color="rgb")
            sizes.add(im.size)
        return (f"{len(self.files)} pairs, crop {self.size}, "
                f"source sizes {sorted(sizes)}")


def check_dataset(input_dir, size):
    """Fail early and loudly rather than 40 minutes into a run."""
    import os
    from os import path
    for split in ("train", "eval"):
        d = path.join(input_dir, split)
        if not path.isdir(d):
            raise RuntimeError(f"{d} does not exist -- expected <data-dir>/train and <data-dir>/eval")
        masks = [f for f in os.listdir(d) if f.endswith("_M.png")]
        if not masks:
            raise RuntimeError(f"{d} contains no *_M.png pairs")
        missing = [f for f in masks if not path.exists(path.join(d, f.replace("_M.png", "_C.png")))]
        if missing:
            raise RuntimeError(f"{d}: {len(missing)} masks have no _C.png partner, e.g. {missing[0]}")
    return True


if __name__ == "__main__":
    import sys
    ds = NTInpaintDataset(sys.argv[1], model_offset=16, training=True,
                          size=int(sys.argv[2]) if len(sys.argv) > 2 else 256)
    print(ds.describe())
    x, mask, y, i = ds[0]
    print("x", tuple(x.shape), "mask", tuple(mask.shape), mask.dtype,
          "y", tuple(y.shape), "mask px", int(mask.sum()))
