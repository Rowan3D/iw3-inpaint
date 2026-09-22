"""
ntrainer.dataset_io -- write training pairs in exactly the format
iw3/training/inpaint/{create_training_data,dataset}.py agree on.

The contract, read off the consumer code rather than assumed:

* Layout      `<data_dir>/train/` and `<data_dir>/eval/`.
* Pairing     `dataset.InpaintDataset.load_files()` lists every file ending
              `_M.png` and requires the same name with `_M.png` -> `_C.png`.
              It raises if the RGB partner is missing, so the two must always
              be written together.
* `_C.png`    the clean RGB crop.  This is BOTH the model input and the
              ground truth -- `dataset.__getitem__` does `y = x.clone()`.
              There is no "damaged" image on disk.
* `_M.png`    8-bit grayscale.  **White (non-zero) is the hole to inpaint.**
              `LightInpaintV1.preprocess()` does `x = x * (1 - mask)`, i.e.
              the masked region is erased, and the final composite is
              `src * (1 - mask) + prediction * mask`.
* Polarity    mask 1 = inpaint, 0 = keep.  Not the other way round.
* THRESHOLD   `dataset.__getitem__` ends with `mask = mask > 0`.  Any pixel at
              or above 1/255 becomes a hole, so a soft / anti-aliased mask
              silently dilates by its entire falloff.  Masks must be written
              already binarised -- `save_pair()` enforces this.
* Size        crops must be >= 256 (`dataset.SIZE`), since training random-
              crops 256 out of them and eval slides a 256 window.
* Sparse      `create_training_data.save_images()` drops a crop whose mask
              sums to <= 300, to avoid feeding near-empty masks.
"""
from __future__ import annotations

import torch
from torchvision.transforms import functional as TF
from torchvision import transforms as T


__all__ = ["random_crop", "random_hard_example_crop", "save_pair", "detail_map",
           "MIN_MASK_SUM", "TRAIN_CROP_SIZE"]


MIN_MASK_SUM = 300       # create_training_data.save_images()
TRAIN_CROP_SIZE = 256    # dataset.SIZE


def detail_map(rgb):
    """
    Local high-frequency energy, as mean |gradient| per pixel.

    This is what decides whether a sample teaches detail or teaches blur.  The
    model's job is to reconstruct whatever sits UNDER the mask, so if that
    region is an out-of-focus background the only correct answer is mush, and
    training on enough of those teaches the model to always produce mush.
    Aesthetic-scored corpora (4KLSDB is Q-Align top-80%) skew heavily toward
    shallow depth of field, so this needs measuring rather than assuming.

    rgb: (3,H,W) in [0,1]  ->  (1,H,W)
    """
    g = rgb.mean(dim=0, keepdim=True)
    dx = (g[:, :, 1:] - g[:, :, :-1]).abs()
    dy = (g[:, 1:, :] - g[:, :-1, :]).abs()
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1), mode="replicate")
    return (dx + dy) * 0.5


def random_crop(size, *images):
    i, j, h, w = T.RandomCrop.get_params(images[0], (size, size))
    return tuple(TF.crop(im, i, j, h, w) for im in images)


def random_hard_example_crop(size, n, *images):
    """Best of `n` random crops by mask area -- mirrors create_training_data.py."""
    assert n > 0
    best = None
    for _ in range(n):
        crops = random_crop(size, *images)
        s = crops[-1].float().sum().item()
        if best is None or s > best[0]:
            best = (s, crops)
    return best


def save_pair(rgb, mask, output_base, size=512, num_samples=2,
              min_mask_sum=MIN_MASK_SUM, tries=8, binarize=True, threshold=0.5,
              detail=None, min_detail=0.0, stats=None):
    """
    Write `_C.png` / `_M.png` crop pairs for one source frame.

    rgb:   (3,H,W) float in [0,1]
    mask:  (1,H,W) float in [0,1], 1 == hole
    detail: (1,H,W) from `detail_map()`. When given, crops are chosen by
            mask area *weighted by how detailed the masked content is*, and
            crops whose masked region is flatter than `min_detail` are dropped.
    stats: optional list; each kept crop appends its masked-detail score.
    Returns the number of pairs written.
    """
    assert rgb.ndim == 3 and mask.ndim == 3, f"want (C,H,W), got {rgb.shape} / {mask.shape}"
    assert size >= TRAIN_CROP_SIZE, (
        f"crop size {size} is below dataset.SIZE={TRAIN_CROP_SIZE}; "
        "training would fail to crop")
    rgb = rgb.detach().float().clamp(0, 1).cpu()
    mask = mask.detach().float().cpu()
    if binarize:
        # dataset.py does `mask > 0`, so anything non-zero is a hole.  Writing a
        # soft mask would dilate it by the whole anti-aliased falloff.
        mask = (mask > threshold).float()
    mask = mask.clamp(0, 1)

    written = 0
    for i in range(num_samples):
        if detail is None:
            mask_sum, (rgb_rect, mask_rect) = random_hard_example_crop(size, tries, rgb, mask)
            d_score = None
        else:
            best = None
            for _ in range(tries):
                r, m, dt = random_crop(size, rgb, mask, detail)
                area = m.sum().item()
                if area <= 0:
                    continue
                dsc = (dt * m).sum().item() / area       # detail of what must be filled
                if best is None or area * dsc > best[0]:
                    best = (area * dsc, area, dsc, (r, m))
            if best is None:
                continue
            _, mask_sum, d_score, (rgb_rect, mask_rect) = best
            if d_score < min_detail:
                continue
        if mask_sum > min_mask_sum:
            TF.to_pil_image(rgb_rect).save(f"{output_base}_{i}_C.png")
            TF.to_pil_image(mask_rect).save(f"{output_base}_{i}_M.png")
            written += 1
            if stats is not None and d_score is not None:
                stats.append(d_score)
    return written
