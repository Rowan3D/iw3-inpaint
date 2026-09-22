"""Small PIL helpers for building labelled comparison sheets."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def _font(size=18):
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _resize(t, width):
    """Downscale on whatever device the tensor is already on.

    Converting a full-resolution frame to PIL and then LANCZOS-resizing it is
    by far the slowest thing this tool does -- it dominates the wall clock per
    divergence.  Doing the resize in tensor space first means PIL only ever
    sees a tile-sized image.
    """
    if width is None or t.shape[-1] == width:
        return t
    h = max(1, round(t.shape[-2] * width / t.shape[-1]))
    need_batch = t.ndim == 3
    if need_batch:
        t = t.unsqueeze(0)
    up = width > t.shape[-1]
    t = F.interpolate(t.float(), size=(h, width), mode="bilinear",
                      align_corners=False, antialias=not up)
    return t[0] if need_batch else t


def to_pil(t, width=None):
    """(1,C,H,W) or (C,H,W) float tensor in [0,1] -> PIL RGB"""
    if t.ndim == 4:
        t = t[0]
    t = _resize(t, width)
    t = t.detach().float().clamp(0, 1).cpu()
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    arr = (t.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
    return Image.fromarray(arr, mode="RGB")


def overlay_mask(rgb, mask, color=(255, 40, 40), alpha=0.62, desaturate=0.55,
                 width=None):
    """RGB tensor + mask tensor -> PIL with the mask tinted on top."""
    if rgb.ndim == 4:
        rgb = rgb[0]
    if mask.ndim == 4:
        mask = mask[0]
    rgb = _resize(rgb, width).detach().float().clamp(0, 1).cpu()
    m = mask.detach().float()
    if m.shape[0] != 1:
        m = m[0:1]
    m = _resize(m, width).clamp(0, 1).cpu()
    gray = rgb.mean(dim=0, keepdim=True).repeat(3, 1, 1)
    base = rgb * (1 - desaturate) + gray * desaturate
    col = torch.tensor(color, dtype=torch.float32).view(3, 1, 1) / 255.0
    out = base * (1 - m * alpha) + col * (m * alpha)
    return to_pil(out)


def overlay_two(rgb, mask_a, mask_b, alpha=0.75, desaturate=0.8, width=None):
    """Red = mask_a only, green = mask_b only, yellow = both."""
    if rgb.ndim == 4:
        rgb = rgb[0]
    a = (mask_a[0] if mask_a.ndim == 4 else mask_a)[0:1].detach().float()
    b = (mask_b[0] if mask_b.ndim == 4 else mask_b)[0:1].detach().float()
    a = _resize(a, width).clamp(0, 1).cpu()
    b = _resize(b, width).clamp(0, 1).cpu()
    rgb = _resize(rgb, width).detach().float().clamp(0, 1).cpu()
    gray = rgb.mean(dim=0, keepdim=True).repeat(3, 1, 1)
    base = rgb * (1 - desaturate) + gray * desaturate
    col = torch.cat([a, b, torch.zeros_like(a)], dim=0)          # R=old, G=new
    m = torch.maximum(a, b)
    out = base * (1 - m * alpha) + col * (m * alpha)
    return to_pil(out)


def label(img, text, height=26, bg=(24, 24, 28), fg=(235, 235, 240)):
    w, h = img.size
    out = Image.new("RGB", (w, h + height), bg)
    out.paste(img, (0, height))
    d = ImageDraw.Draw(out)
    d.text((6, 4), text, fill=fg, font=_font(16))
    return out


def grid(rows, pad=6, bg=(16, 16, 18)):
    """rows: list[list[PIL.Image]] -> one sheet"""
    col_w = [0] * max(len(r) for r in rows)
    row_h = []
    for r in rows:
        row_h.append(max(im.size[1] for im in r))
        for i, im in enumerate(r):
            col_w[i] = max(col_w[i], im.size[0])
    W = sum(col_w) + pad * (len(col_w) + 1)
    H = sum(row_h) + pad * (len(rows) + 1)
    sheet = Image.new("RGB", (W, H), bg)
    y = pad
    for r, rh in zip(rows, row_h):
        x = pad
        for i, im in enumerate(r):
            sheet.paste(im, (x, y))
            x += col_w[i] + pad
        y += rh + pad
    return sheet


def fit(img, width):
    if img.size[0] == width:
        return img
    h = max(1, round(img.size[1] * width / img.size[0]))
    return img.resize((width, h), Image.LANCZOS)
