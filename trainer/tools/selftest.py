"""
Correctness self-test for ntrainer.geometry -- no depth model, no images needed.
Verifies the analytic mask against hand-computable ground truth, including at
divergences where iw3's iterative warp breaks down.
"""
from __future__ import annotations

import os
import sys
from os import path

_HERE = path.dirname(path.dirname(path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402

from ntrainer.geometry import disocclusion_mask, dibr_warp, fill_along_x, shift_scale  # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} torch={torch.__version__}")
    ok = True

    # 1. single step edge -> band must start exactly on the first background px
    W = 300
    d = torch.full((1, 1, 1, W), 0.2, device=dev)
    d[..., :100] = 0.8
    div = 100.0 / (0.005 * W)                                   # S == 100px
    m = disocclusion_mask(d, div, 0.0, base_size=W, soft=False)
    idx = m[0, 0, 0].nonzero().flatten()
    good = idx.min().item() == 100 and idx.numel() == 60
    ok &= good
    print(f"[{'ok' if good else 'FAIL'}] step edge: band=[{idx.min().item()},{idx.max().item() + 1}) "
          f"expected [100,160)")

    # 2. no iteration cap: coherent single band at any divergence
    W = 1920
    d = torch.full((1, 1, 1, W), 0.10, device=dev)
    d[..., :800] = 0.95
    for div in (5, 9, 16, 24, 40):
        m = disocclusion_mask(d, div, 0.0, base_size=W, soft=False)
        idx = m[0, 0, 0].nonzero().flatten()
        contiguous = idx.numel() > 0 and (idx.max() - idx.min() + 1).item() == idx.numel()
        expect = 0.85 * shift_scale(div, W)
        good = contiguous and idx.min().item() == 800 and abs(idx.numel() - expect) <= 1.5
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] div={div:3} shift={shift_scale(div, W):6.1f}px "
              f"band start={idx.min().item()} width={idx.numel()} expected~{expect:.1f} "
              f"contiguous={contiguous}")

    # 3. left view == mirror of right view
    W = 200
    d = torch.rand(1, 1, 1, W, device=dev)
    d = F.avg_pool1d(d.view(1, 1, W), 7, stride=1, padding=3).view(1, 1, 1, W)
    div = 50 / (0.005 * W)
    mr = disocclusion_mask(d, div, 0.5, base_size=W, view="right")
    ml = disocclusion_mask(d.flip(-1), div, 0.5, base_size=W, view="left")
    diff = (mr - ml.flip(-1)).abs().max().item()
    ok &= diff < 1e-5
    print(f"[{'ok' if diff < 1e-5 else 'FAIL'}] left/right mirror symmetry: max diff {diff:.2e}")

    # 4. background-preferring unbounded fill
    v = torch.tensor([[[[1., 2., 3., 0., 0., 0., 7., 8.]]]], device=dev)
    valid = torch.tensor([[[[1, 1, 1, 0, 0, 0, 1, 1]]]], device=dev).bool()
    dk = torch.tensor([[[[.9, .9, .9, 2., 2., 2., .1, .1]]]], device=dev)
    f = fill_along_x(v, valid, dk, prefer="background")[0, 0, 0].tolist()
    good = f[3] == 7.0 and f[5] == 7.0
    ok &= good
    print(f"[{'ok' if good else 'FAIL'}] background fill: {f}")

    # 5. warp: target holes agree with the analytic source mask
    W, H = 1920, 256
    d = torch.full((1, 1, H, W), 0.15, device=dev)
    d[..., 600:1100] = 0.85
    rgb = torch.rand(1, 3, H, W, device=dev)
    for div in (5, 16, 24):
        _, hole = dibr_warp(rgb, d, div, 0.0, base_size=W)
        src = disocclusion_mask(d, div, 0.0, base_size=W, soft=False)
        a, b = hole.sum().item() / H, src.sum().item() / H
        expect = 0.70 * shift_scale(div, W)
        good = abs(a - expect) <= 2.0 and abs(b - expect) <= 2.0
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] div={div:3} warp holes={a:6.1f}px/row "
              f"analytic mask={b:6.1f}px/row expected={expect:6.1f}")

    # 6. speed at 1080p
    if dev == "cuda":
        d = torch.rand(1, 1, 1080, 1920, device=dev)
        rgb = torch.rand(1, 3, 1080, 1920, device=dev)
        import time
        for fn, name in ((lambda: disocclusion_mask(d, 16, 0.5, base_size=1920), "disocclusion_mask"),
                         (lambda: dibr_warp(rgb, d, 16, 0.5, base_size=1920), "dibr_warp")):
            fn(); torch.cuda.synchronize()
            t = time.time()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            print(f"     {name}: {(time.time() - t) / 20 * 1000:.2f} ms @1920x1080 div=16")


    # 7. real-world case: a blurred depth edge must still report the FULL hole
    #    width (this is the failure the "envelope" method exists to fix)
    W = 512
    d = torch.zeros(1, 1, 1, W, device=dev)
    d[..., :300] = 0.9
    d_ramp = F.avg_pool1d(d.view(1, 1, W), 11, stride=1, padding=5).view(1, 1, 1, W)
    for div in (2, 4, 8, 16, 24):
        expect = 0.9 * shift_scale(div, W)
        bg = disocclusion_mask(d_ramp, div, 0.5, base_size=W, soft=False,
                               anchor="background").sum().item()
        edge = disocclusion_mask(d_ramp, div, 0.5, base_size=W, soft=False,
                                 anchor="edge").sum().item()
        col = disocclusion_mask(d_ramp, div, 0.5, base_size=W, soft=False,
                                method="column").sum().item()
        # anchor="background" reproduces the hole width; anchor="edge" adds the
        # 11px ramp so the mask touches the silhouette instead of leaving a gap
        good = (abs(bg - expect) <= max(2.0, 0.08 * expect)
                and 8 <= (edge - bg) <= 13)
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] blurred edge div={div:3}: "
              f"anchor=bg {bg:5.0f}px  anchor=edge {edge:5.0f}px  "
              f"per-column {col:5.0f}px  true hole {expect:5.1f}px")

    # 7b. anchor="edge" must reach the silhouette on a perfectly sharp edge too
    d_step = torch.zeros(1, 1, 1, W, device=dev)
    d_step[..., :300] = 0.9
    m = disocclusion_mask(d_step, 16, 0.5, base_size=W, soft=False, anchor="edge")[0, 0, 0]
    nz = m.nonzero().flatten()
    good = nz.min().item() == 300
    ok &= good
    print(f"[{'ok' if good else 'FAIL'}] sharp edge, anchor=edge: band starts at "
          f"{nz.min().item()} (object edge is 300)")

    # 8. a gently sloped surface is a stretch, not a hole
    d_slope = torch.linspace(0.9, 0.0, W, device=dev).view(1, 1, 1, W)
    s_mask = disocclusion_mask(d_slope, 16, 0.5, base_size=W, soft=False).sum().item()
    ok &= s_mask == 0
    print(f"[{'ok' if s_mask == 0 else 'FAIL'}] gentle slope produces no hole: {s_mask:.0f}px")

    # 9. occlusion: a band is cut where a nearer surface would cover it
    d_bar = torch.zeros(1, 1, 1, W, device=dev)
    d_bar[..., :100] = 0.9
    d_bar[..., 120:130] = 0.95
    m = disocclusion_mask(d_bar, 16, 0.5, base_size=W, soft=False)[0, 0, 0]
    nz = m.nonzero().flatten().tolist()
    cut = 120 not in nz and 119 in nz
    ok &= cut
    print(f"[{'ok' if cut else 'FAIL'}] occluded band truncated at the bar (x=120)")


    # 10. the real-world "broken bands" case: a moderate edge that the depth
    #     model has blurred over ~9px, with depth noise on top.  Its PER-COLUMN
    #     drop is only ~0.018, so a per-column threshold either misses it
    #     entirely or fires intermittently; the band must still come out whole.
    torch.manual_seed(0)
    W, H = 1024, 256
    yy = torch.linspace(0, 6.28, H, device=dev).view(1, 1, H, 1)
    xx = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, 1, W)
    edge = 400 + torch.sin(yy) * 40                          # wavy silhouette
    d = torch.where(xx < edge, torch.tensor(0.45, device=dev), torch.tensor(0.30, device=dev))
    d = d + torch.randn(1, 1, H, W, device=dev) * 0.004
    d = F.avg_pool2d(F.pad(d, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
    per_col = (d[..., :-1] - d[..., 1:]).amax().item()
    for div in (5, 8, 16):
        m = disocclusion_mask(d, div, 0.5, base_size=W, soft=False)
        starts = ((m[..., 1:] - m[..., :-1]).clamp_min(0).sum() + m[..., 0].sum()).item()
        runs = starts / H
        cov = m.sum().item() / H
        hole = 0.15 * shift_scale(div, W)
        good = 0.9 <= runs <= 1.15 and cov >= hole
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] blurred+noisy edge div={div:3}: "
              f"runs/row={runs:.2f} (want 1.00)  masked={cov:.1f}px/row "
              f"(hole {hole:.1f} + ~9px ramp)  per-column drop only {per_col:.4f}")


    # 11. single-pixel depth spikes must not anchor a band inside the object,
    #     and the vertical median must straighten the outer boundary
    torch.manual_seed(1)
    W, H = 1920, 400
    lw, lh = 784, int(400 * 784 / 1920)
    yy = torch.linspace(0, 6.28, lh, device=dev).view(1, 1, lh, 1)
    xx = torch.arange(lw, device=dev, dtype=torch.float32).view(1, 1, 1, lw)
    e = 300 + torch.sin(yy * 1.5) * 40
    dl = torch.where(xx < e, torch.tensor(0.70, device=dev), torch.tensor(0.25, device=dev))
    dl = dl + torch.randn(1, 1, lh, lw, device=dev) * 0.003
    d = F.interpolate(dl, size=(H, W), mode="bilinear", align_corners=False)
    d[0, 0, torch.arange(5, H, 17, device=dev), 1150] = 0.95     # spikes in the object

    def _edges(**kw):
        m = (disocclusion_mask(d, 16, 0.5, base_size=W, soft=False, **kw) > 0.5).float()
        idx = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, 1, W)
        first = torch.where(m > 0.5, idx, torch.full_like(idx, W)).amin(-1)[0, 0]
        last = torch.where(m > 0.5, idx, torch.full_like(idx, -1.0)).amax(-1)[0, 0]
        v = first < W
        b = v[1:] & v[:-1]
        return (first[1:] - first[:-1]).abs()[b].mean().item(), \
               (last[1:] - last[:-1]).abs()[b].mean().item(), first.min().item()

    # the shipped default config is what must be clean, however it gets there
    i, o, f = _edges()
    good = i < 3.0 and o < 3.0 and f > 500
    ok &= good
    print(f"[{'ok' if good else 'FAIL'}] default config on spiked depth: inner jitter "
          f"{i:.2f}px, outer jitter {o:.2f}px, leftmost mask col {f:.0f} (want <3, <3, >500)")

    # what each guard contributes, with the tear_px pixel criterion re-enabled
    # (that criterion is what made spikes destructive in the first place)
    iL, oL, fL = _edges(tear_px=2.0, despike=0, outer_smooth_rows=0)
    iD, oD, fD = _edges(tear_px=2.0, despike=1, outer_smooth_rows=0)
    iV, oV, fV = _edges(tear_px=2.0)
    print(f"     tear_px=2, no guards : inner {iL:5.2f}px outer {oL:5.2f}px leftmost col {fL:.0f}")
    print(f"     + horizontal median  : inner {iD:5.2f}px outer {oD:5.2f}px leftmost col {fD:.0f}")
    print(f"     + vertical median    : inner {iV:5.2f}px outer {oV:5.2f}px leftmost col {fV:.0f}")


    # 12. every tool must go through ntrainer.pipeline, never call the mask
    #     primitives directly.  This is not pedantry: compare_masks and
    #     make_dataset each had their own copy of the sequence, drifted apart,
    #     and make_dataset silently shipped worse masks (different bilateral
    #     settings plus a depth mapper that was never in the validated path).
    import re
    tools_dir = path.dirname(path.abspath(__file__))
    offenders = []
    for fn in sorted(os.listdir(tools_dir)):
        if not fn.endswith(".py") or fn == "selftest.py":
            continue
        src = open(path.join(tools_dir, fn), encoding="utf-8").read()
        for bad in ("disocclusion_mask(", "refine_depth("):
            if re.search(r"(?<![.\w])" + re.escape(bad), src):
                offenders.append(f"{fn} calls {bad}")
    good = not offenders
    ok &= good
    print(f"[{'ok' if good else 'FAIL'}] tools use ntrainer.pipeline, not the mask "
          f"primitives directly" + ("" if good else f" -- {offenders}"))

    # the validated defaults must stay put
    from ntrainer.pipeline import MaskConfig, MAPPER_SETS
    cfg = MaskConfig()
    expect = dict(bilateral_radius=3, bilateral_iter=1, depth_step=0.02, tear_px=0.0,
                  sharpness=0.6, edge_frac=0.1, despike=2, smooth_px=1, anchor="edge")
    drift = {k: (getattr(cfg, k), v) for k, v in expect.items() if getattr(cfg, k) != v}
    ok &= not drift
    print(f"[{'ok' if not drift else 'FAIL'}] MaskConfig defaults unchanged"
          + ("" if not drift else f" -- drifted: {drift}"))
    good = "mul_1" not in MAPPER_SETS["safe"] and "mul_3" not in MAPPER_SETS["safe"]
    ok &= good
    print(f"[{'ok' if good else 'FAIL'}] mul_* excluded from the default mapper set "
          f"(measured 2.1px edge jitter vs 1.0px)")

    print("\n" + ("ALL OK" if ok else "*** SOME CHECKS FAILED ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
