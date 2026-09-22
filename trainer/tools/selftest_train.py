r"""
Self-test for the training stack: model, CAME, dataset, monitor.

Separate from selftest.py (which covers the warping/mask geometry and needs no
model code) because this one imports iw3 and torch models.

  selftest_train.bat
"""
from __future__ import annotations

import os
import sys
import tempfile
from os import path

_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch                                                     # noqa: E402

import iw3.models                                                # noqa: E402,F401
from nunif.models import create_model, save_model, load_model    # noqa: E402
import ntrainer.models                                           # noqa: E402,F401
from ntrainer.models.inpaint_v2 import PRESETS                   # noqa: E402
from ntrainer.optim import CAME                                  # noqa: E402


def check(ok_all, cond, msg):
    print(f"[{'ok' if cond else 'FAIL'}] {msg}")
    return ok_all and cond


def main():
    ok = True
    dev = "cpu"
    print(f"torch={torch.__version__}")

    # ---- model -----------------------------------------------------------
    x = torch.rand(1, 3, 128, 128)
    mask = torch.zeros(1, 1, 128, 128)
    mask[:, :, 20:100, 40:80] = 1

    for preset in PRESETS:
        name = f"inpaint.nt_inpaint_v2_{preset}"
        torch.manual_seed(0)
        net = create_model(name).to(dev)
        net.train()
        with torch.no_grad():
            z = net(*net.preprocess(x, mask))
        std = z.std().item()
        params = sum(p.numel() for p in net.parameters())
        # The regression this guards: nunif's GMLP already adds its own
        # shortcut, so `x + gmlp(x)` doubles the activation scale every block.
        # With 10-19 blocks that reached std 5e4 at init, which overflows fp16.
        ok = check(ok, std < 5.0,
                   f"{name}: {params / 1e6:.2f}M params, init output std {std:.3f} (< 5)")

    net = create_model("inpaint.nt_inpaint_v2_b")
    net.eval()
    ok = check(ok, net.i2i_offset == 16 and net.i2i_scale == 1,
               f"offset {net.i2i_offset} / scale {net.i2i_scale} match iw3's tiling")
    with torch.no_grad():
        out_train_path = net(*net.preprocess(x, mask))
        out_infer = net.infer(x, mask)
    ok = check(ok, tuple(out_train_path.shape[-2:]) == (96, 96),
               f"train path crops the offset: 128 -> {tuple(out_train_path.shape[-2:])}")
    ok = check(ok, tuple(out_infer.shape) == tuple(x.shape),
               f"infer keeps the input size: {tuple(out_infer.shape)}")

    odd = torch.rand(1, 3, 173, 291)
    odd_m = (torch.rand(1, 1, 173, 291) > 0.85).float()
    with torch.no_grad():
        z = net.infer(odd, odd_m)
    ok = check(ok, tuple(z.shape) == tuple(odd.shape),
               f"non-aligned input padded and unpadded: {tuple(z.shape)}")

    pad = (-net.align) % net.align
    ok = check(ok, pad == 0,
               f"aligned input is not padded by a whole extra {net.align}px block")

    with torch.no_grad():
        keep = net.infer(x, torch.zeros_like(mask))
    ok = check(ok, torch.allclose(keep, x, atol=1e-5),
               "empty mask is a no-op (output == input)")

    with tempfile.TemporaryDirectory() as d:
        p = path.join(d, "m.pth")
        save_model(net, p)
        net2, meta = load_model(p)
        same = (net2.name == net.name
                and all(torch.equal(a, b) for a, b in zip(net.state_dict().values(),
                                                          net2.state_dict().values())))
    ok = check(ok, same, "save_model/load_model round trip through nunif's registry")

    # ---- CAME ------------------------------------------------------------
    # The gist copy of CAME ends the non-factored branch with `update = exp_avg`
    # and then does `update.mul_(lr)`, scaling the momentum buffer in place.
    # Every 1-D parameter then trains at ~lr/10 with no momentum.
    w = torch.nn.Parameter(torch.randn(64))          # 1-D -> non-factored
    opt = CAME([w], lr=1e-2)
    (w.sum()).backward()
    opt.step()
    ea = opt.state[w]["exp_avg"]
    ok = check(ok, ea.abs().max().item() > 1e-3,
               f"CAME does not scale exp_avg by lr on 1-D params "
               f"(max |exp_avg| {ea.abs().max().item():.4f})")

    p2 = torch.nn.Parameter(torch.randn(8, 4, 3, 3))
    opt2 = CAME([p2], lr=1e-3)
    p2.sum().backward()
    opt2.step()
    st = opt2.state[p2]
    ok = check(ok, "exp_avg_sq_row" in st and "exp_avg_sq" not in st,
               "CAME factors the second moment for ndim>=2, full for 1-D")

    torch.manual_seed(3)
    w = torch.nn.Parameter(torch.randn(64, 64))
    target = torch.randn(64, 64)
    opt = CAME([w], lr=1e-2)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = ((w - target) ** 2).mean()
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    ok = check(ok, loss.item() < first * 0.01,
               f"CAME minimises a quadratic: {first:.3f} -> {loss.item():.5f}")

    sd = opt.state_dict()
    w2 = torch.nn.Parameter(torch.randn(64, 64))
    opt3 = CAME([w2], lr=1e-2)
    opt3.load_state_dict(sd)
    ok = check(ok, opt3.state[w2]["step"] == 200, "CAME state_dict round trip (resume works)")

    scaler = torch.amp.GradScaler("cpu", enabled=False)
    opt.zero_grad()
    ((w - target) ** 2).mean().backward()
    scaler.step(opt)
    scaler.update()
    ok = check(ok, True, "GradScaler.step(CAME) runs (AMP path)")

    # CAME does not bias-correct its confidence term, so without warmup the very
    # first step moves ~11x further than the RMS clip intends. That blew a real
    # run's training loss from 0.11 to 116 in 15 epochs. Guard both directions.
    def first_step_ratio(warmup):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(128, 128) * 0.05)
        o = CAME([p], lr=8e-5, warmup_steps=warmup)
        before = p.detach().clone()
        o.zero_grad()
        p.grad = torch.randn_like(p) * 0.01
        o.step()
        d = p.detach() - before
        return (d.norm() / (d.numel() ** 0.5)).item() / 8e-5

    raw, warm = first_step_ratio(0), first_step_ratio(1000)
    ok = check(ok, raw > 5.0,
               f"unwarmed CAME really does overshoot on step 1 ({raw:.1f}x lr) -- "
               f"this is why warmup_steps exists")
    ok = check(ok, warm < 0.05,
               f"warmup_steps=1000 holds step 1 to {warm:.3f}x lr")

    # ---- dataset ---------------------------------------------------------
    from ntrainer.train.dataset import NTInpaintDataset, check_dataset
    from PIL import Image
    import numpy as np
    with tempfile.TemporaryDirectory() as d:
        for split in ("train", "eval"):
            sub = path.join(d, split)
            os.makedirs(sub)
            for i in range(4):
                im = np.random.default_rng(i).integers(0, 255, (512, 512, 3), dtype=np.uint8)
                m = np.zeros((512, 512), np.uint8)
                m[100:400, 200:240] = 255          # a single narrow band
                Image.fromarray(im).save(path.join(sub, f"s{i}_C.png"))
                Image.fromarray(m).save(path.join(sub, f"s{i}_M.png"))
        check_dataset(d, 256)
        ds = NTInpaintDataset(path.join(d, "train"), model_offset=16, training=True, size=256)
        hits = 0
        for _ in range(20):
            _, mk, _, _ = ds[0]
            hits += int(mk.sum() > 0)
        ok = check(ok, hits == 20,
                   f"training crops always contain mask ({hits}/20) -- upstream picks by "
                   f"colour variance and would often miss it entirely")
        xs, mk, ys, _ = ds[0]
        ok = check(ok, mk.dtype == torch.bool and tuple(ys.shape[-2:]) == (224, 224),
                   f"dataset returns bool mask and offset-cropped target {tuple(ys.shape[-2:])}")

        ev = NTInpaintDataset(path.join(d, "eval"), model_offset=16, training=False, size=256)
        a = ev[0][0]
        b = ev[0][0]
        ok = check(ok, torch.equal(a, b),
                   "eval crops are deterministic (epochs stay comparable)")

        try:
            check_dataset(path.join(d, "nope"), 256)
            bad = False
        except RuntimeError:
            bad = True
        ok = check(ok, bad, "check_dataset rejects a missing split up front")


    # ---- video model ------------------------------------------------------
    from ntrainer.models import transfer_image_weights, SEQ_LEN
    from ntrainer.train.trainer import TemporalSmoothingPenalty

    vx = torch.rand(SEQ_LEN, 3, 128, 128)
    vm = torch.zeros(SEQ_LEN, 1, 128, 128)
    vm[:, :, 20:100, 40:80] = 1
    for preset in PRESETS:
        name = f"inpaint.nt_video_inpaint_v2_{preset}"
        torch.manual_seed(0)
        vnet = create_model(name)
        vnet.train()
        with torch.no_grad():
            z = vnet(*vnet.preprocess(vx, vm))
        n = sum(p.numel() for p in vnet.parameters())
        ok = check(ok, z.std().item() < 5.0,
                   f"{name}: {n / 1e6:.2f}M params, init output std {z.std().item():.3f}")

    vnet = create_model("inpaint.nt_video_inpaint_v2_s").eval()

    def cross_frame_delta(net):
        with torch.no_grad():
            a = net(*net.preprocess(vx, vm))
            x2 = vx.clone()
            x2[0] = torch.rand(3, 128, 128)
            b = net(*net.preprocess(x2, vm))
        d = (a - b).abs().flatten(1).max(1).values
        return d[1:].max().item()

    with torch.no_grad():
        for blk in list(vnet.temporal2) + list(vnet.temporal3):
            torch.nn.init.normal_(blk.gmlp.gmlp.proj_spatial.weight, 0, 0.05)
    vd = cross_frame_delta(vnet)
    id_ = cross_frame_delta(create_model("inpaint.nt_inpaint_v2_s").eval())
    ok = check(ok, vd > 1e-3,
               f"video model mixes across frames: changing frame 0 moves other frames "
               f"by {vd:.4f}")
    ok = check(ok, id_ == 0.0,
               f"image model cannot: same test gives exactly {id_}")

    n_loaded, _ = transfer_image_weights(create_model("inpaint.nt_video_inpaint_v2_b"),
                                         create_model("inpaint.nt_inpaint_v2_b").state_dict(),
                                         verbose=False)
    ok = check(ok, n_loaded > 150,
               f"image weights transfer into the video model ({n_loaded} tensors)")
    try:
        transfer_image_weights(create_model("inpaint.nt_video_inpaint_v2_b"),
                               create_model("inpaint.nt_inpaint_v2_s").state_dict(),
                               verbose=False)
        ok = check(ok, False, "mismatched transfer should have raised")
    except RuntimeError:
        ok = check(ok, True, "a mismatched transfer raises instead of training from noise")

    for n in (SEQ_LEN, 7, 25):
        with torch.no_grad():
            z = vnet.infer(torch.rand(n, 3, 128, 128), (torch.rand(n, 1, 128, 128) > 0.85).float())
        ok = check(ok, z.shape[0] == n, f"infer() pads and trims {n} frames -> {z.shape[0]}")

    # The upstream penalty indexes the channel axis, so it scores smooth motion
    # 42.2 and maximum flicker 0.0 -- backwards. Ours must do the opposite.
    pen = TemporalSmoothingPenalty()
    smooth = torch.zeros(12, 3, 8, 8)
    smooth[:, 0] = torch.arange(12).view(12, 1, 1)
    flicker = torch.zeros(12, 3, 8, 8)
    flicker[::2] = 1.0
    chan = torch.zeros(12, 3, 8, 8)
    chan[:, 1] = 1.0
    ok = check(ok, pen(smooth).item() == 0.0 and pen(flicker).item() > 1.0
               and pen(chan).item() == 0.0,
               f"temporal penalty: smooth={pen(smooth).item():.2f} "
               f"flicker={pen(flicker).item():.2f} channels-only={pen(chan).item():.2f}")

    # ---- monitor ---------------------------------------------------------
    sys.path.insert(0, path.dirname(path.abspath(__file__)))
    import monitor
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(path.join(d, "eval"))
        with open(path.join(d, "progress.csv"), "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,eval_loss,lr,weight_decay,seconds,timestamp,preview\n")
            f.write("1,0.5,0.6,8e-05,0.001,12.3,2026-01-01T00:00:00,epoch0001_1.png\n")
            f.write("2,0.4,,8e-05,0.001,12.0,2026-01-01T00:01:00,\n")
        for n in ("epoch0001_1.png", "epoch0002_1.png", "epoch0002_2.png"):
            open(path.join(d, "eval", n), "wb").close()
        rows = monitor.read_progress(d)
        by_epoch, slots = monitor.read_previews(d)
        ok = check(ok, len(rows) == 2 and rows[1]["eval"] is None and rows[0]["eval"] == 0.6,
                   "monitor reads progress.csv, blank eval stays blank")
        ok = check(ok, sorted(by_epoch) == [1, 2] and slots == [1, 2],
                   f"monitor indexes previews by epoch {sorted(by_epoch)} and slot {slots}")
        st = monitor.state(d)
        ok = check(ok, st["preview_epochs"] == [1, 2], "monitor state() serialises")
        ok = check(ok, monitor.EVAL_RE.match("../progress.csv") is None,
                   "monitor refuses anything but epochNNNN_i.png (no path traversal)")

    print("\n" + ("ALL OK" if ok else "*** SOME CHECKS FAILED ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
