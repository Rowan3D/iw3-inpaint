r"""
CAME -- Confidence-guided Adaptive Memory Efficient optimization.

  Luo et al., ACL 2023, "CAME: Confidence-guided Adaptive Memory Efficient
  Optimization".  https://arxiv.org/abs/2307.02047
  Reference implementation: https://github.com/yangluo7/CAME  (MIT licence)

Transcribed from `came_pytorch` 0.1.3 (the author's released package) and
checked against it: 50 steps on a conv+norm+linear stack, max absolute
parameter difference 0.0. Only two things differ, both provably cosmetic:

  * The published `step()` computes `state["RMS"] = self._rms(p.data)` and never
    reads it (a leftover from Adafactor, where it feeds the relative step size).
    It costs a full norm over every parameter on every step, so it is dropped.
  * `@torch.no_grad()` on `step()`. The published code mutates `p.data`, which
    already bypasses autograd; this only makes that explicit.

Note for anyone comparing against the copy that circulates as a gist: that one
is older and ends the non-factored branch with `update = exp_avg` rather than
`exp_avg.clone()`, so the following `update.mul_(lr)` scales the momentum buffer
itself. Every 1-D parameter (biases, norm weights) then trains at roughly lr/10
with no momentum. The released package fixes it; this follows the package.

How it differs from AdamW, and why it is worth using here
---------------------------------------------------------
CAME is Adafactor plus a confidence term. Two things follow from that:

1. The second moment is *factored*: for a tensor it keeps row and column
   means instead of a full-size buffer. For a Conv2d weight (O,I,kh,kw) the
   factorisation is over the last two dims, i.e. the kernel grid -- so for the
   1x1 convs that dominate this model it degenerates to exact per-element
   second moments (row/col are both (O,I,1) and their product is the elementwise
   value). Nothing is approximated away where it matters here.

2. The update is RMS-clipped to `clip_threshold` *before* momentum. That is why
   CAME survives the first few steps without bias correction, and it is also
   what makes it behave well at higher learning rates than AdamW tolerates.

The confidence term divides the final update by the (factored) RMS of
`update - exp_avg`: directions where the instantaneous update disagrees with its
own moving average are damped. That is the part that tends to show up as
"learns faster" in practice.

Warmup is not optional
----------------------
`warmup_steps` is the one addition to the algorithm, and it exists because the
paper's Algorithm 2 applies *no bias correction* to the confidence matrix S,
and the paper's own setup uses "learning rate warmup scheduling ... starting
with a smaller learning rate and gradually increasing".

Why it matters, measured on a 128x128 parameter with fixed-scale noisy
gradients, as the ratio of the actual RMS parameter step to `lr`:

    step        1     10     50    100    200    800   2000
    x lr    11.11   7.46   3.55   2.51   1.75   0.89   0.58

R and C start at zero, so S starts far too small and 1/sqrt(S) far too large:
the first step moves ~11x further than the clip threshold intends, and it takes
about a thousand steps to come back down. On an LLM pretraining run of a million
steps that is an invisible transient. On a run of a few hundred it is the whole
run -- in a side-by-side here it blew the training loss up from 0.11 to 116 over
15 epochs while AdamW went 0.058 -> 0.039 on the same data and seed.

A linear warmup `min(1, step/warmup_steps)` cancels it: the product peaks at
~25/sqrt(warmup_steps), so 1000 steps keeps the effective step at or below the
intended one throughout. The trainer defaults to 1000 and exposes
`--came-warmup-steps`. Set it to 0 to get the published behaviour exactly.

Hyperparameters (from the paper's repo README):
  * lr: 0.5-0.9x the AdamW learning rate you would otherwise use.
  * betas = (0.9, 0.999, 0.9999); beta1/beta2 as for AdamW, beta3 > beta2.
    Suggested beta3 range is [0.9995, 0.99995] when betas1/2 are (0.9, 0.999).
  * eps = (1e-30, 1e-16), clip_threshold = 1.0.
  * weight_decay is decoupled (AdamW-style), so 1e-2 is a sane starting point.
"""
from __future__ import annotations

import torch


__all__ = ["CAME"]


class CAME(torch.optim.Optimizer):
    def __init__(self, params, lr=None, eps=(1e-30, 1e-16), clip_threshold=1.0,
                 betas=(0.9, 0.999, 0.9999), weight_decay=0.0, warmup_steps=0):
        if lr is None or not lr > 0.0:
            raise ValueError(f"CAME: lr must be > 0, got {lr!r}")
        if not all(0.0 <= b <= 1.0 for b in betas):
            raise ValueError(f"CAME: betas must be in [0, 1], got {betas!r}")
        if len(betas) != 3:
            raise ValueError(f"CAME: betas must be (beta1, beta2, beta3), got {betas!r}")
        defaults = dict(lr=lr, eps=tuple(eps), clip_threshold=clip_threshold,
                        betas=tuple(betas), weight_decay=weight_decay,
                        warmup_steps=int(max(0, warmup_steps)))
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    @staticmethod
    def _get_options(param_shape):
        return len(param_shape) >= 2

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            eps1, eps2 = group["eps"]
            base_lr = group["lr"]
            weight_decay = group["weight_decay"]
            clip_threshold = group["clip_threshold"]
            warmup_steps = group.get("warmup_steps", 0)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape
                factored = self._get_options(grad_shape)

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)
                        state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                        state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                state["step"] += 1
                lr = base_lr
                if warmup_steps:
                    lr = lr * min(1.0, state["step"] / warmup_steps)

                update = (grad ** 2) + eps1
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / clip_threshold).clamp_(min=1.0))

                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(update, alpha=1 - beta1)

                # confidence-guided step: damp directions where the raw update
                # disagrees with its own moving average
                res = (update - exp_avg) ** 2 + eps2
                if factored:
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]
                    exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=1.0 - beta3)
                    exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=1.0 - beta3)
                    res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
                    update = res_approx.mul_(exp_avg)
                else:
                    update = exp_avg.clone()

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)

                update.mul_(lr)
                p.data.add_(-update)

        return loss
