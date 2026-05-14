"""
RMS Norm reference implementation. All scenarios compare against this.

  Y = X * rsqrt(mean(X^2, -1) + eps) * W            # plain
  Y = (X + R) * rsqrt(mean((X+R)^2, -1) + eps) * W  # residual
  Y = X * rsqrt(mean(X^2, -1) + eps) * W + B        # affine (with bias)

The "plain" variant is the default; others are opt-in via the workload spec.
"""

from __future__ import annotations

import torch


def rmsnorm_plain(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    y = x_fp32 * torch.rsqrt(var + eps)
    y = y * w.to(torch.float32)
    return y.to(orig_dtype)


def rmsnorm_residual(
    x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return rmsnorm_plain(x + r, w, eps)


def rmsnorm_affine(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    y = x_fp32 * torch.rsqrt(var + eps)
    y = y * w.to(torch.float32) + b.to(torch.float32)
    return y.to(orig_dtype)


def dispatch_reference(variant: str, *args, **kwargs) -> torch.Tensor:
    if variant == "plain":
        return rmsnorm_plain(*args, **kwargs)
    if variant == "residual":
        return rmsnorm_residual(*args, **kwargs)
    if variant == "affine":
        return rmsnorm_affine(*args, **kwargs)
    raise ValueError(f"unknown variant: {variant}")
