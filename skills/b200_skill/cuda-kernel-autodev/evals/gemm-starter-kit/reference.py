"""
GEMM (fused matmul + bias + GELU) reference implementation.

  Y = GELU(X @ W.T + bias)

Shapes:
  X    [M, K]    input activations
  W    [N, K]    weight matrix (row-major, N rows)
  bias [N]       per-output bias
  Y    [M, N]    output

The naive GEMM kernel variant="plain" computes the matmul only (no bias, no GELU).
The "fused" variant adds bias and GELU inside the kernel (no extra pass).
"""

from __future__ import annotations

import torch


def gemm_plain(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    return (x.to(torch.float32) @ w.to(torch.float32).t()).to(orig_dtype)


def gemm_fused(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    y = x.to(torch.float32) @ w.to(torch.float32).t()
    y = y + bias.to(torch.float32)
    y = torch.nn.functional.gelu(y, approximate="tanh")
    return y.to(orig_dtype)


def dispatch_reference(variant: str, *args, **kwargs) -> torch.Tensor:
    # The shared harness passes args as (x, w, [b|r]) depending on variant.
    # For GEMM: variant="plain" → matmul only; variant="fused" → + bias + GELU.
    if variant == "plain":
        x, w = args[0], args[1]
        return gemm_plain(x, w)
    if variant == "fused":
        # harness passes (x, weight, b) for affine variant
        x, w, b = args[0], args[1], args[2]
        return gemm_fused(x, w, b)
    raise ValueError(f"unknown variant: {variant}")
