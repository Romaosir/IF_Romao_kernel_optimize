"""
Softmax reference implementation. All scenarios compare against this.

  Y[i, :] = exp(X[i, :] - max(X[i, :])) / sum(exp(X[i, :] - max(X[i, :])))

Row-wise softmax over the last dim. fp32 accumulator internally.
"""

from __future__ import annotations

import torch


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    x_max = x_fp32.max(dim=-1, keepdim=True).values
    e = torch.exp(x_fp32 - x_max)
    y = e / e.sum(dim=-1, keepdim=True)
    return y.to(orig_dtype)


def dispatch_reference(variant: str, *args, **kwargs) -> torch.Tensor:
    # softmax only has one variant; the first positional arg is x.
    # Accept extra positional/kwargs so the shared harness can pass (x, w, eps)
    # style tuples without a signature mismatch.
    if variant == "plain":
        x = args[0]
        return softmax_ref(x)
    raise ValueError(f"unknown variant: {variant}")
