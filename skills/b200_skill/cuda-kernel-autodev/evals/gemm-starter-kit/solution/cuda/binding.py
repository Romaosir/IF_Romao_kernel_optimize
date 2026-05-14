"""
GEMM binding — JIT-compiles kernel.cu and exposes an `rmsnorm(...)` entry
for harness compatibility. For GEMM:
  - x = X [M, K]
  - w = W [N, K]
  - variant = "plain"  -> Y = X @ W.T
  - variant = "fused"  -> Y = GELU(X @ W.T + bias);  bias is passed as `b`
  - eps, r are ignored.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load as _load_extension

_HERE = Path(__file__).resolve().parent
_ARCH = os.environ.get("GEMM_ARCH", "sm_100")
_EXTRA_CFLAGS = ["-O3", "-std=c++17"]
_EXTRA_CUDA_CFLAGS = [
    "-O3", "-std=c++17",
    f"-gencode=arch=compute_{_ARCH.replace('sm_','')},code={_ARCH}",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "--use_fast_math",
    "--ptxas-options=-v",
]

_WRAPPER_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

extern "C" {
void gemm_plain_bf16_launch(const void* X, const void* W, void* Y,
                            int M, int N, int K, cudaStream_t stream);
void gemm_fused_bf16_launch(const void* X, const void* W, const void* B_, void* Y,
                            int M, int N, int K, cudaStream_t stream);
}

torch::Tensor gemm_plain(torch::Tensor X, torch::Tensor W) {
    TORCH_CHECK(X.is_cuda() && X.is_contiguous(), "X must be contiguous CUDA");
    TORCH_CHECK(W.is_cuda() && W.is_contiguous(), "W must be contiguous CUDA");
    TORCH_CHECK(X.dim() == 2 && W.dim() == 2, "X and W must be 2D");
    TORCH_CHECK(X.size(1) == W.size(1), "X.size(1) must equal W.size(1) (K)");
    TORCH_CHECK(X.scalar_type() == at::kBFloat16, "bf16 only in starter");
    int M = X.size(0), K = X.size(1), N = W.size(0);
    auto Y = torch::empty({M, N}, X.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    gemm_plain_bf16_launch(X.data_ptr(), W.data_ptr(), Y.data_ptr(), M, N, K, stream);
    return Y;
}

torch::Tensor gemm_fused(torch::Tensor X, torch::Tensor W, torch::Tensor B_) {
    TORCH_CHECK(X.is_cuda() && X.is_contiguous(), "X must be contiguous CUDA");
    TORCH_CHECK(W.is_cuda() && W.is_contiguous(), "W must be contiguous CUDA");
    TORCH_CHECK(B_.is_cuda() && B_.is_contiguous(), "bias must be contiguous CUDA");
    TORCH_CHECK(X.dim() == 2 && W.dim() == 2 && B_.dim() == 1, "X,W 2D; bias 1D");
    TORCH_CHECK(X.size(1) == W.size(1), "X.size(1) must equal W.size(1) (K)");
    TORCH_CHECK(W.size(0) == B_.size(0), "W.size(0) must equal bias.size(0) (N)");
    TORCH_CHECK(X.scalar_type() == at::kBFloat16, "bf16 only in starter");
    int M = X.size(0), K = X.size(1), N = W.size(0);
    auto Y = torch::empty({M, N}, X.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    gemm_fused_bf16_launch(X.data_ptr(), W.data_ptr(), B_.data_ptr(), Y.data_ptr(), M, N, K, stream);
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_plain", &gemm_plain, "GEMM plain (X @ W.T)");
    m.def("gemm_fused", &gemm_fused, "GEMM fused (GELU(X @ W.T + bias))");
}
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is not None:
        return _ext
    wrapper_path = _HERE / "_wrapper.cpp"
    wrapper_path.write_text(_WRAPPER_SRC)
    _ext = _load_extension(
        name=f"gemm_kernel_{abs(hash(str(_HERE))) & 0xFFFFFFFF:x}",
        sources=[str(_HERE / "kernel.cu"), str(wrapper_path)],
        extra_cflags=_EXTRA_CFLAGS,
        extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
        verbose=False,
    )
    return _ext


def rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    variant: str,
    r: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unified entry name. For GEMM: x=[M,K], w=[N,K].
    variant='plain' -> Y=X@W.T;  variant='fused' -> Y=GELU(X@W.T + b)."""
    ext = _get_ext()
    if variant == "plain":
        return ext.gemm_plain(x, w)
    if variant == "fused":
        assert b is not None, "variant='fused' requires b="
        return ext.gemm_fused(x, w, b)
    raise ValueError(f"unknown variant: {variant}")
