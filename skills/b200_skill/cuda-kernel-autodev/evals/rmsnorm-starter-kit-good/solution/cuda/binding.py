"""
Binding layer — JIT-compiles kernel.cu via torch.utils.cpp_extension and
exposes a single `rmsnorm(x, w, eps, variant, r=None, b=None)` entry point.

The agent SHOULD edit `kernel.cu` freely. It MAY edit this file when the
kernel signature changes (e.g., to add a new launch parameter, switch to
a persistent-kernel launch pattern, or add workspace handling), but should
preserve the `rmsnorm(...)` entry point signature so eval scripts keep working.

Compilation is cached at the torch.utils.cpp_extension level — edits to
kernel.cu trigger a rebuild automatically.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load as _load_extension

_HERE = Path(__file__).resolve().parent

# Compile with -arch=sm_100 for B200. Override via RMS_ARCH env var if needed.
_ARCH = os.environ.get("RMS_ARCH", "sm_100")
_EXTRA_CFLAGS = ["-O3", "-std=c++17"]
_EXTRA_CUDA_CFLAGS = [
    "-O3",
    "-std=c++17",
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

# A short wrapper source that exposes the C-launch functions as PyTorch ops.
_WRAPPER_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

extern "C" {
void rmsnorm_plain_bf16_launch(const void* X, const void* W, void* Y,
                               int M, int D, float eps,
                               int block_threads, cudaStream_t stream);
void rmsnorm_plain_fp16_launch(const void* X, const void* W, void* Y,
                               int M, int D, float eps,
                               int block_threads, cudaStream_t stream);
void rmsnorm_plain_fp32_launch(const void* X, const void* W, void* Y,
                               int M, int D, float eps,
                               int block_threads, cudaStream_t stream);
void rmsnorm_residual_bf16_launch(const void* X, const void* R, const void* W, void* Y,
                                  int M, int D, float eps,
                                  int block_threads, cudaStream_t stream);
void rmsnorm_affine_bf16_launch(const void* X, const void* W, const void* B, void* Y,
                                int M, int D, float eps,
                                int block_threads, cudaStream_t stream);
}

static int pick_block_threads(int D) {
    // Conservative: min(D, 1024), rounded to nearest power of 2 >= 64.
    if (D >= 1024) return 1024;
    if (D >= 512) return 512;
    if (D >= 256) return 256;
    if (D >= 128) return 128;
    return 64;
}

torch::Tensor rmsnorm_plain(torch::Tensor X, torch::Tensor W, double eps) {
    TORCH_CHECK(X.is_cuda() && X.is_contiguous(), "X must be contiguous CUDA");
    TORCH_CHECK(W.is_cuda() && W.is_contiguous(), "W must be contiguous CUDA");
    TORCH_CHECK(X.dim() == 2, "X must be 2D [M, D]");
    TORCH_CHECK(W.dim() == 1 && W.size(0) == X.size(1), "W shape must be [D]");
    auto Y = torch::empty_like(X);
    int M = X.size(0), D = X.size(1);
    int bt = pick_block_threads(D);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (X.scalar_type() == at::kBFloat16) {
        rmsnorm_plain_bf16_launch(X.data_ptr(), W.data_ptr(), Y.data_ptr(), M, D, (float)eps, bt, stream);
    } else if (X.scalar_type() == at::kHalf) {
        rmsnorm_plain_fp16_launch(X.data_ptr(), W.data_ptr(), Y.data_ptr(), M, D, (float)eps, bt, stream);
    } else if (X.scalar_type() == at::kFloat) {
        rmsnorm_plain_fp32_launch(X.data_ptr(), W.data_ptr(), Y.data_ptr(), M, D, (float)eps, bt, stream);
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }
    return Y;
}

torch::Tensor rmsnorm_residual(torch::Tensor X, torch::Tensor R, torch::Tensor W, double eps) {
    TORCH_CHECK(X.is_cuda() && X.is_contiguous() && R.is_contiguous() && W.is_contiguous(), "contiguous CUDA required");
    TORCH_CHECK(X.scalar_type() == at::kBFloat16, "residual variant: bf16 only (naive starter)");
    auto Y = torch::empty_like(X);
    int M = X.size(0), D = X.size(1);
    int bt = pick_block_threads(D);
    auto stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_residual_bf16_launch(X.data_ptr(), R.data_ptr(), W.data_ptr(), Y.data_ptr(), M, D, (float)eps, bt, stream);
    return Y;
}

torch::Tensor rmsnorm_affine(torch::Tensor X, torch::Tensor W, torch::Tensor B, double eps) {
    TORCH_CHECK(X.is_cuda() && X.is_contiguous() && W.is_contiguous() && B.is_contiguous(), "contiguous CUDA required");
    TORCH_CHECK(X.scalar_type() == at::kBFloat16, "affine variant: bf16 only (naive starter)");
    auto Y = torch::empty_like(X);
    int M = X.size(0), D = X.size(1);
    int bt = pick_block_threads(D);
    auto stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_affine_bf16_launch(X.data_ptr(), W.data_ptr(), B.data_ptr(), Y.data_ptr(), M, D, (float)eps, bt, stream);
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_plain",    &rmsnorm_plain,    "RMS norm plain");
    m.def("rmsnorm_residual", &rmsnorm_residual, "RMS norm with residual add");
    m.def("rmsnorm_affine",   &rmsnorm_affine,   "RMS norm with affine (weight+bias)");
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
        name=f"rmsnorm_kernel_{abs(hash(str(_HERE))) & 0xFFFFFFFF:x}",
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
    """Unified entry point. `variant` ∈ {"plain","residual","affine"}."""
    ext = _get_ext()
    if variant == "plain":
        return ext.rmsnorm_plain(x, w, float(eps))
    if variant == "residual":
        assert r is not None, "variant='residual' requires r="
        return ext.rmsnorm_residual(x, r, w, float(eps))
    if variant == "affine":
        assert b is not None, "variant='affine' requires b="
        return ext.rmsnorm_affine(x, w, b, float(eps))
    raise ValueError(f"unknown variant: {variant}")
