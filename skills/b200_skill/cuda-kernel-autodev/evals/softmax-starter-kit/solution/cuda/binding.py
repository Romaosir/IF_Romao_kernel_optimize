"""
Softmax binding — JIT-compiles kernel.cu and exposes an `rmsnorm(...)` entry
point (so the harness code is reusable). For softmax, only `x` is used; `w`,
`eps`, `r`, `b` are accepted and ignored. The harness passes a dummy weight
tensor which is safe to ignore.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load as _load_extension

_HERE = Path(__file__).resolve().parent
_ARCH = os.environ.get("SM_ARCH", "sm_100")
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
void softmax_plain_bf16_launch(const void* X, void* Y, int M, int D,
                               int block_threads, cudaStream_t stream);
void softmax_plain_fp16_launch(const void* X, void* Y, int M, int D,
                               int block_threads, cudaStream_t stream);
void softmax_plain_fp32_launch(const void* X, void* Y, int M, int D,
                               int block_threads, cudaStream_t stream);
}

static int pick_block_threads(int D) {
    if (D >= 1024) return 1024;
    if (D >= 512) return 512;
    if (D >= 256) return 256;
    if (D >= 128) return 128;
    return 64;
}

torch::Tensor softmax_plain(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.is_contiguous(), "X must be contiguous CUDA");
    TORCH_CHECK(X.dim() == 2, "X must be 2D [M, D]");
    auto Y = torch::empty_like(X);
    int M = X.size(0), D = X.size(1);
    int bt = pick_block_threads(D);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (X.scalar_type() == at::kBFloat16) {
        softmax_plain_bf16_launch(X.data_ptr(), Y.data_ptr(), M, D, bt, stream);
    } else if (X.scalar_type() == at::kHalf) {
        softmax_plain_fp16_launch(X.data_ptr(), Y.data_ptr(), M, D, bt, stream);
    } else if (X.scalar_type() == at::kFloat) {
        softmax_plain_fp32_launch(X.data_ptr(), Y.data_ptr(), M, D, bt, stream);
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_plain", &softmax_plain, "Softmax (plain)");
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
        name=f"softmax_kernel_{abs(hash(str(_HERE))) & 0xFFFFFFFF:x}",
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
    """Unified entry name (the harness expects `rmsnorm`). For softmax this
    is softmax(x). `w`, `eps`, `r`, `b` are ignored."""
    ext = _get_ext()
    if variant != "plain":
        raise ValueError(f"softmax starter supports variant='plain' only, got {variant!r}")
    return ext.softmax_plain(x)
