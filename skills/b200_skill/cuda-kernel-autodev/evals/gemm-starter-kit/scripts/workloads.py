"""
Workload definitions for the GEMM eval harness.

Operator: Y = GELU(X @ W.T + bias) with X=[M,K], W=[N,K], bias=[N], Y=[M,N].
Common transformer-MLP shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Variant = Literal["plain", "fused"]
Dtype = Literal["bf16", "fp16"]


@dataclass(frozen=True)
class Workload:
    name: str
    M: int
    # For GEMM: interpret D as N (output feature dim); the harness packs K into the weight's K dim.
    # We store M, N, K explicitly via the extra fields. M is the batch*seq dimension.
    D: int  # alias for N (output features) — harness expects this field name
    K: int = 4096
    dtype: Dtype = "bf16"
    variant: Variant = "fused"
    eps: float = 0.0  # unused for GEMM; kept for harness compat
    abs_tol: float = 5e-2   # GEMM on bf16 accumulates more rounding than pointwise ops
    rel_tol: float = 3e-2
    tags: tuple[str, ...] = field(default_factory=tuple)


ALL_WORKLOADS: list[Workload] = [
    # --- Baseline MLP shapes ---
    Workload("mlp_1024_4k_4k", M=1024, D=4096,  K=4096,  dtype="bf16", variant="fused",
             tags=("baseline", "bf16")),
    Workload("mlp_2048_4k_4k", M=2048, D=4096,  K=4096,  dtype="bf16", variant="fused",
             tags=("baseline",)),
    Workload("mlp_1024_11k_4k", M=1024, D=11008, K=4096, dtype="bf16", variant="fused",
             tags=("baseline", "llama_mlp")),

    # --- Skinny / decoding-style ---
    Workload("decode_32_4k_4k",   M=32,  D=4096,  K=4096, dtype="bf16", variant="fused",
             tags=("decode", "small_batch")),
    Workload("decode_128_11k_4k", M=128, D=11008, K=4096, dtype="bf16", variant="fused",
             tags=("decode",)),

    # --- Plain (no fusion) for comparison if needed ---
    Workload("plain_1024_4k_4k", M=1024, D=4096, K=4096, dtype="bf16", variant="plain",
             tags=("plain_only",)),
]


def select(tags_csv: str | None) -> list[Workload]:
    if not tags_csv or tags_csv.strip().lower() == "all":
        return list(ALL_WORKLOADS)
    wanted = {t.strip() for t in tags_csv.split(",") if t.strip()}
    return [w for w in ALL_WORKLOADS if wanted & set(w.tags)]


DTYPE_MAP = {"bf16": "bfloat16", "fp16": "float16"}
