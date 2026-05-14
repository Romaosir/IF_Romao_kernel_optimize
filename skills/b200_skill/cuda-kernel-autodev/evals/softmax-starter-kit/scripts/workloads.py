"""
Workload definitions for the softmax eval harness.

Row-wise softmax on X shape [M, D].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Variant = Literal["plain"]
Dtype = Literal["bf16", "fp16", "fp32"]


@dataclass(frozen=True)
class Workload:
    name: str
    M: int
    D: int
    dtype: Dtype = "bf16"
    variant: Variant = "plain"
    eps: float = 0.0  # softmax has no eps; kept for harness compatibility
    abs_tol: float = 3e-2
    rel_tol: float = 1e-2
    tags: tuple[str, ...] = field(default_factory=tuple)


ALL_WORKLOADS: list[Workload] = [
    # Baseline
    Workload("bs32_d4096", M=32 * 4096, D=4096, dtype="bf16", tags=("baseline", "bf16")),
    Workload("bs16_d4096", M=16 * 4096, D=4096, dtype="bf16", tags=("baseline",)),
    Workload("bs8_d8192",  M=8 * 4096,  D=8192, dtype="bf16", tags=("baseline", "long_d")),

    # Long reduction
    Workload("long_d16k",  M=8 * 2048,  D=16384, dtype="bf16", tags=("long_reduction",)),
    Workload("long_d12k",  M=8 * 2048,  D=12288, dtype="bf16", tags=("long_reduction",)),

    # Small batch
    Workload("bs1_d4096", M=1 * 512, D=4096, dtype="bf16", tags=("small_batch",)),
    Workload("bs1_d8192", M=1 * 512, D=8192, dtype="bf16", tags=("small_batch",)),

    # fp16
    Workload("fp16_bs32_d4096", M=32 * 4096, D=4096, dtype="fp16",
             abs_tol=5e-3, rel_tol=3e-3, tags=("fp16",)),
    Workload("fp16_bs16_d8192", M=16 * 4096, D=8192, dtype="fp16",
             abs_tol=5e-3, rel_tol=3e-3, tags=("fp16",)),

    # fp32
    Workload("fp32_bs16_d4096", M=16 * 4096, D=4096, dtype="fp32",
             abs_tol=1e-5, rel_tol=1e-5, tags=("fp32",)),

    # Numerical stress: large-magnitude X
    Workload("large_x_d4096", M=16 * 4096, D=4096, dtype="bf16",
             abs_tol=3e-2, rel_tol=1e-2, tags=("numerical_stress",)),
]


def select(tags_csv: str | None) -> list[Workload]:
    if not tags_csv or tags_csv.strip().lower() == "all":
        return list(ALL_WORKLOADS)
    wanted = {t.strip() for t in tags_csv.split(",") if t.strip()}
    return [w for w in ALL_WORKLOADS if wanted & set(w.tags)]


DTYPE_MAP = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
