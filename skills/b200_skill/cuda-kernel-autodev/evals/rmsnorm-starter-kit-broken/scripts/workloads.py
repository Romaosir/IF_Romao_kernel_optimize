"""
Workload definitions for the RMS norm eval harness.

A workload is a concrete shape + dtype + variant combination that exercises
the kernel. The eval runs all workloads belonging to the active scenario.

Each scenario (defined per-prompt in `evals.json`) selects a subset of these
workloads via the WORKLOAD_SET env var (e.g. WORKLOAD_SET=baseline,small).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Variant = Literal["plain", "residual", "affine"]
Dtype = Literal["bf16", "fp16", "fp32"]


@dataclass(frozen=True)
class Workload:
    name: str
    M: int  # flattened rows (batch * seq)
    D: int  # hidden dim
    dtype: Dtype = "bf16"
    variant: Variant = "plain"
    eps: float = 1e-6
    # Correctness tolerances — dtype-default here, overridable per workload.
    # bf16 has only ~7 mantissa bits; 1 ULP near 1.0 is ~2^-7 ≈ 0.008, and
    # differences in reduction order add another ULP or two. 3e-2 / 1e-2
    # accommodates honest rounding noise without hiding real bugs.
    abs_tol: float = 3e-2
    rel_tol: float = 1e-2
    # Which scenario families include this workload (used by WORKLOAD_SET)
    tags: tuple[str, ...] = field(default_factory=tuple)


ALL_WORKLOADS: list[Workload] = [
    # --- Baseline shapes (scenario 1) ---
    Workload("bs32_d4096_plain", M=32 * 4096, D=4096, dtype="bf16", variant="plain",
             tags=("baseline", "bf16")),
    Workload("bs16_d4096_plain", M=16 * 4096, D=4096, dtype="bf16", variant="plain",
             tags=("baseline",)),
    Workload("bs8_d8192_plain",  M=8 * 4096,  D=8192, dtype="bf16", variant="plain",
             tags=("baseline", "long_d")),

    # --- Small batch — occupancy pressure (scenario 2) ---
    Workload("bs1_d4096",  M=1 * 512,  D=4096, dtype="bf16", variant="plain",
             tags=("small_batch",)),
    Workload("bs1_d2048",  M=1 * 512,  D=2048, dtype="bf16", variant="plain",
             tags=("small_batch",)),
    Workload("bs1_d8192",  M=1 * 512,  D=8192, dtype="bf16", variant="plain",
             tags=("small_batch", "long_d")),

    # --- Large batch (scenario 3) ---
    Workload("bs64_d4096", M=64 * 8192, D=4096, dtype="bf16", variant="plain",
             tags=("large_batch",)),
    Workload("bs32_d8192", M=32 * 8192, D=8192, dtype="bf16", variant="plain",
             tags=("large_batch", "long_d")),

    # --- Long reduction (scenario 4) ---
    Workload("long_d16k",  M=8 * 2048,  D=16384, dtype="bf16", variant="plain",
             tags=("long_reduction",)),
    Workload("long_d12k",  M=8 * 2048,  D=12288, dtype="bf16", variant="plain",
             tags=("long_reduction",)),

    # --- Affine variant (scenario 5) ---
    Workload("affine_bs32_d4096", M=32 * 4096, D=4096, dtype="bf16", variant="affine",
             tags=("affine",)),
    Workload("affine_bs16_d8192", M=16 * 4096, D=8192, dtype="bf16", variant="affine",
             tags=("affine",)),

    # --- Residual variant (scenario 6) ---
    Workload("residual_bs32_d4096", M=32 * 4096, D=4096, dtype="bf16", variant="residual",
             tags=("residual",)),
    Workload("residual_bs16_d8192", M=16 * 4096, D=8192, dtype="bf16", variant="residual",
             tags=("residual",)),

    # --- fp16 (scenario 7) — fp16 has 10 mantissa bits, tighter tolerances ---
    Workload("fp16_bs32_d4096", M=32 * 4096, D=4096, dtype="fp16", variant="plain",
             abs_tol=5e-3, rel_tol=3e-3, tags=("fp16",)),
    Workload("fp16_bs16_d8192", M=16 * 4096, D=8192, dtype="fp16", variant="plain",
             abs_tol=5e-3, rel_tol=3e-3, tags=("fp16",)),

    # --- fp32 (scenario 8) ---
    Workload("fp32_bs16_d4096", M=16 * 4096, D=4096, dtype="fp32", variant="plain",
             abs_tol=1e-5, rel_tol=1e-5, tags=("fp32",)),
    Workload("fp32_bs8_d8192",  M=8 * 4096,  D=8192, dtype="fp32", variant="plain",
             abs_tol=1e-5, rel_tol=1e-5, tags=("fp32",)),
]


def select(tags_csv: str | None) -> list[Workload]:
    """Pick workloads whose `tags` intersect the given comma-separated tag list.
    If `tags_csv` is None or 'all', return everything.
    """
    if not tags_csv or tags_csv.strip().lower() == "all":
        return list(ALL_WORKLOADS)
    wanted = {t.strip() for t in tags_csv.split(",") if t.strip()}
    return [w for w in ALL_WORKLOADS if wanted & set(w.tags)]


DTYPE_MAP = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
