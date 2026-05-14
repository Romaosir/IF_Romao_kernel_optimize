# MoE dispatch decision trees

Operational dispatch logic for the FuseMoE kernel. Each threshold was measured on B200; do not tune without a reason. For the *generic* "what to try next" framework see [`cuda-roofline-strategy`](../../../../cuda-roofline-strategy/SKILL.md); the trees here are the MoE-specific runtime choices that the campaign converged on.

---

## Tree 1 — FP8 GEMM backend selection

```
Need FP8 GEMM on B200?
│
├── What's the scale format in your data?
│   │
│   ├── Per-tensor (single scalar per tensor)
│   │   → cuBLAS FP8 (rare in real pretrained models; verify the data source)
│   │
│   ├── 128-block float32 (Hopper-style; DeepSeek-V3, Kimi-K2, etc. — most common)
│   │   → CUTLASS FP8 grouped GEMM with blockwise epilogue
│   │     (collective: KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100)
│   │   │
│   │   └── After optimisation plateaus and GEMM > 60% of iter?
│   │       → Hand-written tcgen05 (persistent warp-spec, TMEM accumulator)
│   │         Start from gau-nernst matmul_v7 template
│   │
│   └── 32-element UE8M0 (MXFP8, B200-native)
│       → CUTLASS with MXFP8 path, or tcgen05 with matching scale layout
│       → Any B200 native FP8 API works (rare in pretrained weights)
│
└── What about cuBLAS FP8?
    → All four cuBLAS FP8 APIs fail on B200 — see hardware-b200.md
```

| Data format | Backend |
|---|---|
| 128-block float32 | CUTLASS blockwise → (later) tcgen05 |
| MXFP8 32-block | CUTLASS MXFP8 or tcgen05 |
| Per-tensor | cuBLAS FP8 (if API works for your path) |
| cuBLAS FP8 (all paths) | **Not viable on B200** |

---

## Tree 2 — Tile shape selection for grouped GEMM

```
max_M_estimate = total_rows / num_active_experts + 1
│
├── max_M ≤ 256 → 64×128×128 tile, 1-SM
└── max_M > 256 → 128×128×128 tile, 1-SM
```

**Constraints (from `hardware-b200.md` and campaign measurement):**
- Do not use M ≥ 256 — requires 2-SM cluster, which is **−18%** on SM100 for this workload class.
- Do not use N = 256 — **−60%** from wave quantisation (both 128×256 and 64×256 regress).
- Keep N = 128.

**Why this threshold.** Below max_M = 256, the 64-tile produces enough parallelism and each expert's work is small. Above, the 64-tile produces *too many* CTAs and per-tile overhead dominates; the 128-tile amortises it.

**Implementation.** Compile both variants; pick at runtime — see [`../../code-examples/moe-dual-tile-dispatch.md`](../../code-examples/moe-dual-tile-dispatch.md). Worth +13% — the largest single CUTLASS-internal lever in the campaign.

---

## Tree 3 — T-dependent GEMM2 backend dispatch

```
GEMM2 (K = 2048, N = 7168) total tokens T?
│
├── T ≤ 2000  → CUTLASS FP8 grouped (better small-T occupancy)
└── T >  2000 → cuBLAS FP16 (COMPUTE_32F_FAST_16BF, better at high T saturation)

GEMM1 (larger K, ≥ ~4k): always FP8.
The FP8 bandwidth advantage on large-K matmul is too large to give up.
```

**Why the threshold is around 2000.** Below T ≈ 2000, CUTLASS grouped GEMM uses few tiles total and benefits from its static schedule. Above, cuBLAS FP16 fills the 148 SMs more evenly because it doesn't have the per-expert indirection.

**Why GEMM1 is always FP8.** K = 7168 is large enough that memory bandwidth dominates. FP8 halves the weight bandwidth vs FP16. The correctness fragility on the CUTLASS FP8 path is handled through other means (T-dependent dispatch of *GEMM2*, plus the `fp8-correctness-modes.md` catalogue).

**Threshold tuning.** Workload-dependent — measure on your actual T distribution before changing. See [`../../code-examples/moe-t-dependent-dispatch.md`](../../code-examples/moe-t-dependent-dispatch.md).

---

## Tree 4 — When to move from CUTLASS to hand-written tcgen05

```
Is CUTLASS at the 14% occupancy wall?
│
├── NCU check: SM occupancy ≈ 14% AND smem ≈ 218 KB AND regs ≈ 168?
│   → YES → CUTLASS is at the wall, tcgen05 is viable
│   → NO  → tune CUTLASS further first
│
├── Is GEMM > 60% of total runtime?
│   → YES → tcgen05 effort is worthwhile
│   → NO  → focus on non-GEMM parts first (zero-sync, pipeline, scatter)
│
├── Have all upstream optimisations been applied?
│   - Zero-sync fast path ✓
│   - Dual-tile dispatch ✓
│   - Pipeline overlap ✓
│   - Static compile ✓
│   → YES → ready for tcgen05
│   → NO  → finish those first
│
└── Do you have a tcgen05 reference template?
    → YES (gau-nernst matmul_v7, or a CUTLASS example) → adapt it
    → NO  → writing from scratch is significant effort; obtain a template
```

This is the same four-condition test described as "the hand-write trigger" in [`cuda-roofline-strategy/references/strategy-matrix.md`](../../../../cuda-roofline-strategy/references/strategy-matrix.md) "Compute-bound × Plateau". The strategy-matrix version states the principle generically; this one names the specific MoE thresholds.

**Expected gain.** tcgen05 on GEMM1 only: +3.5%. tcgen05 on GEMM1 + GEMM2 with dual TMA descriptors: additional +3.3% (total +6.8%). These are the *last few percent* — earlier ladder items contributed much more.

**Size gating example.** Use a `TCGEN05_MIN_T` runtime flag — typical sweet spot **500**. Below that, CUTLASS's static schedule wins. Very long sequences (seq_len ≥ ~12k, T in the tens of thousands) may also lose 2–7% with tcgen05 vs CUTLASS — keep CUTLASS as fallback for the largest T as well.

---

## Tree 5 — Scale-alignment and grouped-GEMM argument setup

Constructing arguments for a 128-block float32 FP8 grouped GEMM:

```
SFA layout (activation scales)
  → One float32 per 128 elements of A (per-row, per-K-block)
  → Stride per row: K / 128
  → Pointer per expert group: SFA + row_offset × (K / 128)

SFB layout (weight scales)
  → One float32 per 128×128 block of B
  → Shape: [num_experts, N/128, K/128]
  → Pointer per expert group: SFB + expert_id × (N/128) × (K/128)

A pointer per group: A + row_offset × K   (row-major, per-expert contiguous rows)
B pointer per group: B + expert_id × N × K
D pointer per group: D + row_offset × N

Strides: cutlass::make_cute_packed_stride with (M, K, 1), (N, K, 1), (M, N, 1)
SFA uses tile_atom_to_shape_SFA; SFB uses tile_atom_to_shape_SFB
```

See [`../../code-examples/moe-dual-tile-dispatch.md`](../../code-examples/moe-dual-tile-dispatch.md) for the complete `prep` kernel that builds these arrays. Getting any one of these wrong is a silent-correctness bug — the scales misalign, output looks plausible (e.g. matched_ratio ~ 0.85), and bench numbers stay believable until you hit a workload where the misalignment becomes catastrophic.

---

## Tree 6 — When to add PSS / griddepcontrol

```
Adjacent kernel pair on the same stream, second consumes first's output?
│
├── Gap visible in NCU timeline between them?
│   → YES → PSS or griddepcontrol helps (+0.5–0.7% per pair)
│   → NO  → don't bother
│
├── Kernels are on different streams?
│   → PSS doesn't help; use events
│
└── Second kernel depends on host-side state after first?
    → Can't use PSS; need host sync
```

For the underlying mechanism and code pattern, see [`cuda-kernel-techniques/references/parallelism.md`](../../parallelism.md) "Programmatic Dependent Launch (PDL) and `griddepcontrol`" and [`../../code-examples/zero-sync-fast-path.md`](../../code-examples/zero-sync-fast-path.md).

---

## How these trees relate to the rest of the skill

- **Trees 1, 2, 3, 5** answer *"which backend / shape / argument layout do I pick at runtime"* — MoE-specific operational logic. They live here, not in the catalogue, because the thresholds (256, 2000, 60%, etc.) are MoE-workload-specific.
- **Tree 4** is the MoE-specific instance of the *generic* trigger described in [`cuda-roofline-strategy/strategy-matrix.md`](../../../../cuda-roofline-strategy/references/strategy-matrix.md): "move to a hand-written next-gen primitive when at the library's occupancy ceiling". That document covers the principle; this tree names the thresholds.
- **Tree 6** is a thin pointer to a generic technique that lives in `parallelism.md`.

For *which optimisation to add next* (rather than how to dispatch among existing ones), use [`optimization-ladder.md`](optimization-ladder.md) in this folder + [`cuda-roofline-strategy/strategy-matrix.md`](../../../../cuda-roofline-strategy/references/strategy-matrix.md).
