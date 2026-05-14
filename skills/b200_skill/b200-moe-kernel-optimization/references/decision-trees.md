# Decision Trees

Operational logic for the most-asked questions during MoE optimization. Every threshold below was measured — don't tune without a reason.

---

## Tree 1: FP8 GEMM backend selection

```
Need FP8 GEMM on B200?
│
├── What's the scale format in your data?
│   │
│   ├── Per-tensor (single scalar per tensor)
│   │   → cuBLAS FP8 (but rare in real pretrained models; verify data source)
│   │
│   ├── 128-block float32 (Hopper-style, DeepSeek-V3, Kimi-K2, most common)
│   │   → CUTLASS FP8 grouped GEMM with blockwise epilogue
│   │     (collective: KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100)
│   │   │
│   │   └── After optimization plateaus and GEMM > 60% runtime?
│   │       → Hand-written tcgen05 (persistent warp-spec, TMEM accumulator)
│   │         Start from gau-nernst matmul_v7 template
│   │
│   └── 32-element UE8M0 (MXFP8, B200-native)
│       → CUTLASS with MXFP8 path, or tcgen05 with matching scale layout
│       → Any B200 native FP8 API works (rare in practice)
│
└── What about cuBLAS FP8?
    → All four cuBLAS FP8 APIs fail on B200:
      - cublasGemmBatchedEx → SIMT fallback, no tensor cores
      - cublasGemmStridedBatchedEx → correctness failures on a subset of workloads
      - cublasLtMatmul BLK128x128_32F → NOT_SUPPORTED
      - cublasGemmGroupedBatchedEx → RUNTIME_ERROR
    → Do not waste time here
```

### Quick reference
| Data format | Backend |
|---|---|
| 128-block float32 | CUTLASS blockwise → (later) tcgen05 |
| MXFP8 32-block | CUTLASS MXFP8 or tcgen05 |
| Per-tensor | cuBLAS FP8 (if API works for your path) |
| cuBLAS FP8 (all paths) | **Not viable on B200** |

---

## Tree 2: Tile shape selection for grouped GEMM

```
Estimate max per-expert M (typically total_rows / num_active_experts):
│
├── max_M ≤ 256
│   → 64 × 128 × 128 tile, 1-SM
│     (CUTLASS collective built with TileShape_MNK = Shape<_64,_128,_128>)
│
└── max_M > 256
    → 128 × 128 × 128 tile, 1-SM
      (TileShape_MNK = Shape<_128,_128,_128>)
```

### Constraints
- **Do not use M ≥ 256** — requires 2-SM cluster, which is **−18%** on SM100
- **Do not use N = 256** — **−60%** from wave quantization (both 128×256 and 64×256 regress)
- **Keep N = 128** always on SM100 MoE workloads

### Why this threshold
- Below 256, the 64-tile has enough parallelism and each expert's work is small
- Above 256, the 64-tile produces too many CTAs and per-CTA overhead dominates
- 128-tile amortizes that overhead

### Implementation
Dual-dispatch: compile both tile variants, pick at runtime:

```cpp
int max_M_estimate = total_rows / num_active_experts + 1;
CutlassBwFn gemm_fn = (max_M_estimate > 256 && g_cutlass_fn_128)
    ? g_cutlass_fn_128
    : cutlass_bw;
```

See [`code-examples/dual-tile-dispatch.md`](code-examples/dual-tile-dispatch.md).

---

## Tree 3: T-dependent GEMM2 backend dispatch

MoE workloads span a wide range of `T` (total tokens × top-k). Different backends win at different ranges.

```
GEMM2 (K=2048, N=7168) — backend selection
│
├── T ≤ 2000
│   → CUTLASS FP8 grouped
│     (better occupancy at low tile counts)
│
└── T > 2000
    → cuBLAS FP16 (COMPUTE_32F_FAST_16BF)
      (better saturation at large tile counts)

GEMM1 (larger K, e.g., K ≥ ~4k):
  → ALWAYS FP8 — the FP8 bandwidth advantage on large-K matmul
    is too large to give up; disabling produces a clear regression
```

### Why the threshold is around 2000
Below T ≈ 2000, CUTLASS grouped GEMM uses few tiles total and benefits from its static schedule. Above, cuBLAS FP16 fills the 148 SMs more evenly because it doesn't have the per-expert indirection.

### Why GEMM1 is always FP8
K=7168 is large enough that memory bandwidth dominates. FP8 halves the weight bandwidth vs FP16. The correctness fragility on the CUTLASS FP8 path is handled through other means (T-dependent dispatch of GEMM2, not GEMM1; plus the fp8-correctness-modes catalog).

### Threshold tuning
Threshold is workload-dependent. Measure on your actual T distribution before changing. See [`code-examples/t-dependent-dispatch.md`](code-examples/t-dependent-dispatch.md).

---

## Tree 4: When to move from CUTLASS to hand-written tcgen05

```
Is CUTLASS at the 14% occupancy wall?
│
├── NCU check: SM occupancy ≈ 14%?
│   AND shared memory per block ≈ 218 KB?
│   AND registers per thread ≈ 168?
│   → YES to all → CUTLASS is at the wall, tcgen05 is viable
│   → NO → you have other GEMM headroom; stay on CUTLASS, tune it first
│
├── Is GEMM > 60% of total runtime?
│   → YES → tcgen05 effort is worthwhile
│   → NO → focus on the non-GEMM parts first (zero-sync, pipeline, scatter)
│
├── Have all upstream optimizations been applied?
│   - Zero-sync fast path ✓
│   - Dual-tile dispatch ✓
│   - Pipeline overlap ✓
│   - Static compile ✓
│   → YES → ready for tcgen05
│   → NO → go back and finish those first
│
└── Do you have a tcgen05 reference template?
    → YES (e.g., gau-nernst matmul_v7) → adapt it
    → NO → writing from scratch is a significant undertaking; obtain a template
```

### Expected gain
- tcgen05 on GEMM1 only: +3.5%
- tcgen05 on GEMM1 + GEMM2 (dual TMA descriptors): additional +3.3% (total +6.8% from adding tcgen05)

These are the *last few percent* — the earlier items in the optimization ladder were much more impactful. Only proceed if everything above is done.

### Size gating
Typical runtime flags (name them however your project prefers; examples):

```
TCGEN05_MIN_T = 500   (skip tcgen05 for T < 500 — CUTLASS wins there)
NO_TCGEN05 = 1        (disable tcgen05, revert to CUTLASS)
USE_TCGEN05 = 1       (force tcgen05 regardless of size)
```

Very long sequence workloads (seq_len ≥ ~12k, T in the tens of thousands) may lose 2–7% with tcgen05 vs CUTLASS — keep CUTLASS as a fallback for the largest T as well.

---

## Tree 5: What's the bottleneck? (Where to optimize next)

```
NCU profile: what's dominating time?
│
├── Dominant kernel = GEMM, SM occupancy 14%, memory-bound
│   → You're at the CUTLASS wall
│   → Next step: hand-written tcgen05 (Tree 4)
│
├── Dominant kernel = GEMM, SM occupancy > 20%, compute-bound
│   → Something is wrong; grouped FP8 GEMM on MoE should be memory-bound
│   → Check: are you using CUTLASS 64-tile when you should use 128? (Tree 2)
│   → Check: is the kernel actually dispatching or silently falling back?
│
├── Dominant kernel = scatter / pull-scatter
│   → uint4 loads/stores (+2.1%)
│   → PSS overlap with preceding GEMM2 (+0.7%)
│
├── Dominant kernel = routing / top-k
│   → Zero-sync fast path if not done (+16.3%)
│   → Scan-scatter fusion (+2.8%)
│
├── Visible host stalls / CPU wait in timeline
│   → Zero-sync fast path (+16.3%) — this is the #1 case
│   → Move expert sort + prefix scan + CUTLASS argument setup to GPU
│
├── Cold-start vs warm-start discrepancy ~2×
│   → CUTLASS .so JIT is counted in first run
│   → Static compile + embedded headers (+7%)
│
├── Kernel has many small launches
│   → threadfence removal + metadata fusion (+5.8%)
│   → Scan-scatter fusion (+2.8%)
│
└── Nothing clearly dominant, speedup stuck
    → Measure more carefully (N ≥ 3 runs, isolated GPU)
    → Check for silent CUTLASS fallback (verify kernel names)
    → Re-read dead-ends-catalog before proposing anything exotic
```

---

## Tree 6: Scale alignment and grouped GEMM setup

```
Setting up a 128-block float32 FP8 grouped GEMM:
│
├── SFA layout (activation scales)
│   → One float32 per 128 elements of A (per-row, per-K-block)
│   → Stride per row: K / 128
│   → Pointer per expert group: SFA + row_offset × (K / 128)
│
├── SFB layout (weight scales)
│   → One float32 per 128×128 block of B
│   → Shape: [num_experts, N/128, K/128]
│   → Pointer per expert group: SFB + expert_id × (N/128) × (K/128)
│
├── A pointer per group
│   → A + row_offset × K   (row-major A, per-expert contiguous rows)
│
├── B pointer per group
│   → B + expert_id × N × K   (weights already per-expert)
│
├── D pointer per group
│   → D + row_offset × N
│
└── Strides
   → Use cutlass::make_cute_packed_stride with (M, K, 1), (N, K, 1), (M, N, 1)
   → SFA uses tile_atom_to_shape_SFA, SFB uses tile_atom_to_shape_SFB
```

See [`code-examples/dual-tile-dispatch.md`](code-examples/dual-tile-dispatch.md) for the complete `prep` kernel that builds these arrays.

---

## Tree 7: When to add PSS (`programmaticStreamSerialization`)

```
Adjacent kernel pair on the same stream, second consumes first's output?
│
├── Gap visible in NCU timeline between them?
│   → YES → PSS is likely to help (+0.5 to +0.7% per pair)
│   → NO → don't bother; overhead of the launch attribute may exceed gain
│
├── Kernels are on different streams?
│   → PSS doesn't help; use events for cross-stream dependency
│
└── Second kernel depends on host-side state after first?
    → Can't use PSS; you still need a host sync
```

PSS is applied via `cudaLaunchAttributeProgrammaticStreamSerialization` in `cudaLaunchKernelEx`. See [`code-examples/zero-sync-fast-path.md`](code-examples/zero-sync-fast-path.md) for the usage pattern.
