# Optimization Ladder (ordered by observed ROI)

Every row in this table was measured in production MoE optimization work on B200. Apply top-down. Don't skip to later items unless earlier items are actually done — the ladder is not a menu.

| # | Technique | Typical gain | Scope |
|---|---|---|---|
| 1 | cuBLAS FP16 → CUTLASS FP8 grouped GEMM | **+60%** | Largest cumulative gain — the GEMM backend swap |
| 2 | Zero-sync fast path (metadata → GPU) | **+16.3%** | Largest single-step gain |
| 3 | Dual-tile CUTLASS dispatch (64 / 128) | **+13%** | MoE-specific tile selection |
| 4 | Static compile + embedded CUTLASS headers | **+7.0%** | Eliminates dynamic-load cache noise |
| 5 | Pipeline overlap (stream async + D2H) | **+5.7%** | After sync is clean |
| 6 | threadfence removal + metadata fusion | **+5.8%** | Reduce GPU-side sync primitives |
| 7 | tcgen05 on GEMM1 (hand-written PTX) | **+3.5%** | Breaks 14% occupancy wall |
| 8 | tcgen05 on GEMM1 + GEMM2 (dual TMA desc) | **+3.3%** | Applying tcgen05 to the second GEMM with its own descriptors |
| 9 | Dual (fused) prep kernel for GEMM1+GEMM2 | **+2–3%** | One kernel writes both argument-array sets — removes one launch |
| 10 | Scan-scatter fusion (↓ kernel count) | **+2.8%** | Reduces launch overhead |
| 11 | uint4 (128-bit) scatter loads/stores | **+2.1%** | Memory-bound scatter kernels |
| 12 | T-dependent GEMM2 backend | **+1.4%** | Mixed FP8 CUTLASS / cuBLAS FP16 |
| 13 | Warp-parallel routing group-scores | **+1.4%** | Non-GEMM: shuffle-reduce vs lane-0 serial |
| 14 | Routing: keep intermediates in regs (drop `s[]` shmem) | **+1.0%** | Non-GEMM micro-opt |
| 15 | Memset fusion into pull_scatter tail | **+1.0%** | Removes a dedicated `cudaMemsetAsync` launch |
| 16 | PSS (`programmaticStreamSerialization`) | **+0.7%** | Adjacent kernel overlap |
| 17 | `griddepcontrol` PTX (grid-level dependency) | **+0.5%** | Finer-grained PSS sibling; see `code-examples/zero-sync-fast-path.md` |
| 18 | Skip cuBLAS allocs on fast path | **+0.5%** | Guard cuBLAS setup behind fallback flag |
| 19 | Routing sync-barrier removal | **+0.3%** | Drop redundant `__syncthreads()` in top-k loop |
| 20 | SwiGLU 8 rows per block | **+0.3–0.5%** | Better smem/occupancy balance |
| 21 | Grid tightening to actual work | **+1–2% (large)** | Don't over-launch empty CTAs for non-persistent kernels |
| — | BF16 fast compute (`COMPUTE_32F_FAST_16BF`) | stable shift | Always enable on cuBLAS FP16 |
| — | CUTLASS `max_swizzle_size = 4` | stable shift | Applied on every CUTLASS argument path |
| — | `can_implement` + `get_workspace_size` caching | cold-start speedup | Both are O(ms) first call; cache per (tile, shape) |
| — | GEMV fast-path for `M ≤ 2` | enables viable small-T | Dedicated kernel when a per-expert M is degenerate |
| — | Bucketed grouped GEMM (shape-aware) | neutral at defaults | Threshold 128, max-buckets 6; see `tuning-knobs.md` |

**Key insight:** Items 1 + 2 sum to **+76%** — more than all remaining items combined. Don't start with GEMM internals.

## Expected speedup landmarks

Use these as sanity checks. If you are below the landmark after applying the preceding items, one of them isn't actually there (common cause: CUTLASS silent fallback, or `sm_100a` compile target missing).

| After applying items… | Typical range |
|---|---|
| Item 1 only (CUTLASS FP8 grouped GEMM) | ~43× |
| Items 1–2 (+ zero-sync fast path) | ~50× |
| Items 1–4 (+ dual-tile + static compile) | ~57–65× |
| Items 1–6 (+ pipeline + threadfence/fusion) | ~70–78× |
| Items 1–8 (+ tcgen05 on both GEMMs) | ~85–90× |
| Items 1–8 plus the non-GEMM micro-ops (items 9–21) | ~90–93× |

Being stuck at ~48× means: item 2 (zero-sync) and items 7–8 (tcgen05) are the most likely missing pieces. Check them explicitly.

## Revert discipline

When a named ladder item **regresses** after you implement it, do NOT silently revert. Instead:

1. **Read the matching code-example first.** Does your implementation match its pattern exactly?
2. **Suspect the orthogonal bug.** Common in practice: a pointer-alignment bug or stale buffer corrupts output but *looks* like "the ladder item broke things". Fix the orthogonal bug, re-apply the ladder item.
3. **Record the hypothesis in a snapshot_doc note** before reverting, so the next attempt has breadcrumbs.
4. **Re-attempt after fixing the underlying bug.** Every named ladder item with a measured percentage is known to work on this hardware — if it regresses on first try, the implementation has a bug.

---

## 1. cuBLAS FP16 → CUTLASS FP8 grouped GEMM (+60%)

### Context
The cuBLAS FP16 baseline gets ~26.8×. The goal is FP8 tensor-core throughput, but all four cuBLAS FP8 paths fail on B200 (see `hardware-constraints.md`). The only working path for 128-block float32 scales is CUTLASS with a custom blockwise epilogue.

### Implementation
1. Embed CUTLASS source in your tree (vendor headers)
2. Use the `KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100` collective
3. Write a `prep` kernel that builds pointer arrays, strides, and per-expert problem shapes
4. JIT-compile with `nvcc -arch=compute_100a` to `/tmp/.so`, `dlopen` at runtime

See [`code-examples/cutlass-jit-compile.md`](code-examples/cutlass-jit-compile.md) for the JIT pattern, and [`code-examples/dual-tile-dispatch.md`](code-examples/dual-tile-dispatch.md) for the grouped GEMM setup.

### NCU metrics to confirm
- Kernel name contains `...Blockwise1SmSm100...` (not `cublas*` — catches silent fallback)
- DRAM throughput > 50% (FP8 should be memory-bound on realistic MoE sizes)
- SM occupancy ~14% (this is expected, confirms you're at the wall)

### Pitfalls
- **Silent fallback to cuBLAS FP16** — verify with NCU kernel names
- **Wrong scale layout** — `SFA` (per-row) and `SFB` (per-block-per-expert) need correct strides; see `prep` kernel example

---

## 2. Zero-sync fast path (+16.3%)

> **Before attempting, read `code-examples/zero-sync-fast-path.md` in full.**
> It has the correct memset-ordering, alignment, PSS, and `griddepcontrol` PTX pattern.
> Implementing this from intuition typically regresses or corrupts output on the first attempt —
> and the regression looks like a different bug (CUTLASS fallback, pointer alignment).
> That's why this gets incorrectly abandoned. Re-read the example first.

### Context
Before this change, the pipeline is:
```
routing → D2H memcpy → cudaStreamSynchronize (CPU stall ~50µs)
       → CPU: prefix scan, expert sort, CUTLASS argument setup
       → H2D → gather → CUTLASS GEMM
```
For small-T workloads (seq_len ≤ 16), the 50µs CPU stall is ~37% of total time.

### Implementation
Move every piece of metadata construction onto the GPU:

- Expert count prefix scan → GPU kernel (simple cub::DeviceScan or hand-written)
- Active-expert sort → GPU kernel (bitonic or radix)
- CUTLASS argument arrays (pointer arrays, strides, problem shapes) → GPU kernel

Then launch downstream kernels with grid-level dependency control:
- Use `programmaticStreamSerialization` (PSS) to chain kernels without host sync
- Use `griddepcontrol.wait` / `griddepcontrol.launch_dependents` PTX for fine-grained dependency

**`cudaStreamSynchronize` must be eliminated from the hot path.**

See [`code-examples/zero-sync-fast-path.md`](code-examples/zero-sync-fast-path.md) for the full pattern.

### NCU metrics
- No CPU stall windows visible in timeline
- `Host Wait` time near zero
- Kernels chain back-to-back without gaps

### Pitfalls
- Forgetting to initialize a counter on the GPU (was previously zeroed on the CPU)
- Missing a `__threadfence_system()` that was implicit in the old `cudaStreamSynchronize`

---

## 3. Dual-tile CUTLASS dispatch (+13%)

### Context
A single CUTLASS configuration cannot cover both small-M experts (few tokens per expert) and large-M experts (many tokens per expert) efficiently:
- 64×128×128 tile → efficient for small M, wastes work for large M
- 128×128×128 tile → efficient for large M, wastes work for small M

MoE batches contain both at once.

### Implementation
Compile two CUTLASS grouped GEMM variants, dispatch based on `max_M`:

```cpp
int max_M_estimate = total_rows / num_active_experts + 1;

CutlassBwFn gemm1_fn = cutlass_bw;  // 64-tile default
if (max_M_estimate > 256 && g_cutlass_fn_128) {
    gemm1_fn = g_cutlass_fn_128;   // 128-tile for large M
}
int ret = gemm1_fn(&cargs, stream);
```

See [`code-examples/dual-tile-dispatch.md`](code-examples/dual-tile-dispatch.md) for the full dispatch logic.

### Threshold
`max_M > 256` switches to 128-tile. Below that, 64-tile wins.

### Pitfalls
- Threshold is workload-dependent; verify on your actual expert distribution
- Don't try 256-wide M-tile (needs 2-SM cluster, −18%)

---

## 4. Static compile + embedded CUTLASS headers (+7.0%)

### Context
When CUTLASS is JIT-compiled to a `.so` and `dlopen`-ed at runtime, the very first run of a benchmark includes compile time. Subsequent runs hit the cache. This creates cold-vs-warm measurement noise and genuinely slows the first invocation.

### Implementation
Inline the CUTLASS source as a raw string literal inside `kernel.cu`, and compile at kernel init (not at invocation):

```cpp
static const char kCutlassSrc[] = R"CUTLASS_SRC(
  // ... CUTLASS grouped GEMM source (~500 lines)
)CUTLASS_SRC";

static void build_cutlass_so() {
    // Write kCutlassSrc to /tmp/cutlass_src.cu
    // Call: nvcc -arch=compute_100a -O2 --shared ... -o /tmp/libcutlass.so
    // dlopen the result at module init
}
```

This guarantees the `.so` exists before any benchmark invocation — the first run no longer pays compile cost.

Observed gain: **+7%**.

### Pitfalls
- Source literal must be one string; escape `\` and `"` carefully
- Build command must target `sm_100a` for tcgen05 features

---

## 5. Pipeline overlap (+5.7%)

### Context
After zero-sync is done, routing → gather → GEMM1 → swiglu → GEMM2 → scatter still executes serially. Async streams + D2H overlap expose more parallelism.

### Implementation
- Use non-default CUDA streams for D2H memcpy
- Dispatch scatter on a separate stream from GEMM2, with event dependency
- Use `cudaLaunchAttributeProgrammaticStreamSerialization` for chained dependent kernels

### Pitfalls
- Event creation overhead can exceed the overlap benefit for small T — measure carefully
- Make sure D2H on the side stream doesn't race with the next iteration's H2D

---

## 6. threadfence removal + metadata fusion (+5.8%)

### Context
The GPU-side planner (from step 2) introduces `__threadfence()` for consistency. Most can be removed if dependencies are already enforced by grid-level ordering.

### Implementation
- Audit every `__threadfence()` and `__threadfence_system()`
- Remove any that are subsumed by a subsequent mbarrier or grid dependency
- Fuse adjacent metadata kernels — two small kernels with the same grid shape can often become one

### NCU metrics
- Kernel count in timeline drops
- `stall_membar` reason in SM stall breakdown decreases

---

## 7. tcgen05 on GEMM1 (+3.5%)

> **Before attempting, read `code-examples/tcgen05-kernel-skeleton.md` in full**
> (and `code-examples/dual-tma-descriptors.md` before attempting item 8).
> This is the hardest item on the ladder. Most "tcgen05 didn't help" investigations
> are actually "compile target is missing `_a` suffix so tcgen05 was silently dropped"
> — check `tuning-knobs.md` compilation-flags section first.

### Context
CUTLASS is at the 14% occupancy wall (218 KB smem + 168 regs/thread = 1 CTA/SM). tcgen05 uses Tensor Memory (TMEM) instead of register accumulators, freeing register pressure and allowing higher occupancy.

### Implementation

Architecture:
- Persistent warp-specialized kernel, 6 warps:
  - 1 TMA producer (issues `cp.async.bulk` loads)
  - 1 MMA issuer (issues `tcgen05.mma` instructions)
  - 4 epilogue drain warps (TMEM → shared → global)
- 7-stage pipeline (multi-barrier producer-consumer)
- Tile shape: BM = BN = BK = 128
- `__launch_bounds__(192, 1)`
- 1 CTA/SM, using ~224 KB shared memory

Start from gau-nernst `matmul_v7` as a reference template. Adapt for MoE by:
- Linear scan of expert groups (G ≤ 32)
- Swizzle S=4 tile decode for L2 B-matrix locality
- Per-expert stride into A, B, SFA, SFB pointers

See [`code-examples/tcgen05-kernel-skeleton.md`](code-examples/tcgen05-kernel-skeleton.md) for the complete structure.

### When to apply
- GEMM is > 60% of kernel runtime
- CUTLASS NCU profile shows 14% occupancy and memory-bound
- Further gains from non-GEMM optimizations have been exhausted

### Pitfalls
- Requires `sm_100a` compile target; without `_a` suffix, tcgen05 is silently dropped
- Multi-barrier coordination has to be exactly right — one missed `mbarrier.arrive` hangs the kernel
- XL workloads (seq_len ≥ 12000) may lose 2–7% vs CUTLASS — keep CUTLASS as a fallback

---

## 8. tcgen05 on GEMM1 + GEMM2 (+3.3%)

### Context
After tcgen05 is working on GEMM1, GEMM2 still uses CUTLASS. GEMM2 has different shape than GEMM1 (different K and N), so a single TMA descriptor set cannot serve both.

### Implementation
Maintain **two independent TMA descriptor sets**:
- `d_Atm1`, `d_Btm1` — sized for GEMM1 inputs (rows × K1, E × N1 × K1)
- `d_Atm2`, `d_Btm2` — sized for GEMM2 inputs (rows × K2, E × N2 × K2)

At kernel launch, pass the appropriate descriptor pair based on which GEMM is being invoked.

See [`code-examples/dual-tma-descriptors.md`](code-examples/dual-tma-descriptors.md) for the descriptor setup.

### Observed landmarks
Adding tcgen05 to GEMM2 (on top of tcgen05 GEMM1) produces the **+3.3%** row in the ladder. Run-to-run variance is around ±0.6× at this level of the stack — multi-run averaging is required to confirm.

### Environment-variable dispatch control (example names)
```
NO_TCGEN05=1        # force fallback to CUTLASS
USE_TCGEN05=1       # force tcgen05 on all sizes
TCGEN05_MIN_T=500   # size threshold; below this, prefer CUTLASS
```

Reasonable default: tcgen05 for `T ≥ ~500`, CUTLASS for smaller T (where its static schedule wins). Verify the threshold on your workload distribution.

---

## 9. Scan-scatter fusion (+2.8%)

Combine the output-offset prefix scan into the scatter kernel itself. Reduces total launched kernels from ~12 to 9 on the critical path.

NCU metric: total kernel launches per MoE invocation drops.

---

## 10. uint4 (128-bit) scatter loads/stores (+2.1%)

Change scatter kernel from `float` or `half` loads to `uint4` (16-byte) loads. Reduces iterations 7 → 4 per element group and saturates DRAM bandwidth better.

```cpp
// Before
for (int i = 0; i < 7; i++) dst[i] = src[i];

// After
uint4 v = *reinterpret_cast<const uint4*>(src);
*reinterpret_cast<uint4*>(dst) = v;
// with boundary handling for non-multiple-of-4 elements
```

Ensure 16-byte alignment of buffers before using `uint4` loads.

---

## 11. T-dependent GEMM2 backend (+1.4%)

### Context
CUTLASS FP8 grouped GEMM excels at small-T (better occupancy at low tile counts). cuBLAS FP16 excels at large-T (better saturation). Neither wins at both ends.

### Dispatch rule
```
GEMM2 (smaller K, typically K ≤ ~2k) total tokens T?
├── T ≤ ~2000 → CUTLASS FP8 grouped
└── T >  ~2000 → cuBLAS FP16 (COMPUTE_32F_FAST_16BF)

GEMM1 (larger K, typically K ≥ ~4k): always FP8 — bandwidth win is non-negotiable
```

See [`code-examples/t-dependent-dispatch.md`](code-examples/t-dependent-dispatch.md) for the dispatch code.

### Pitfalls
- The ~2000 threshold is workload-dependent; measure on your actual T distribution
- Never disable FP8 on the large-K (bandwidth-heavy) GEMM — the loss of FP8 bandwidth advantage is too costly

---

## 12. PSS (`programmaticStreamSerialization`) (+0.7%)

### Context
Adjacent dependent kernels normally wait for host-side launch serialization. PSS allows the GPU to start the next kernel as soon as the prior kernel's grid completes, without a round-trip through host.

### Implementation

```cpp
cudaLaunchConfig_t cfg = {};
cfg.gridDim = ...;
cfg.blockDim = ...;
cfg.stream = stream;
cudaLaunchAttribute attr[1] = {};
attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attr[0].val.programmaticStreamSerializationAllowed = true;
cfg.numAttrs = 1;
cfg.attrs = attr;
cudaLaunchKernelEx(&cfg, my_kernel, args...);
```

Apply to any kernel whose output is immediately consumed by the next kernel on the same stream.

### Pitfalls
- Doesn't work across streams — for cross-stream overlap use events
- Gain is small; don't reach for this before items 1–8

---

## BF16 fast compute (always on, not in ladder)

This is a default setting, not an optimization step. Always enable for cuBLAS FP16 GEMM on B200:

```cpp
cublasGemmEx(..., CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT);
```

Without it, cuBLAS runs a slower FP32-accumulator path and your FP16 baseline is ~20% lower than it should be.
