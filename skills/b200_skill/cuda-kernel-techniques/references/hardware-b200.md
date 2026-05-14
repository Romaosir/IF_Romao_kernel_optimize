# B200 (Blackwell, SM100) hardware facts

A single-page reference of B200 hardware parameters and platform-level constraints that frame every optimisation decision on the chip. Operator-agnostic — use alongside the technique files in this catalog.

For the operator-specific consequences of these facts (e.g. "the FuseMoE FP8 grouped GEMM hits the occupancy wall at 14% because of this specific collective"), see `references/operators/<operator>/`.

---

## Compute capability and compile targets

| Target | What it covers | When to use |
|---|---|---|
| `sm_100` / `compute_100` | Generic Blackwell. Compiles, but **silently omits architecture-specific instructions** (notably `tcgen05.*`, TMEM allocation). | Only for code that doesn't use SM100-only features. |
| **`sm_100a` / `compute_100a`** (note the `a`) | Architecture-specific. Required for `tcgen05`, any TMEM operation, any PTX referencing Blackwell-only features. | **Use this for anything beyond plain SM90 carry-over.** |

**Silent failure mode.** `nvcc` accepts source containing `tcgen05.mma`, `tcgen05.alloc`, etc. when compiled with `-arch=compute_100,code=sm_100` (no `a`), but the generated SASS does **not** contain the actual instruction — it falls back to alternative codegen. The kernel runs correctly, just slowly, and NCU does not flag the issue. Verify with `cuobjdump --dump-sass | grep TCGEN05` after a build.

---

## Memory hierarchy

| Resource | Limit |
|---|---|
| SM count | 148 |
| Shared memory per SM | 228 KB |
| Max shared memory per block (opt-in) | ~227 KB |
| Registers per SM | 64 K (32-bit) |
| Max threads per SM | 2048 |
| Max blocks per SM | 32 |
| Tensor Memory (TMEM) per SM | 512 entries × 128 bits = 8 KB |
| DRAM peak | ~8 TB/s (HBM3e) |
| L2 cache | 240 MB total (shared across SMs) |

**Opt-in shared memory > 48 KB**: kernel must call `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)` once at init. Caching the flag in a `static bool` matters on small-T workloads — fresh attribute setting per launch is a measurable overhead. See `occupancy.md`.

**The 14% occupancy wall (CUTLASS SM100 grouped FP8 example).** A working FP8 blockwise grouped GEMM collective uses ~168 reg/thread + ~218 KB shared memory. Combined: only 1 CTA fits per SM (228 KB total budget, ~168 reg ≈ 32K used of 64K). Achieved occupancy ~14%. This is the wall that motivates hand-rolled tcgen05 for that kernel class — TMEM-backed accumulators relax the register pressure. Other collectives have different occupancy profiles; do the same calculation for yours.

---

## cuBLAS FP8 on B200 — all four paths fail

Confirmed on cuBLAS 13.2.1. Operator-independent (the API is the issue, not the workload).

| Path | Status | Root cause |
|---|---|---|
| `cublasGemmBatchedEx` (pointer-array) | SIMT fallback — **no tensor cores** | API dispatches SIMT sgemm for FP8 in every compute type; slower than FP16. |
| `cublasGemmStridedBatchedEx` | NVJet tensor core (5.9× GEMM) — but correctness fails on a subset of workloads | Grow-only buffer stale-padding bug. |
| `cublasLtMatmul` + `BLK128x128_32F` | `CUBLAS_STATUS_NOT_SUPPORTED` | cuBLAS 13.x runtime returns 0 algorithms for this mode. |
| `cublasGemmGroupedBatchedEx` | RUNTIME_ERROR on most workloads | API crashes on B200 with FP8. |

**Implication.** For FP8 on B200, the practical choices are: **CUTLASS** (custom collective with the blockwise scaling format you need) or **hand-written tcgen05**. cuBLAS FP16 still works correctly and is a valid baseline or fallback path.

---

## tcgen05 (5th-generation tensor core, SM100a)

The instruction family that makes B200 distinct from H100. Uses Tensor Memory (TMEM) as the MMA accumulator, freeing register pressure.

### Key PTX

| Instruction | Purpose |
|---|---|
| `tcgen05.alloc.cta_group::<g>.sync.aligned.shared::cta.b32` | Allocate TMEM slot |
| `tcgen05.dealloc.cta_group::<g>.sync.aligned.b32` | Release TMEM slot |
| `tcgen05.mma.cta_group::<g>.kind::f8f6f4` | FP8 MMA on TMEM accumulator |
| `tcgen05.mma.cta_group::<g>.kind::f16` | BF16/FP16 MMA on TMEM accumulator |
| `tcgen05.commit.cta_group::<g>.mbarrier::arrive::one.shared::cluster.b64` | Signal MMA done via mbarrier |
| `cp.async.bulk.tensor.<N>d.shared::cluster.global.mbarrier.complete_tx::bytes` | TMA load with mbarrier arrival |
| `mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64` | Producer arrival with byte-count expectation |

### CTA grouping

- `cta_group::1` — single-CTA tcgen05. Default and most common.
- `cta_group::2` — 2-SM cluster cooperating on one MMA. Useful in theory; empirically slower on grouped-FP8 MoE workloads (−18% measured) because wave quantization halves work-per-wave and doubles the epilogue-to-mainloop ratio at the typical K. Test on your workload before committing.

### Scale layouts supported

- **128×128 float32 blockwise** (Hopper-compatible) — what pretrained-weight datasets typically use.
- **32-element UE8M0** (MXFP8) — B200's native layout. Less common in real workloads.

### Reference template

Open-source starting point: **gau-nernst `matmul_v7`** — achieves ~1438 FP8 TFLOPS in isolation (~98% of cuBLAS FP8 peak). Adaptations needed for grouped semantics: linear scan of expert/group IDs (`G ≤ 32` → linear beats binary-search), S=4 tile-decode swizzle for L2 B-matrix locality, separate TMA descriptor pairs per distinct (K, N) shape.

For a complete persistent warp-spec kernel skeleton, see [`code-examples/tcgen05-kernel-skeleton.md`](code-examples/tcgen05-kernel-skeleton.md).

---

## CUTLASS SM100 tile-shape constraints

For grouped-batched GEMM collectives at `cta_group::1`:

- M-tile: 64 or 128 (1-SM). Tiles ≥ 256 require 2-SM cluster.
- N-tile: 128 reliable. **256 regresses heavily** (60% slower on typical workloads — wave quantization at the inner contraction-K cost). Keep N at 128.
- K-tile: 128 standard.

The 2-SM cluster variants (`Shape<_2,_1,_1>`) are available in the CollectiveBuilder but in practice rarely beat 1-SM on B200 for grouped workloads.

---

## Measurement noise on B200

These numbers shape how you should design any benchmarking discipline (see also `cuda-kernel-autodev/experiment-loop.md`):

- **Run-to-run variance: ~3–4%** on isolated GPU with consistent workload
- **Multi-GPU contention: up to ±30%** when agents on different GPUs share host resources (NUMA, PCIe)
- **Cold vs warm compilation: up to 2×** if a `.so` JIT compile is timed as part of the kernel — always pre-warm

**Mitigation:** N ≥ 3 runs averaged for any decision near the noise floor (< 5% delta); same GPU pinned via `CUDA_VISIBLE_DEVICES`; isolated machine if possible.

---

## Quick-reference: symptom → constraint

| Symptom in NCU / build | Likely constraint |
|---|---|
| `CUBLAS_STATUS_NOT_SUPPORTED` on FP8 path | cuBLAS 13.x has no B200 algorithm for that FP8 mode — switch to CUTLASS |
| Kernel builds but runs the speed of an SM90 codegen | `compute_100` instead of `compute_100a` — `tcgen05` got silently dropped |
| Grouped GEMM at 14% SM occupancy and memory-bound | At the smem+register wall — only way through is tcgen05 with TMEM accumulator |
| GEMM 60% slower after widening N-tile | N=256 wave quantization; revert to N=128 |
| Cold start much slower than warm | `.so` JIT being timed — pre-warm or embed CUTLASS statically |
| Bench number changes by 30% between runs | Probably GPU contention; pin and re-isolate |

---

## Related reading

- For *which* compile flags, what runtime API, how to JIT-compile and `dlopen` CUTLASS on this hardware → [`code-examples/cutlass-jit-compile.md`](code-examples/cutlass-jit-compile.md)
- For a complete hand-written tcgen05 kernel scaffolding → [`code-examples/tcgen05-kernel-skeleton.md`](code-examples/tcgen05-kernel-skeleton.md)
- For zero-sync host-to-GPU dispatch using `griddepcontrol` PTX + `programmaticStreamSerialization` → [`code-examples/zero-sync-fast-path.md`](code-examples/zero-sync-fast-path.md)
- For the operator-specific occupancy wall and the path through it on FuseMoE → [`operators/moe/optimization-ladder.md`](operators/moe/optimization-ladder.md)
- For how to discover these facts on a new chip (Hopper successor, etc.) → [`cuda-kernel-autodev/references/hardware-discovery.md`](../../cuda-kernel-autodev/references/hardware-discovery.md)
