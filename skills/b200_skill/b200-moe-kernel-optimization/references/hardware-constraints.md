# B200 / SM100 Hardware Constraints

This reference covers the concrete hardware facts that shape every optimization decision in this skill. Every constraint below was directly observed in production MoE optimization work on B200.

## Compute capability targets

- **`sm_100`** — generic Blackwell target. Usable for most features but **misses** architecture-specific instructions like tcgen05.
- **`sm_100a`** / **`compute_100a`** — architecture-specific. Required for tcgen05, any TMEM operation, and any PTX that references Blackwell-only features.

If you compile tcgen05 code without `_a` suffix, nvcc will accept the source but the generated SASS will not contain the instruction — it falls back to alternative codegen and your "optimized" kernel quietly runs slower than intended.

---

## FP8 format compatibility matrix

| Format | B200 native | Scale granularity | Scale dtype | Typical use |
|---|---|---|---|---|
| **MXFP8** | Yes (required for native tensor core) | 32-element block | `e8m0fnu` | B200 native FP8 GEMM |
| **Hopper-style blockwise FP8** | No | 128×128 block | `float32` | Needs custom CUTLASS epilogue or hand-written tcgen05 |
| **Per-tensor FP8** | No (for data quantized blockwise) | Single scalar | `float32` | Re-quantization to this format introduces ~12.5% per-element error |

### Why the format mismatch matters

Most modern pretrained weights (DeepSeek-V3, Kimi-K2, etc.) are quantized with **128-block float32 scales** (Hopper-style). B200's native FP8 tensor-core instructions require **MXFP8** (32-block, e8m0fnu).

This means B200 cannot run such weights natively without one of:

1. **Custom CUTLASS epilogue** that applies 128-block scales during matmul write-back
2. **Hand-written tcgen05** kernel that handles the scale layout in shared memory
3. **Warmup-time format conversion** (adds startup cost; cold-path vs warm-path matters)
4. **Re-quantization to per-tensor scales** → introduces error that exceeds typical tolerance

Approaches 1 and 2 are what this skill recommends. Approach 4 is **never correct** for realistic tolerance targets.

---

## cuBLAS FP8 path status on B200

**Every cuBLAS FP8 path fails in some way on B200.** Confirmed on cuBLAS 13.2.1.

| Path | Status | Root cause |
|---|---|---|
| `cublasGemmBatchedEx` (pointer-array) | SIMT fallback — **no tensor cores** | API dispatches SIMT sgemm for FP8 in all compute types; ends up slower than FP16 |
| `cublasGemmStridedBatchedEx` | NVJet tensor core (5.9× GEMM speedup) — but **correctness fails on a subset of workloads** | Grow-only buffer stale padding rows (see `fp8-correctness-modes.md` mode 3) |
| `cublasLtMatmul` + `BLK128x128_32F` | `CUBLAS_STATUS_NOT_SUPPORTED` | cuBLAS 13.x runtime returns 0 algorithms for this mode |
| `cublasGemmGroupedBatchedEx` | **RUNTIME_ERROR on most workloads** | API crashes on B200 with FP8 |

### What this means for the skill

Don't waste time exploring cuBLAS FP8 paths. The compatibility was exhaustively tested; all four APIs fail. **Use CUTLASS for FP8 on B200**, or hand-written tcgen05 for the last bit of performance.

cuBLAS FP16 still works correctly — it's a valid baseline and fallback path for large-T GEMM2 (see `decision-trees.md` for the T-dependent dispatch rule).

---

## CUTLASS SM100 constraints

### M-tile shape constraint

- Supported for 1-SM grouped GEMM: `M ∈ {64, 128}`
- `M = 256` requires 2-SM cluster
- **2-SM cluster was empirically −18%** vs 1-SM on SM100 with this workload class
- **Do not use M ≥ 256** for grouped MoE GEMM on B200

### N-tile shape constraint

- `N = 128` works reliably
- `N = 256` (e.g., 128×256×128 or 64×256×128) **regresses 60%** on MoE workloads — N-tile too wide for typical MoE expert shapes
- Keep N at 128

### Occupancy wall

A working CUTLASS FP8 grouped GEMM collective on SM100 uses approximately:

- **168 registers / thread**
- **218 KB shared memory per block**

Result: only **1 CTA per SM** → occupancy ~14%. This is the wall that motivates moving to tcgen05.

### Working collective for blockwise FP8 MoE

```
KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100
```

This is the confirmed-working SM100 grouped GEMM collective for 128-block float32 FP8. The 2SM variant (`...2SmSm100`) did not decisively beat 1SM in measurements.

### JIT compilation flags

```
nvcc -std=c++17 \
     -gencode arch=compute_100a,code=sm_100a \
     -O2 \
     --shared -Xcompiler -fPIC \
     <source.cu> \
     -o /tmp/<name>.so \
     -lcuda
```

- Cache the resulting `.so` at a stable path (e.g., `/tmp/`)
- Use `dlopen` + `dlsym` to load at runtime
- Note: first compile takes tens of seconds; cache hits are fast

---

## tcgen05 (5th-generation Blackwell tensor core)

### Why it matters

tcgen05 uses **Tensor Memory (TMEM)** as the MMA accumulator instead of registers. This releases register pressure, breaks the 14% occupancy wall, and is the primary path for exceeding CUTLASS throughput on B200 blockwise FP8.

### Key PTX instructions

| Instruction | Purpose |
|---|---|
| `tcgen05.alloc.cta_group::<g>.sync.aligned.shared::cta.b32` | Allocate TMEM slot |
| `tcgen05.dealloc.cta_group::<g>.sync.aligned.b32` | Release TMEM slot |
| `tcgen05.mma.cta_group::<g>.kind::f8f6f4` | FP8 MMA on TMEM accumulator |
| `tcgen05.commit.cta_group::<g>.mbarrier::arrive::one.shared::cluster.b64` | Signal MMA done via mbarrier |
| `cp.async.bulk.tensor.<N>d.shared::cluster.global.mbarrier...` | TMA load with mbarrier arrival |
| `mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64` | Producer arrival with byte-count expectation |

### Scale layouts supported

- **128×128 float32 blockwise** (Hopper-compatible) — what real workloads typically use
- **32-element UE8M0** (MXFP8) — B200 native layout; less common in pretrained weights

For a complete tcgen05 persistent warp-spec kernel skeleton, see [`code-examples/tcgen05-kernel-skeleton.md`](code-examples/tcgen05-kernel-skeleton.md).

### 1-SM vs 2-SM

Both variants tested in production. **1-SM was consistently faster** for MoE grouped FP8 on B200.

### Reference template

An open-source starting point that works in practice: **gau-nernst `matmul_v7`**, which achieves ~1438 FP8 TFLOPS (~98% of cuBLAS FP8 peak) in isolation. It can be adapted into a MoE kernel as GEMM1 and GEMM2 backends.

For adapting this template to MoE-grouped usage, the key changes are:
- Linear scan of expert groups (with `G ≤ 32`, linear scan beats binary search due to fewer branch mispredictions)
- Swizzle `S=4` tile-decode ordering for L2 B-matrix locality across consecutive tiles
- Separate TMA descriptor pairs for GEMM1 and GEMM2 (see the "tcgen05 on GEMM1 + GEMM2" section in `optimization-ladder.md`)

---

## B200 memory hierarchy

Relevant numbers for kernel sizing:

| Resource | Limit |
|---|---|
| SM count | 148 |
| Shared memory per SM | 228 KB |
| Max shared memory per block (opt-in) | ~227 KB |
| Registers per SM | 64 K (32-bit) |
| Max threads per SM | 2048 |
| Max blocks per SM | 32 |
| Tensor Memory (TMEM) per SM | 512 entries × 128 bits = 8 KB |

The 14% occupancy wall comes from the combination of 168 reg/thread + 218 KB smem — once a CUTLASS block is using 218 KB, no second CTA can fit (228 KB total), and the 168 reg/thread pins each thread to one SM slot.

---

## Measurement noise on B200

- **Run-to-run variance: ~3–4%** on isolated GPU with consistent workload
- **Multi-GPU contention: up to ±30%** when agents on different GPUs share host resources
- **Cold vs warm compilation: up to 2×** if `.so` JIT is timed as part of the kernel

These numbers directly motivate the baseline re-measurement rule in `agent-team-patterns.md`.

---

## Quick-reference: constraint violations

| Symptom | Likely constraint |
|---|---|
| `CUBLAS_STATUS_NOT_SUPPORTED` on FP8 path | cuBLAS 13.x has no B200 algorithm for that FP8 mode |
| Kernel builds but runs slowly | Missing `sm_100a` target — tcgen05 code silently dropped |
| Grouped GEMM shows 14% SM occupancy and memory-bound | CUTLASS is at the shmem+register wall; next step is tcgen05 |
| GEMM 60% slower than baseline after adding wider tile | N-tile too wide; stay at N=128 |
| Multi-workload sequence produces correctness failures | Likely stale padding in grow-only buffer (see fp8-correctness-modes mode 3) |
