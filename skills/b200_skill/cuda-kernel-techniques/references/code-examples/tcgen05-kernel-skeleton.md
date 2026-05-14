# tcgen05 Kernel Skeleton

Structure of a hand-written tcgen05 persistent warp-specialized FP8 grouped GEMM for B200 SM100. This is the template that produces the +3.5% / +3.3% gains listed in the optimization ladder.

## Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│  Persistent grid: 1 CTA per SM, 148 CTAs total on B200     │
│                                                              │
│  Each CTA has 6 warps (192 threads):                        │
│    Warp 0: TMA producer    (issues cp.async.bulk loads)     │
│    Warp 1: MMA issuer      (issues tcgen05.mma)             │
│    Warps 2-5: Epilogue drain (TMEM → shared → global)       │
│                                                              │
│  Each CTA loops over tiles:                                 │
│    while (tile_queue_has_work):                             │
│      tile = decode_next_tile(tile_idx)                      │
│      producer: load A, B for this tile via TMA              │
│      issuer: fire tcgen05.mma, wait for completion          │
│      epilogue: drain TMEM → gmem                            │
│                                                              │
│  Pipeline depth: 7 stages                                   │
│  Tile shape: BM=128, BN=128, BK=128                         │
└─────────────────────────────────────────────────────────────┘
```

## PTX primitives

### mbarrier helpers

```cpp
__device__ __forceinline__ void mbar_init(int mbar_addr, int thread_count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(thread_count));
}

__device__ __forceinline__ void mbar_wait(int a, int p, int t = 10000000) {
    asm volatile(
        "{\n\t.reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t}"
        :: "r"(a), "r"(p), "r"(t));
}

__device__ __forceinline__ void mbar_arrive_tx(int a, int expected_bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
        :: "r"(a), "r"(expected_bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(int a) {
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
        :: "r"(a) : "memory");
}
```

### TMA helpers

```cpp
template <int G = 1>
__device__ __forceinline__ void tma_g2s(
    int smem_dst, const void* tma_desc,
    int x, int y, int z,     // 3-D tile coordinates
    int mbar)
{
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::%6 "
        "[%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(smem_dst), "l"(tma_desc),
           "r"(x), "r"(y), "r"(z), "r"(mbar), "n"(G)
        : "memory");
}
```

### tcgen05 helpers

```cpp
// Allocate TMEM columns
template <int G = 1>
__device__ __forceinline__ void tc_alloc(int smem_addr, int ncols) {
    asm volatile(
        "tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(smem_addr), "r"(ncols), "n"(G));
}

// Release TMEM
template <int G = 1>
__device__ __forceinline__ void tc_dealloc(int tmem_addr, int ncols) {
    asm volatile(
        "tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
        :: "r"(tmem_addr), "r"(ncols), "n"(G));
}

// FP8 MMA: D += A * B
template <int G = 1>
__device__ __forceinline__ void tc_mma_f8(
    int tmem_d,          // TMEM accumulator address
    uint64_t desc_a,     // descriptor for A operand in smem
    uint64_t desc_b,     // descriptor for B operand in smem
    uint32_t inst,       // instruction descriptor (encoded shape)
    int accumulate)      // 0 = overwrite, 1 = accumulate
{
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::%5.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
        :: "r"(tmem_d), "l"(desc_a), "l"(desc_b),
           "r"(inst), "r"(accumulate), "n"(G));
}

// Commit: signal MMA completion via mbarrier
template <int G = 1>
__device__ __forceinline__ void tc_commit(int mbar) {
    asm volatile(
        "tcgen05.commit.cta_group::%1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(mbar), "n"(G) : "memory");
}

// Descriptor encoding (smem offset → tcgen05 descriptor bits)
__device__ __forceinline__ constexpr uint64_t desc_enc(uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;
}
```

## Tile decode — structure (write this yourself)

For MoE grouped GEMM with `G ≤ 32` experts:

1. **Linear scan** over groups to find which group owns `tile_idx` — with G ≤ 32, linear scan beats binary search because of fewer branch mispredictions
2. Compute the tile-local index within the group (`tile_idx − tile_offsets[group]`)
3. Decode `(m_tile, n_tile)` from the local index using a **swizzle of stride `S = 4`** (group 4 M-tiles together, iterate N within the group, wrap back) — this keeps B-matrix columns in L2 across consecutive tiles
4. Fall back to plain `(m / grid_n, m % grid_n)` when the last group has fewer than `S` M-tiles

Return `{group_id, m_tile, n_tile, m_offset, group_M}`.

## Main kernel — structure (write this yourself)

Constants (measured sweet spots — see `tuning-knobs.md`):

```
BM = BN = BK = 128
NUM_STAGES  = 7
TB_SIZE     = 192      // 6 warps × 32 threads
__launch_bounds__(192, 1)
```

### Kernel signature
Takes: two `CUtensorMap*` (one for A, one for B), BF16 output pointer, `scale_a` and `scale_b` pointers, `m_indptr` (per-group row offsets), `expert_ids` (per-group weight-slice index), dims `G, N, K`.

### Kernel body (high-level)

```
entry:
  griddepcontrol.wait / launch_dependents PTX (enables PSS chaining)

  shared-memory layout (compute offsets at build time):
    - A staging: NUM_STAGES × BM × BK bytes (FP8)
    - B staging: NUM_STAGES × BN × BK bytes (FP8)
    - mbarriers: NUM_STAGES each for A-ready, B-ready, MMA-done
    - scale staging + output staging
  Total: must fit in per-SM shared memory (224 KB budget)

  warp 0  (thread 0-31)     → TMA producer
  warp 1  (thread 32-63)    → MMA issuer
  warps 2-5 (thread 64-191) → epilogue drain

  init all mbarriers (lane 0)
  allocate TMEM slot (lane 0 of warp 0) — keep for the whole CTA lifetime

  persistent loop over tile_idx = blockIdx.x, stride gridDim.x:
    ti = decode_tile(tile_idx)

    producer warp:
      for k = 0 to K step BK:
        stage = (k/BK) % NUM_STAGES
        wait on mbar_MMA[stage] so the previous MMA using this slot is done
        (lane 0) issue two TMA loads — one A tile, one B tile — each
        announcing expected bytes on mbar_A[stage] / mbar_B[stage]

    issuer warp:
      for k = 0 to K step BK:
        stage = (k/BK) % NUM_STAGES
        wait on mbar_A[stage] and mbar_B[stage]
        (lane 0) build descriptors for A, B; issue tcgen05.mma
        commit on mbar_MMA[stage] so the producer can refill

    epilogue warps:
      after full K accumulation, read TMEM → smem, apply blockwise scales
      (SFA for this row range, SFB for this expert's block), write BF16 out

  free TMEM slot before kernel exit
```

### Correctness details you have to get right

- mbarrier parity bit alternates every `NUM_STAGES` iterations — the wait pattern is `(k/BK / NUM_STAGES) & 1`. Getting this wrong reads stale data without crashing
- TMA byte-count must match the tile size exactly in `mbar_arrive_tx`
- Writing to TMEM between `tc_alloc` and `tc_dealloc` — any other ordering produces UB
- Only lane 0 of a warp may issue `tma_g2s` / `tc_mma_f8` / `tc_commit`
- `griddepcontrol.wait` must appear before any shared-memory use, otherwise the wait is against the wrong predecessor

### Pitfalls specific to writing this from scratch

- **`sm_100a` compile target** — without the `_a` suffix, every tcgen05 instruction is silently dropped. The kernel will compile, run, produce correct output, and be slow. Verify with `cuobjdump --dump-sass`.
- **Stage count vs smem budget** — with 7 stages, smem is ~224 KB. Going to 8 stages blows the budget; 5 starves the pipeline.
- **Launch-bounds too tight** — 192 threads × 1 block/SM = ~340 regs/thread headroom. Tighter bounds cause register spill to local memory.

## Dual TMA descriptors

When the second GEMM has different shape (different K or N), maintain two descriptor pairs and select at dispatch. See [`dual-tma-descriptors.md`](dual-tma-descriptors.md) for the field-by-field descriptor specification.

## Launch setup — structure

At the host-side launch function:

1. Check the TMA descriptors have been built (guard a setup-done flag)
2. Compute shared-memory size needed from `NUM_STAGES, BM, BN, BK` plus mbarrier slots
3. Call `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)` — required for smem > 48 KB
4. Configure a `cudaLaunchConfig_t` with:
   - `gridDim = full SM count` (persistent kernel)
   - `blockDim = TB_SIZE` (192)
   - `dynamicSmemBytes = smem_bytes`
   - A `cudaLaunchAttributeProgrammaticStreamSerialization` attribute for PSS chaining with the prior/next kernel
5. Launch with `cudaLaunchKernelEx`

The launch function returns 0 on success, negative error codes on setup failure — so callers can fall back to the CUTLASS path when tcgen05 is unavailable.

## Reference template

A practical starting point is the **gau-nernst `matmul_v7`** open-source BF16 example. Adaptations for MoE FP8:

1. Change MMA instruction from BF16 (`tcgen05.mma.kind::f16`) to FP8 (`kind::f8f6f4`)
2. Add blockwise scale application in the epilogue (128×128 float32 scales)
3. Add linear expert group scan + tile swizzle
4. Add dual TMA descriptor support for GEMM1 vs GEMM2

## Pitfalls

| Pitfall | Symptom |
|---|---|
| Forgot `sm_100a` (used `sm_100`) | Kernel compiles but runs slower than CUTLASS — tcgen05 silently dropped |
| Missed an `mbarrier.arrive` | Kernel hangs (consumer waits forever) |
| Wrong mbarrier parity in `mbar_wait` | Stale data consumed, correctness failures |
| `tc_alloc` / `tc_dealloc` mismatch | TMEM leak, second invocation fails |
| Descriptor encoding off | Wrong data loaded into MMA |
| `__launch_bounds__` too tight | Compiler can't fit required registers; kernel runs slower |
| Stage count vs smem capacity mismatch | Kernel launch fails with smem-too-large |

## NCU verification

- SM occupancy rises above 14% (up to ~30% typical for this kernel shape)
- TMEM usage reported
- DRAM throughput remains high (this is still memory-bound)
- No stalls on `mbarrier` — indicates pipeline depth is sufficient
- Kernel name in timeline matches (confirm no silent fallback)

If occupancy is still 14%, the tcgen05 kernel isn't providing what it should — recheck `__launch_bounds__` and shared memory sizing.
