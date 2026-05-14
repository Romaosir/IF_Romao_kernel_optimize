# Memory access techniques

Loading data well is usually the largest single lever in a CUDA kernel. Most first-generation kernels are bandwidth-bound; most of the techniques here turn that around.

---

## Coalesced global memory access

**What it does.** Guarantees that the 32 threads of a warp request 32 consecutive addresses in a single transaction, so the hardware can fetch them in one 128-byte memory transaction per warp.

**When it helps.** Every time. A warp that accesses 32 scattered addresses costs ~32× more memory transactions than a coalesced warp reading the same amount of data.

**When it hurts.** It doesn't — but getting it wrong silently halves or quarters your bandwidth.

**How.** Arrange your data so the fastest-varying index across a warp maps to the fastest-varying dimension of the tensor (typically the innermost stride-1 dim). If that's not possible because of how the input is shaped, transpose once into shared memory and read from there.

```cuda
// GOOD — tid is the innermost index, stride-1 access
int tid = threadIdx.x;
float4 v = *reinterpret_cast<const float4*>(&x[row * N + tid * 4]);

// BAD — tid indexes a stride-N dimension, scattered across warp
float4 v = *reinterpret_cast<const float4*>(&x[tid * N + col]);
```

**Hardware notes.** On all SM 7.0+ architectures, the coalescing unit is the warp (32 threads). A partial coalesce (e.g., warp accesses addresses within a 128-byte line but not stride-1) is still fine — the hardware does one 128B transaction. The pathological case is when a warp's addresses straddle multiple cache lines.

---

## Vectorized loads (`float4`, `__ldg`, 128-bit)

**What it does.** Each thread issues a 16-byte load instead of a 4-byte load, cutting the per-thread instruction count 4× and improving bandwidth utilization.

**When it helps.** When the data is aligned to 16 bytes and stride-1 within the warp. Common for dense tensors, typed activations, output writes.

**When it hurts.**
- Misaligned pointers → hardware silently issues two loads or (worse) the compiler inserts a slow path. Always ensure `__align__(16)` on shared-memory tiles and alignment on global pointers.
- Scattered (gather-style) access — each thread loading a different 16-byte chunk doesn't help much; the coalescer still works at warp granularity.
- Small reads where 4 elements is more than needed — now you're loading dead data.

**How.**

```cuda
// float4 load (4 × float or 8 × bf16)
float4 a = *reinterpret_cast<const float4*>(&x[i]);

// __ldg gives the compiler a hint to route through the read-only cache (harmless on recent arch)
float4 b = __ldg(reinterpret_cast<const float4*>(&x[i]));

// For bf16 tensors:
using bf16x8 = __nv_bfloat168;        // or uint4 as a stand-in for 16-byte bf16 pack
uint4 raw = *reinterpret_cast<const uint4*>(&bf16_tensor[i]);
```

**Field note (DSA).** Moving from scalar bf16 loads to `float4` on the CKV cache reads was a 5–10% gain. Aligning the kernel's shared-memory tile to 16B was required.

**Field note (MoE).** In the FuseMoE run, the pull-scatter kernel was rewritten to use `uint4` (128-bit) load/store on the row-output write-back; cut iteration count 7→4 per row group and contributed +2.1% on the all-19 average. Buffer 16-byte alignment was already in place from the upstream FP8 gather kernel — the change was purely the cast and inner-loop rewrite.

---

## Asynchronous copies (`cp.async`)

**What it does.** Issues a global → shared memory copy that doesn't block the thread. Combined with `__pipeline_wait_prior`, it enables multi-stage software pipelining — load tile N+1 into shared memory while the compute consumes tile N from a separate buffer.

**When it helps.** Any latency-bound or memory-bound kernel with a clear outer loop over tiles. Essentially mandatory for matmul-style and attention-style kernels on SM80+.

**When it hurts.**
- Stage count too high — each stage doubles the smem budget for the staged tensor. On small-smem GPUs (L4, consumer cards), 4 stages may exceed the budget.
- On SM100 B200, `cp.async` still works, but TMA is usually better for large contiguous tiles (see next section).

**How (2-stage double-buffered).**

```cuda
// Two shared-memory buffers
__shared__ __align__(16) bf16 tile[2][TILE_M][TILE_K];

// Prime the pipeline — start loading tile 0
__pipeline_memcpy_async(&tile[0][row][col], &gmem[0 + row * K + col], sizeof(bf16) * 8);
__pipeline_commit();

for (int t = 0; t < num_tiles; t++) {
    // Start loading tile t+1 (if any)
    if (t + 1 < num_tiles) {
        __pipeline_memcpy_async(&tile[(t+1)%2][row][col],
                                &gmem[(t+1)*TILE_K + row*K + col],
                                sizeof(bf16)*8);
        __pipeline_commit();
    }

    // Wait for tile t to land
    __pipeline_wait_prior(t + 1 < num_tiles ? 1 : 0);
    __syncthreads();

    // Compute on tile[t%2]
    compute(tile[t%2]);
}
```

**Hardware notes.**
- `cp.async.ca`: cache in L1. `cp.async.cg`: bypass L1 (for data that won't be reused by the same block through L1).
- SM80+ required.
- Use `__pipeline_wait_prior(N)` — wait until there are at most N outstanding commits. This is how you overlap.

**Field note (DSA).** The DSA kernel's 2-stage pipeline contributed +5% when first introduced (V2 v10); subsequent runs all kept it.

**Field note (MoE).** The hand-written tcgen05 GEMM in the FuseMoE kernel runs a **7-stage `cp.async.bulk.tensor` pipeline** with multi-barrier producer/consumer (6-warp split: 1 TMA producer + 1 MMA issuer + 4 epilogue drain warps). Pipeline depth 7 was the empirical sweet spot — 5 stages starved the issuer, 8 stages blew the shared-memory budget. See `code-examples/tcgen05-kernel-skeleton.md`.

---

## TMA and `cp.async.bulk` (SM90+)

**What it does.** Hardware Tensor Memory Accelerator — a dedicated copy engine that handles large descriptor-based transfers without tying up the SMs' address generation logic. One PTX instruction launches the whole transfer.

**When it helps.** Large tiles (≥ 1 KB), contiguous or strided-rectangular shapes, where the cp.async instruction overhead becomes a meaningful fraction of the load time. Matmul kernels on H100/B200 are the headline use case.

**When it hurts.**
- Scattered / gather-style access (e.g., sparse attention with index-based KV gather) — TMA's tile descriptors are rectangular. You'll work harder than with `cp.async`.
- Small tiles (≤ 256 B) — the descriptor setup cost dwarfs the transfer.
- Pre-SM90 hardware — not available.

**How.** Via CUTLASS's `cute::copy` with `cp_async_bulk_tensor` APIs, or hand-rolled PTX:

```cuda
// PTX sketch (SM90)
asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile"
             " [%0], [%1, {%2, %3}], [%4];\n"
             :: "r"(smem_ptr), "l"(tensor_map), "r"(coord_m), "r"(coord_n), "r"(mbar_ptr));
```

In practice, use CUTLASS or cuda-samples. Hand-rolled TMA is correct-tricky.

**Hardware notes.** SM90+ required. On SM100 B200, TMA is mature and the default for GEMM-shaped loads.

**Field note (DSA).** TMA was tried and reverted (V3) — the gather pattern didn't fit TMA's rectangular assumption. For dense matmul or attention with dense K, TMA is usually the right call on SM90+.

**Field note (MoE).** TMA is the foundation of the FuseMoE FP8 grouped-GEMM path — both the CUTLASS collective (`KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100`) and the hand-written tcgen05 kernel rely on `cp.async.bulk.tensor`. **Critical detail**: when one tcgen05 kernel template serves two GEMMs with different shapes (GEMM1: K=7168; GEMM2: K=2048), you need **two separate `CUtensorMap` descriptor pairs** — one per GEMM. Reusing one descriptor pair across both is the most common silent-correctness bug. See `code-examples/moe-dual-tma-descriptors.md`.

---

## L2 cache locality and tile swizzle

**What it does.** Reorders the block-index → tile-coordinate mapping so that blocks which run near each other in time access tiles that are near each other in memory, boosting L2 hit rate.

**When it helps.** Kernels where tiles of one operand are reused by many blocks (e.g., matmul blocks on the same row reuse the same A-tile). When L2 hit rate is < 50%.

**When it hurts.**
- Doesn't — the risk is spending time swizzling without profile evidence that L2 is underused.
- Excessive swizzling formulas can add instruction cost in the tile-index computation.

**How (matmul-style diagonal swizzle).**

```cuda
// Standard order: blocks along row of output matrix
int bid = blockIdx.x;
int bm = bid / N_TILES;
int bn = bid % N_TILES;

// Swizzled (group of WIDTH blocks fight over same A-tile)
constexpr int WIDTH = 8;
int group_id   = bid / (WIDTH * N_TILES);
int group_size = min(WIDTH, M_TILES - group_id * WIDTH);
int bm = group_id * WIDTH + (bid % group_size);
int bn = (bid % (WIDTH * N_TILES)) / group_size;
```

Or for flash-attention-style kernels, swap the grid dimensions so tiles of K/V are reused across sequential blocks.

**Field note (DSA).** The swap from `blockIdx = token * HEADS * SPLIT_K + head * SPLIT_K + split` → `blockIdx = token * SPLIT_K * HEADS + split * HEADS + head` improved L2 reuse and was worth +1–6%.

**Field note (MoE).** Two swizzle moves applied in the FuseMoE kernel: (1) the CUTLASS grouped-GEMM uses `args.scheduler.max_swizzle_size = 4` on every argument-array path — measured stable gain across all workloads; (2) the hand-written tcgen05 tile-decode also uses an **S=4 swizzle** (group 4 M-tiles together, iterate N inside) to keep the B-matrix columns in L2 across consecutive tiles. Without the tile-decode swizzle the long-seq workloads lose 3–4%.

---

## `__restrict__` on input pointers

**What it does.** Tells the compiler that two pointers do not alias (do not write through pointer A then read through pointer B expecting to see the write). This unlocks load reordering, CSE, and caching.

**When it helps.** Every kernel that reads from multiple input tensors.

**When it hurts.** If you lie — put `__restrict__` on pointers that do alias, and you get subtle wrong answers. In practice, almost no real kernel has aliased inputs because tensors are distinct allocations.

**How.**

```cuda
__global__ void kernel(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    bf16* __restrict__ out) { ... }
```

Add it to every input pointer in every kernel. Free performance.

---

## Quick reference: which to reach for

| Symptom (from NCU) | Try first |
|---|---|
| Uncoalesced loads reported | Coalescing |
| Many-instruction LD/ST hot | Vectorization |
| Long Scoreboard stalls dominant | `cp.async` pipelining |
| High DRAM %, low L2 hit rate | L2 tile swizzle |
| Same tile loaded many times | Shared memory staging + swizzle |
| Compiler emits duplicate loads | `__restrict__` |
| Large matmul on SM90+ with cp.async still bandwidth-bound | TMA |
