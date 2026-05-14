# Data placement techniques

Given that data has been loaded, *where it lives during compute* is the next lever. This file covers register vs shared memory tradeoffs and the mechanical avoidance of shared-memory bank conflicts.

---

## Register-centric data path

**What it does.** Keeps hot data (query vectors, output accumulators, loop-invariant constants) in registers for the duration of the inner loop instead of streaming through shared memory. Shared memory is used only for data that must be communicated between threads.

**When it helps.**
- Inner loops that re-read the same tile across many iterations.
- Small-N dimensions (query length, head dim, etc.) that fit in a modest register footprint.
- Any kernel with low compute intensity per smem access.

**When it hurts.**
- When the "hot data" is too large to fit in registers without spilling (spills go to local memory, which is L1-cached but slow). Always check register count with `--ptxas-options=-v`.
- When data needs to be shared across threads of a block (then it must be in smem).

**How.**

```cuda
// BAD — query lives in smem, each thread reads its slice of smem in every KV iter
__shared__ bf16 q_smem[HEAD_DIM];
for (int k = 0; k < num_kv; k++) {
    float qk = dot(q_smem + lane * VEC, k_smem + k * HEAD_DIM + lane * VEC);
    ...
}

// GOOD — query in registers, loaded once
float q_reg[VEC];
#pragma unroll
for (int i = 0; i < VEC; i++) q_reg[i] = float(q_gmem[lane * VEC + i]);
for (int k = 0; k < num_kv; k++) {
    float qk = dot_inline(q_reg, k_smem + k * HEAD_DIM + lane * VEC);
    ...
}
```

For the output accumulator, the pattern is analogous: keep running `out_reg[VEC]` (and `max_reg`, `sum_reg` for online softmax) in registers until you write back at the end.

**Field note (DSA).** The biggest single family of wins in the DSA run came from this pattern:
- v3 — move output accumulator to registers (+2%)
- v7 — preload query to registers (+23%)
- v9 — load query directly global→register, skip smem (+20%)

Cumulative +78% from register-centric moves before a single parallelism trick was applied.

**Field note (MoE).** Late in the FuseMoE campaign, the routing kernel's per-lane `s[]` intermediate (a `__shared__` array used to stage group scores) was removed by recomputing it in registers via warp shuffles. Worth **+1.0%** on the all-19 average — small in isolation but the pattern matters as a generic move: any `__shared__` scratch in a routing/elementwise kernel whose elements are only read by the same lane that wrote them is a candidate for register-resident replacement. See `operators/moe/optimization-ladder.md` item #14.

---

## Right-sizing the register cache (a.k.a. register-resident row)

**What it does.** When each thread holds K *registers* worth of input data that's
reused across multiple passes (e.g. holding `uint4 xreg[K]` to avoid re-reading
X from HBM), the *size of K* has a big effect on occupancy. Smaller K → fewer
live registers per thread → more blocks per SM → more warps in flight to hide
latency. You want K just large enough to cover the reused data, not the
maximum that "fits".

**When it matters.** Any time the agent has moved from "re-read every pass" to
"register-held row" and is now tuning block size. The obvious first cache size
is often 2× what occupancy wants.

**The knob.** For a row of length `D` split across `blockDim.x` threads, each
thread holds `K = D / blockDim.x` elements (or `D / (blockDim.x * VECTOR_WIDTH)`
if vectorized). You can reduce K by *increasing* `blockDim.x`, or by reducing
the vectorization width. The right K is: "the smallest K that still lets me
cover the row without re-reading, given the reduction scheme and vector width."

**Rule of thumb for B200 / SM100.** At 128 regs/thread limit (the high end
without spills), a `uint4 xreg[4]` costs ~16 registers (4 × 4 uint32) plus
accumulators. Doubling `blockDim.x` from 256 to 512 halves per-thread cache
needs and typically gains 1.5–2× occupancy on bandwidth-bound kernels. Try it
explicitly as a tuning experiment once you have a register-resident version.

**How.**

```cuda
// BEFORE — each thread caches 4 uint4 (32 bf16). For D=4096 with 256 threads,
// D/(threads*8) = 2 — so 4 slots is 2x oversize.
constexpr int VEC_SLOTS = 4;
uint4 xreg[VEC_SLOTS];
#pragma unroll
for (int i = 0; i < VEC_SLOTS; i++) xreg[i] = __ldg(Xvec + tid + i * blockDim.x);

// AFTER — right-sized to actual need
constexpr int VEC_SLOTS = 2;       // D / (threads * 8) for D=4096, threads=256
uint4 xreg[VEC_SLOTS];
#pragma unroll
for (int i = 0; i < VEC_SLOTS; i++) xreg[i] = __ldg(Xvec + tid + i * blockDim.x);
```

Confirm with `--ptxas-options=-v` — you should see registers/thread drop and
`nvcc` reporting a higher max-blocks-per-SM in the launch info.

**When it hurts.**
- Don't go *below* actual need — then you re-read X from HBM and the whole
  register-resident optimization was pointless.
- On kernels where register count isn't the occupancy limiter (e.g. smem-bound),
  shrinking K doesn't help.
- Not all kernels allow the freedom; some have coupling between cache size and
  reduction layout that constrain K.

**Field note.** In the iter-2 softmax long-reduction run, shrinking the X cache
from 4 slots to 2 doubled achieved occupancy and gave +1.4× (the single largest
win in that run). This pattern recurs: after "register-hold the row" is in, the
next experiment is always "can I halve the cache size?"

**Counter-rule: validate each cache pays for itself.** "Right-size" means
keep as little cached as possible — not "cache everything you can in registers."
Every tensor you add to the register cache (X, exp values, weights, output
accumulators, bias) costs register footprint, which costs occupancy, which
costs latency hiding. For each new cache you add, run ONE experiment that
removes it alone to confirm it actually wins ≥1% over the "re-read from L1/L2"
alternative. If it doesn't, remove it.

Why this matters: B200's L1 (192 KB/SM) and L2 (126 MB) are large enough that
many "re-reads" of hot data are effectively free. Register footprint, by
contrast, is strictly budgeted (255 regs/thread max, occupancy drops sharply
past 64-72). On the iter-3 softmax baseline run, caching exp values across
passes in registers *regressed* the kernel by ~12% vs the simpler "re-compute
exp(x-max) in pass 2 and pass 3" approach — the L1 absorbed the re-computed
values for free, and the freed registers let occupancy double. Don't assume
caching always wins; prove it.

---

## Shared memory tiling

**What it does.** Stage a tile of data from global memory into shared memory once, then reuse it across many compute operations within the block. Classic for matmul and attention.

**When it helps.** Operations with tile-reuse factor > 1 (matmul reuses each A-tile across N output columns; attention reuses each K-tile across multiple rows; any reduction reuses each partial).

**When it hurts.**
- Tiles that are "reused" only once. You paid the load cost and the `__syncthreads` cost for zero benefit.
- When the tile can live in registers instead (see "register-centric" above).

**How.** Standard double-buffer pattern:

```cuda
__shared__ __align__(16) bf16 a_tile[2][TILE_M][TILE_K];
__shared__ __align__(16) bf16 b_tile[2][TILE_K][TILE_N];

// Load tile 0
load_async(a_tile[0], ...);
load_async(b_tile[0], ...);
__pipeline_commit();

for (int t = 0; t < num_tiles; t++) {
    if (t + 1 < num_tiles) {
        load_async(a_tile[(t+1)%2], ...);
        load_async(b_tile[(t+1)%2], ...);
        __pipeline_commit();
    }
    __pipeline_wait_prior(t+1 < num_tiles ? 1 : 0);
    __syncthreads();

    mma_compute(a_tile[t%2], b_tile[t%2]);
}
```

**Hardware notes.**
- Default shared memory per SM: 48 KB on Turing/consumer, 164 KB on A100, 228 KB on H100/B100, 228 KB opt-in max on B200.
- For > 48 KB per block, call `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)` at launch time.
- Cache that attribute setting — see `occupancy.md` for the "cached function attributes" trick.

**Field note (MoE).** The B200 FuseMoE GEMM kernels run hard up against the smem budget: the CUTLASS FP8 grouped GEMM uses **~201 KB dynamic shared memory** per block (plus ~1 KB driver overhead), and combined with **168 reg/thread**, this caps the SM at **1 CTA per SM → 14% achieved occupancy** on the slow workload — the famous wall the campaign documented. The hand-written tcgen05 variant sized to **224 KB smem** for its 7-stage pipeline, also 1 CTA/SM but with TMEM-backed accumulators it sustains higher effective throughput. See `operators/moe/tuning-knobs.md` and `operators/moe/optimization-ladder.md` (the "tcgen05 on both GEMMs" item).

---

## Bank conflict avoidance

**What it does.** Shared memory is organized as 32 banks of 4 bytes each (or 8 bytes on some arch). Two threads in a warp accessing the same bank serialize the access. Add a padding element to stride arrays and the conflicts disappear.

**When it helps.** Every time — if you have bank conflicts, eliminating them gives a 10–30% smem-access speedup. If you have none, no change.

**When it hurts.**
- Wastes smem (usually only a few percent).
- Padding can make vectorized loads mis-aligned — double check 16B alignment after padding.

**How (the "add 1" trick).**

```cuda
// Likely conflicts — row stride is 32, matches bank count
__shared__ float tile[32][32];

// No conflicts — row stride is 33
__shared__ float tile[32][33];
```

For 8-byte access (e.g., `__half2`, `bf16x2`), the bank width is 8 bytes, so you need different padding. CUTLASS's swizzle layouts handle this automatically — copy the pattern from `cutlass::layout::RowMajor2x2` or similar.

**Alternative: XOR swizzle.** Instead of padding (which wastes space), map `(row, col) → (row, col ^ row)` so consecutive rows read different banks for the same column. This is how CUTLASS avoids conflicts for tensor-core fragment loads.

```cuda
// Swizzle write and read consistently
int linear = row * 32 + col;
int swizzled = linear ^ ((linear >> 7) << 2);  // simplified; actual formula arch-specific
tile[swizzled] = value;
```

**Hardware notes.** Bank conflicts cost cycles on *all* NVIDIA GPUs. The bank width is 4 bytes except in specific 64-bit mode settings. Check NCU's "Shared Memory Bank Conflicts" metric.

**Field note (DSA).** XOR swizzle was attempted (V5 R4) but didn't show up as a meaningful win — the kernel's smem access was already conflict-free. Not every kernel needs this; profile first.

**Field note (MoE).** Bank-conflict avoidance was not a measurable lever in the MoE FuseMoE kernel — the FP8 GEMM path uses CUTLASS's `ComposedLayout<Swizzle<3, 4, 3>, ...>` which already handles conflicts; the auxiliary kernels (routing / scatter) use `uint4`-aligned access and 256-thread blocks where conflicts didn't show up in NCU. The catalog principle "profile before adding padding" generalised: the only place a hand-written swizzle was needed was the tcgen05 tile-decode (covered in `memory-access.md` field note for L2 swizzle).

---

## Cluster-shared memory (SM90+)

**What it does.** Groups multiple CTAs into a "cluster" that can see each other's shared memory via the distributed shared memory (DSMEM) address space. Effectively, a tile of several CTAs' smem acts like one large smem region accessible via async copy.

**When it helps.** Kernels where a tile size larger than one block's smem budget would help throughput, and 2–4 CTAs can share the work without too much coordination overhead. Large GEMMs on H100/B200 are the canonical use case.

**When it hurts.**
- CTAs in the cluster must fit on the same GPC (GPU processing cluster), which limits cluster size.
- Adds launch complexity and synchronization overhead. Not worth it for small kernels.

**How.**

```cuda
// Kernel with 2-CTA cluster, launched via cudaLaunchAttribute
// (See cuda-samples/Samples/3_CUDA_Features/clusterKernelCnvSampler)
__global__ void __cluster_dims__(2, 1, 1) kernel(...) {
    cg::cluster_group cluster = cg::this_cluster();
    __shared__ float tile[BIG_SIZE];
    cluster.sync();
    // Read peer's shared memory via cluster.map_shared_rank
    float* peer_tile = cluster.map_shared_rank(tile, peer_rank);
    ...
    cluster.sync();
}
```

**Hardware notes.** SM90+ only. On SM100, the cluster launch API is mature; CUTLASS uses it aggressively.

**Field note (DSA).** Not used in the DSA run — the kernel was small enough that single-CTA smem was sufficient. Most 1-level kernels won't benefit; this is for the final 5% on already-optimized large kernels.

**Field note (MoE).** Cluster mode was explicitly tested in the MoE FuseMoE campaign and **regressed −18%**. The CUTLASS SM100 collective has a `Shape<_2,_1,_1>` (2-SM cluster) variant available; on the long-seq workloads the 2-SM cluster halves the work-per-wave at K=2048, doubling the epilogue-to-mainloop ratio. 1-SM `KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100` is the only viable choice for this workload class. See `operators/moe/dead-ends.md` for the full regression record.

---

## Quick reference

| Hot question | Answer |
|---|---|
| Should my Q/acc live in smem or registers? | Registers, unless other threads need to see it |
| I have a 256×128 tile with bf16 access — padding? | `[256][129]` (or XOR swizzle) |
| smem > 48 KB — what do I need to do? | `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, bytes)` once, cached |
| I'm on B200 and my tiles are big — cluster smem? | Maybe, but try TMA + single-CTA first; cluster is the last 5% |
