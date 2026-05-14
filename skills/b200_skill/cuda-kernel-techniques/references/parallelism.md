# Parallelism techniques

How the work is divided across SMs, blocks, warps, and threads. The largest algorithmic wins usually live here.

---

## Split-K / FlashDecoding

**What it does.** Splits the reduction dimension across multiple CTAs. Each CTA computes a *partial* result (partial output + per-partial normalization stats). A second "merge" kernel combines the partials into the final answer using online softmax.

**When it helps.**
- Decoding-style attention: few queries, many keys — one query per block leaves the GPU mostly idle.
- Small-batch matmul where the M/N dimensions don't produce enough CTAs to fill the SMs.
- Any kernel where the outer parallelism (tokens / rows / batches) is < `num_SMs × 2` and the reduction dimension is long.

**When it hurts.**
- When outer parallelism already fills the GPU — Split-K just adds merge overhead.
- When the merge kernel launch overhead > the savings. Short reductions usually don't justify it.
- When `SPLIT_K` is too large — partial-output memory blows up, merge cost dominates.

**How (two-kernel pattern).**

```
Kernel 1 (Partial):
  grid = num_outer × SPLIT_K
  each CTA processes 1/SPLIT_K of the reduction range
  writes: partial_out[outer, split, out_dim], partial_lse[outer, split]

Kernel 2 (Merge):
  grid = num_outer
  reads: all SPLIT_K partials for one `outer` index
  applies: global_max = max over k of max[k]
           final = sum_k (exp(max[k] - global_max) * partial_out[k]) / sum_k (exp(max[k] - global_max) * exp(lse[k]))
  writes: final_out, final_lse
```

**Tuning SPLIT_K.** Start with `SPLIT_K = min(max(1, num_SMs / num_outer), reduction_dim / min_chunk)`. Sweep powers of 2 (2, 4, 8, 16, 32). Too large → merge dominates. Too small → SMs idle.

**Hardware notes.** Nothing arch-specific — this is an algorithmic pattern. The speedup vs baseline depends on how much of the GPU was idle without it.

**Field note.** From the DSA run, this was the single largest family of optimizations:
- v17 (V2): SPLIT_K=8 → 30.09x (+108%)
- v19 (V2): SPLIT_K=16 → 36.78x (+22%)
- v2 (V3): SPLIT_K=32 → 58.5x (+59%)
- v31 (V4): Adaptive SPLIT_K per token count → 127.93x

The adaptive variant (different SPLIT_K for different input shapes) is the natural culmination.

---

## Adaptive dispatch

**What it does.** The host-side launcher inspects input shape and selects among multiple kernel variants (or multiple `SPLIT_K` values for the same kernel). Small inputs get one code path; large inputs get another.

**When it helps.** Benchmarks with highly variable shapes (batch size ranging from 1 to 128, sequence lengths from 64 to 4096). Average speedup hides pathologies on the extremes; adaptive dispatch lets you fix them independently.

**When it hurts.**
- Adds code complexity and more paths to maintain.
- Compilation time grows with number of variants.
- If the user cares about a single shape, this is overkill.

**How.**

```cuda
// Host-side
void launch(int num_tokens, int num_kv, ...) {
    if (num_tokens <= 2) {
        // Single-pass kernel, no Split-K
        single_pass<<<grid1, block1, smem1, stream>>>(...);
    } else if (num_tokens <= 4) {
        split_k_kernel<SPLIT_K=4><<<grid2, block2, smem2, stream>>>(...);
    } else if (num_tokens <= 8) {
        split_k_kernel<SPLIT_K=16><<<grid3, block3, smem3, stream>>>(...);
    } else {
        split_k_kernel<SPLIT_K=32><<<grid4, block4, smem4, stream>>>(...);
    }
}
```

**Field note (DSA).** Adaptive SPLIT_K was the key to getting from 78x (uniform K=16) to 128x in V4 of the DSA run. Later, per-workload NCU profiling in V5 showed the 2-token workload still had 6% occupancy — adaptive dispatch added a dedicated SPLIT_K=32 path for that case, worth +7% on the geo-mean.

**Field note (MoE).** Adaptive dispatch is a load-bearing pattern in the FuseMoE kernel at *two* levels:
- **Backend dispatch by total-tokens T:** for GEMM2, `T ≤ 2000 → CUTLASS FP8 grouped`, `T > 2000 → cuBLAS FP16` with pre-dequantized BF16 weights. Worth +1.4% on the all-19 average and required for correctness on the workloads where the CUTLASS FP8 path is fragile. See `operators/moe/optimization-ladder.md` "T-dependent GEMM2 backend".
- **Tile-shape dispatch by max per-expert M:** `max_M ≤ 256 → 64×128×128 tile`, `max_M > 256 → 128×128×128 tile`. Two CUTLASS template instantiations are JIT-compiled into the same `.so`. Worth +13% — the largest CUTLASS-internal lever in the campaign. See `code-examples/moe-dual-tile-dispatch.md`.

Also worth noting: small-T workloads (seq_len = 1, 7, 14, 15, 16) take a **degenerate fast path** — when `M ≤ 2` for any expert, dispatch to `gemv_fp8_blockscale_kernel` instead of going through CUTLASS grouped GEMM at all. Mirrors the "single-pass kernel for 1 token" pattern from DSA.

---

## Multi-heads-per-block (N-heads/block)

**What it does.** Each CTA processes multiple attention heads (or multiple independent tiles) instead of one. The shared KV tile is reused across all heads within the block.

**When it helps.** Any grouped/multi-query attention where KV is shared across heads — the data is already the same, so handling N heads per block amortizes the load cost.

**When it hurts.**
- Registers per thread balloon with N heads. Past 4–8 heads/block on many architectures, register pressure crashes occupancy.
- Block size grows, reducing blocks-per-SM.

**How.** Assign each warp (or warp pair) one head:

```cuda
// 128 threads, 4 warps, 4 heads/block, 32 threads per head
int head_in_block = threadIdx.x / 32;
int lane = threadIdx.x & 31;

float q_reg[HEAD_DIM_PER_LANE];    // this thread's slice of query for its head
float acc_reg[HEAD_DIM_PER_LANE];  // this thread's slice of accumulator

// KV tile is shared across all 4 heads
__shared__ bf16 k_tile[TILE_K][HEAD_DIM];

// Each head (warp) computes independently on the shared KV tile
...
```

**Field note.** In V5 of the DSA run, switching from 1-head/block to 4-heads/block was a +24% jump in a single commit (v32, 128x → 158x). 8-heads/block was tried (V4) and reverted — register pressure drove occupancy too low.

Rule of thumb: start with 1, try 2, 4. Don't go higher unless NCU shows headroom.

---

## Block and grid sizing

**What it does.** Standard tile-size tuning: how many threads per block, how many blocks per kernel launch.

**When it helps.** Every kernel. Block size is usually the single most impactful knob for a first-pass kernel.

**When it hurts.** Not really — but the defaults are almost never optimal.

**How.** Sweep:
- Threads per block: 64, 128, 192, 256, 384, 512 (rarely 1024)
- Blocks: calculate as `num_work_items / threads_per_block`

Typical sweet spots:
- Matmul: 128 or 256 threads, block shape matching tensor-core fragment sizes
- Reduction: 128 or 256 threads, 4–16 blocks per SM
- Attention: 128 or 192 threads (one warp per head for 4-heads/block)

Use `__launch_bounds__(threads, min_blocks_per_sm)` to control occupancy — see `occupancy.md`.

---

## Persistent kernels

**What it does.** Launch `num_SMs` blocks, each looping over multiple work items, instead of launching one block per item. Removes per-launch overhead, improves L2 reuse (since the same SMs keep running), and enables producer/consumer patterns across phases.

**When it helps.** Many-tile workloads (e.g., long reductions, attention over long sequences). When launch overhead is a measurable fraction of kernel time.

**When it hurts.**
- Trivially-parallel work with uniform cost — the scheduler is already good.
- Kernels with variable work per tile — one SM gets stuck on a heavy tile while others finish early.
- Adds complexity (work queue, atomic counters).

**How.**

```cuda
__global__ void persistent(int num_tiles, ...) {
    __shared__ int next_tile;
    while (true) {
        if (threadIdx.x == 0) next_tile = atomicAdd(&global_counter, 1);
        __syncthreads();
        if (next_tile >= num_tiles) return;

        process_tile(next_tile);
    }
}

// Launch
persistent<<<num_SMs, threads, smem, stream>>>(num_tiles, ...);
```

**Hardware notes.** Work best on SM80+ where launch overhead is meaningful relative to per-tile cost.

**Field note (DSA).** Not used in the DSA run — the outer token count was small enough that launch overhead wasn't dominant. For GEMM-style or long-seq attention, persistent kernels are often a key late-phase optimization.

**Field note (MoE).** The hand-written tcgen05 GEMM in the FuseMoE kernel is fully **persistent**: 148 CTAs (one per SM on B200), each loops `for tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x` with the warp-specialised 6-warp body (1 TMA producer + 1 MMA issuer + 4 epilogue drain). The persistent pattern is what makes the MMA-producer/consumer pipeline economical — without it, every tile pays kernel-launch overhead. See `code-examples/tcgen05-kernel-skeleton.md`.

---

## Warp-level shuffle reductions

**What it does.** Uses `__shfl_sync` / `__shfl_down_sync` to sum / max / min across a warp without going through shared memory. Faster than `__syncthreads()` + smem round trip.

**When it helps.** Any intra-warp reduction (32 threads collaborating on one number): softmax max, softmax sum, norm computation, dot-product reduction.

**When it hurts.** Multi-warp reductions (>32 values). For those, do warp-shuffle within each warp, one atomic per warp, or a second reduction stage via shared memory.

**How.**

```cuda
__device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;  // lane 0 has the sum
}

__device__ float warp_max(float v) {
    for (int offset = 16; offset > 0; offset /= 2) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}
```

Full warp-wide reduction (not just lane 0): replace `__shfl_down_sync` with `__shfl_xor_sync` — now every lane has the sum.

**Hardware notes.** `*_sync` variants are required on SM70+. The mask (`0xffffffff`) specifies participating lanes; usually all of them.

**Field note (MoE).** The MoE `routing_kernel` was the late-stage hotspot once the GEMM path was tuned. Three warp-shuffle rewrites (the "routing parallelisation" rounds in the campaign):
- **Top-K parallelisation (Round 6 of the late-stage campaign):** the per-token top-8-of-32 loop was rewritten as 8 warps × `K`-round `__shfl_xor_sync` argmax with winner-masking + warp-0 merge of the kept-group's sorted streams. Eliminated 6 of 8 `__syncthreads` per token. Bit-equivalent output.
- **Group-kept parallelisation (Round 6):** the serial `for k in kTopKGroup { find best ungathered group }` loop was replaced with a 4-round warp-shuffle argmax over 8 lanes.
- **Tail parallelisation (Round 5):** the lane-0 `sum + normalise + atomic` tail was replaced with a warp-0 parallel version using `__shfl_xor_sync` for the reduction.

Combined gains were **+2.85% then +3.38%** on the all-19 average, and the late shape-gated single-warp variant (Round 11) added **+5.9% / +6.7%** on the two long-seq workloads. An earlier and simpler instance — replacing a lane-0 serial scan of per-group scores with one-score-per-lane + `__shfl_sync` warp reduction — was worth **+1.4%** on its own (ladder item #13). See `operators/moe/optimization-ladder.md`.

---

## Programmatic Dependent Launch (PDL) and `griddepcontrol`

**What it does.** Lets a successor kernel's grid begin executing before its predecessor has fully finished, as long as each block reads `griddepcontrol.wait` before touching predecessor output and the predecessor calls `griddepcontrol.launch_dependents` once it's safe. Sibling of `programmaticStreamSerialization` (PSS): PSS is launched as a `cudaLaunchAttribute` and operates at the stream-graph level; the PTX intrinsics give you grid-level control inside a kernel.

**When it helps.** Pipelines of small-to-medium kernels chained on the same stream, where back-to-back launches leave visible idle gaps in the NCU timeline (host launch latency between kernels). Each chained pair saves a few µs of idle time; on a hot path with 7+ kernels, that adds up.

**When it hurts.**
- Cross-stream chains — PDL doesn't span streams. Use events.
- If the successor depends on host-side state, you still need a host sync. PDL is for purely-GPU dataflow.
- If the launch attribute or PTX intrinsic adds an unfavourable instruction-cache footprint to a very short kernel, the win disappears.

**How.**

```cuda
// Host side — successor launched with the PSS attribute
cudaLaunchConfig_t cfg = {};
cfg.gridDim = ...; cfg.blockDim = ...; cfg.stream = stream;
cudaLaunchAttribute attr[1] = {};
attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attr[0].val.programmaticStreamSerializationAllowed = true;
cfg.numAttrs = 1; cfg.attrs = attr;
cudaLaunchKernelEx(&cfg, successor_kernel, args...);

// Predecessor kernel — release waiting successors as early as safe
__global__ void predecessor(...) {
    // ... work ...
    // After this point, successor blocks may begin executing
    asm volatile("griddepcontrol.launch_dependents;");
    // ... finalisation work that doesn't affect successor inputs ...
}

// Successor kernel — first instruction in the grid
__global__ void successor(...) {
    asm volatile("griddepcontrol.wait;");
    // ... safe to read predecessor outputs ...
}
```

**Hardware notes.** SM90+ (Hopper) for `griddepcontrol`; PSS attribute available from CUDA 12. SM100 (B200) has both fully mature. Pair with TMA-based pipelines where the producer (TMA load) and consumer (MMA) can overlap most cleanly.

**Field note (MoE).** PDL chains the entire post-routing pipeline in the FuseMoE kernel: `routing → counting → gather → GEMM1 → SwiGLU+quantise → GEMM2 → pull_scatter`. Combined with the host-side "zero-sync fast path" (no `cudaStreamSynchronize` anywhere), this is what made the hot path effectively GPU-only. Worth ~+0.5% additional once PSS at the stream level was already saturated. See `code-examples/zero-sync-fast-path.md`.

---

## Fast-path / slow-path architectural split

**What it does.** Splits the kernel-orchestration logic into two paths:
- **Fast path** — runs on every invocation; uses only device-side metadata, no host round-trip, no `cudaMalloc`.
- **Slow path** — runs only on weight-change or shape-change events; does any host-side setup (CUTLASS argument validation, descriptor build, workspace alloc).

Most inference invocations hit the fast path. The slow path is rare but tolerates more synchronous work because it only fires on reconfiguration.

**When it helps.** Pipelines where most calls are repeats with the same shape/weights but a handful of bookkeeping operations have to be re-done. Without the split, those operations land on every invocation and put a floor on small-T latency.

**When it hurts.**
- For one-shot kernels or workloads with shape changing on every call, the split adds branches without saving work.
- Adds two-path complexity to the host code; debugging is harder.

**How.**

```cpp
void moe_forward(...) {
    bool weight_changed = check_weight_signature(weights);
    if (weight_changed) {
        // Slow path: one-time setup
        rebuild_cutlass_args_on_host();
        cudaMemsetAsync(workspace, 0, workspace_bytes);
        // ... can include synchronous work; rare event
    }
    // Fast path: every invocation
    launch_kernels_with_device_side_metadata(...);
    // Never blocks on host; metadata-update kernels run on the GPU
}
```

**Field note (MoE).** Enabling the zero-sync fast path in the FuseMoE kernel is in effect "make the fast path large enough to hold the entire forward". The slow path covers only weight-change events. Without the split there's no way to get expert-routing setup off the CPU's critical path. The +16.3% "zero-sync fast path" line in the ladder is enabled by this architectural decision.

---

## Right-sizing grid: launches that match actual work

**What it does.** For non-persistent kernels, launch exactly `ceil(work / block_size)` blocks — not `num_SMs` and not a constant. For persistent kernels, launch `num_SMs` once and use a tile-counter loop. Mixing them (launching `num_SMs * 8` for a non-persistent kernel) just wastes the empty blocks on scheduler bookkeeping.

**When it helps.** Variable-shape benchmarks where one launch config is wrong for some shape. Worth +1–2% on large workloads where over-launching used to leave half the CTAs doing zero work.

**When it hurts.** Pure persistent kernels — the whole point is to launch `num_SMs` and let blocks pull work. Don't confuse the two patterns.

**Field note (MoE).** Several MoE kernels (routing, gather, swiglu, pull_scatter) were initially launched with grid sizes that over-counted by 2–8× for small workloads. Tightening to `ceil(total_rows / block_size)` per kernel was a +1–2% catch on large workloads. The tcgen05 GEMM, by contrast, stays at `gridDim = 148` (one CTA per SM) and uses a tile-counter loop — that's the persistent pattern.

---

## Quick reference

| Scenario | Technique |
|---|---|
| Outer parallelism doesn't fill SMs | Split-K |
| Need different code for small vs large batches | Adaptive dispatch |
| KV shared across multi-query heads | Multi-heads-per-block |
| Kernel launches many tiny blocks | Persistent kernel |
| Inner reduction over ≤ 32 elements | Warp shuffle |
| First-pass kernel, default block size | Sweep 64–256 |
| Visible idle gaps between back-to-back kernels on one stream | PDL (`programmaticStreamSerialization` + `griddepcontrol`) |
| Most calls repeat shape, occasional reconfig | Fast-path / slow-path architectural split |
| Grid clearly larger than actual work | Right-size to `ceil(work / block_size)` |
