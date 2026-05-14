# Non-GEMM Optimizations

After the GEMM backend is sorted (CUTLASS FP8 or tcgen05), the remaining runtime is routing, gather, SwiGLU, quantize, and scatter. Individually small, but they add up — the items below collectively contributed several percent in production.

## Routing / top-k / counting

### Warp-parallel group scores via shuffle

Original: lane-0 computes per-group scores serially.
Optimized: every lane computes one score, then warp reduction via `__shfl_sync`.

**Gain: +1.4%**

### Keep routing intermediates in registers (remove `s[]` shared memory)

Observed that the routing kernel's per-lane `s[]` array can live in registers via shuffle patterns instead of `__shared__` memory.

**Gain: +1.0%**

### Remove redundant sync barriers in top-k

The top-k loop had one extra `__syncthreads()` per iteration that wasn't required — grid-level ordering already guaranteed the invariant.

**Gain: +0.3%**

### Active-expert sort on GPU

Covered under the zero-sync fast path (`code-examples/zero-sync-fast-path.md`). No host round-trip — sort active experts by count directly on the GPU, write sorted IDs to a device buffer.

### Parallelize pull_scatter expert lookup

8 threads collaborate on the per-row expert lookup in scatter, vs 1 thread previously.

**Gain: marginal, sub of `+2.1%` uint4 scatter item**

## Memset / zero-init

### Fuse next-iteration memset into pull_scatter tail

The counts buffer needs to be zeroed before the next MoE invocation. Instead of a separate `cudaMemsetAsync`, have the pull_scatter kernel write zeros during its own tail (threads that finish early).

**Gain: +1.0%**

### Combine multiple memsets

Small memsets of adjacent buffers (counts + row_offsets + active_eids) can be combined into one `cudaMemsetAsync` over the contiguous region.

**Gain: +0.1–0.3% (noise on small workloads, visible on small-T)**

## Gather / SwiGLU / quantize

### 16-byte aligned gather writes

The FP16 gather kernel writes results to a buffer that's consumed by the FP8 quantize kernel. Aligning the gather output to 16 bytes lets the quantize kernel do `uint4` reads.

**Gain: baseline-shift, feeds the +2.1% uint4 scatter**

### SwiGLU 8 rows per block

Default of 1 row/block wasted shared memory and had poor occupancy. 8 rows/block balances smem use and parallelism.

**Gain: +0.3–0.5%**

### GEMV fast-path for M ≤ 2

For the degenerate case where `M ≤ 2` (decode-style single-token workloads), the CUTLASS grouped GEMM path has too much overhead. A dedicated `gemv_fp8_blockscale_kernel` using 148-CTA grid and FP8 × BF16 GEMV pattern handles this case directly.

**Gain: not isolated in bulk metric, but shipped — makes small-T workloads viable**

### Avoid writing `row_scale_out` when unused

The SwiGLU→FP8 quantize kernel wrote a per-row scale output that was unused on the main CUTLASS path. Removing the write saved a small amount of memory bandwidth and one global-memory arrival.

**Gain: marginal**

## Launch and dispatch

### Right-size grid to actual work, not hardware SM count

When launching the main kernel, using `gridDim = 148` (full B200 SM count) makes sense for persistent kernels. For non-persistent kernels that do a fixed amount of work, tighten to `ceil(work / block_size)` — don't over-launch empty CTAs.

**Gain: +1–2% on large workloads**

### Remove `cudaGetLastError()` from hot path

`cudaGetLastError()` synchronizes with the driver. It belongs in debug builds and first-invocation validation, not in every launch.

**Gain: marginal; noise-level**

### PDL (Programmatic Dependent Launch) for fine-grained chaining

PDL is the sibling of PSS (`programmaticStreamSerialization`). Where PSS operates stream-level, PDL can be used at grid granularity via `griddepcontrol.wait` / `griddepcontrol.launch_dependents` PTX in the kernel body. Useful for chaining kernels that share partial state (e.g., gather → quantize).

**Gain: +0.5% on chains where PSS is already saturated**

See `code-examples/zero-sync-fast-path.md` for the PTX pattern.

### Skip cuBLAS allocation when CUTLASS path is active

If the runtime decides CUTLASS/tcgen05 will handle GEMM, don't pay the cost of cuBLAS handle creation or workspace allocation. Guard these behind a "fallback needed" flag.

**Gain: +0.5%**

## CUTLASS argument caching

### Cache `can_implement()` and `get_workspace_size()` results

Both are O(ms) on first call. Mark them `static bool s_validated = false` and compute once per (tile_variant, problem_shape) combination.

**Gain: marginal per-invocation but significant cold-start speedup**

### Dual (fused) prep kernel

Rather than two prep launches (one for GEMM1 args, one for GEMM2 args), fuse both into a single kernel that writes to both argument-array sets.

**Gain: +2–3%** (belongs on the main ladder but worth calling out here as the concrete non-GEMM mechanism)

## GPU planner fast-path vs. slow-path split

This is an architectural choice, not a micro-opt. The main kernel has two modes:

- **Fast path** — runs on every invocation; uses device-side metadata only, no host sync
- **Slow path** — runs only on weight-change events (new expert set, reconfiguration); does any host-side setup

This split lets the fast path be maximally lean — most kernels in inference are fast-path invocations. The slow path is rarely exercised, so can tolerate a synchronous step.

**Gain: enables the zero-sync fast path to exist at all — not a standalone number**

## Pipeline-swiglu (long-seq chunk pipelining)

For very long sequences (seq_len in the multi-thousand range), chunked pipelining of the SwiGLU stage overlaps GEMM1 output with GEMM2 input prep. Guarded by a `longseq_threshold` flag.

**Gain: shipped; variable by workload; not isolated in the bulk metric**

---

## How to approach this class of optimizations

1. They don't beat the ladder items — always finish items 1–8 of `optimization-ladder.md` first
2. Each is small individually (0.5–2%); cumulatively they add 5–8%
3. NCU is critical — most of these are only visible as small specific stalls (branch miss in routing, gather-bandwidth ceiling) not whole-kernel dominance
4. Verify each change with N ≥ 3 runs — they're near the noise floor
5. Bundle related micro-opts (all routing changes in one round, all gather changes in one round) so noise doesn't hide individual effects
