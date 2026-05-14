# Anti-patterns

Things that *sound* clever but usually regress performance or introduce bugs. Most of these are patterns that beginners try on intuition and that experienced CUDA engineers have learned to avoid. Check this list when an experiment regresses unexpectedly — the cause is often here.

---

## Memory access

### Using non-coalesced loads for simplicity

"I'll just read element by element" → every warp pays 8–32× the bandwidth of a coalesced read. Always design the thread-to-data mapping for stride-1 access across the warp.

### Manually unrolling what the compiler already unrolls

`#pragma unroll` is not the same as copy-pasting the loop body 8 times. The compiler handles it correctly; manual unrolling bloats code, evicts the I-cache, and can actually *reduce* performance. Use the pragma.

### Forgetting `__restrict__`

Without it, the compiler assumes every pointer may alias every other pointer, blocking load reordering and CSE. Almost free win. Add it on every input pointer.

### Misaligned vectorized loads

`*reinterpret_cast<float4*>(ptr)` when `ptr` is 8-byte aligned but not 16-byte aligned → hardware does two loads or the compiler inserts a slow path. Always check alignment. Use `__align__(16)` on smem tiles and validate input tensor alignment.

### Using TMA on scattered data

TMA wants rectangular tiles. Gather-style access (sparse indexing) doesn't fit. If your access pattern is "for each query, read 2048 random KV indices," TMA adds descriptor setup cost without giving you the bandwidth win. Use `cp.async`.

---

## Shared memory

### Shared memory bank conflicts

32 banks, 4 bytes each (mostly). A `[32][32]` tile is a classic conflict trap — every row starts at the same bank offset. Add padding (`[32][33]`) or use XOR swizzle.

### Over-using smem as scratch for data that fits in registers

A query vector of 32 floats doesn't need smem — it fits in registers. Using smem adds a round-trip, a sync, and potential bank conflicts. Registers are faster; use them.

### Forgetting `cudaFuncSetAttribute` for > 48 KB smem

The kernel will launch with the default (48 KB) cap and either crash or silently corrupt data. You must opt in via `cudaFuncSetAttribute`. Then cache the flag to avoid per-launch overhead.

---

## Compute

### Using WMMA / tensor cores on scattered access patterns

Tensor-core fragments are rectangular. Sparse attention with gather-based KV doesn't fit, because loading the fragment requires either (a) densifying scattered data through smem (costs more than the TC win) or (b) multiple small fragments (each has fixed overhead).

### Premature use of `atomicAdd`

Every atomic serializes contending threads. Only use for inter-block reduction in Split-K or similar. Within a block, use warp shuffles or smem reductions — they're 10–100× faster.

### Over-unrolling

`#pragma unroll` on a 512-iteration loop tries to emit 512× the code. I-cache thrash, register explosion. Use `#pragma unroll N` with a reasonable N (4, 8, 16) for long loops.

### Integer `/` and `%` on non-constant divisors

These compile to expensive multi-cycle sequences. If the divisor is a compile-time constant, use bit shifts (`>> n`, `& (N-1)`). If it's a runtime value that's usually a power of 2, branch or use `__brevll`.

### Forgetting that `__syncthreads` is per-block

Every `__syncthreads()` is a full block barrier. Between independent groups of warps, you don't need it. Use `__syncwarp` (intra-warp) or atomic-based coordination for producer-consumer patterns.

### Redundant `__syncthreads` whose invariant a higher-level guarantee already provides

A `__syncthreads()` inside an inner loop where the producer/consumer relationship is already enforced *outside* the loop (by grid-level ordering, or because consecutive iterations are independent, or because the data is per-thread). Audit each barrier and ask: *what would break if this sync weren't here?* If nothing, drop it. Observed at **+0.3%** in the FuseMoE routing top-k loop (`operators/moe/optimization-ladder.md` item #19) — one barrier per top-k iteration that grid ordering already enforced.

---

## Control flow

### Warp divergence in inner loops

`if (threadIdx.x % 2 == 0)` splits every warp evenly, serializing both halves. Any condition aligned to less than a warp boundary costs double. Hoist to `warp_id` or `blockIdx` granularity, or use predication / masking instead.

### Using `if` for bounds / validity checks when masking would work

Sparse attention's "if idx >= 0" check is divergence-inducing. Replace with a `-INFINITY` sentinel (after exp it's zero). Predication wins.

---

## Occupancy

### Very large block sizes (512+)

Unless the kernel is genuinely data-parallel with low per-thread state, large blocks force fewer blocks/SM. Register pressure usually forces you to 1 block/SM. Almost always worse than 2 × 128 or 2 × 192.

### Guessing `__launch_bounds__` numbers

`__launch_bounds__(256, 2)` without profiling means you made up a number. Compile with `-Xptxas -v`, look at register count, then compute what blocks-per-SM is feasible. Sweep 1–3 and pick the winner.

### Unnecessary register holding

Keeping a full tile of data in registers for a whole iteration when only half is actively used halves occupancy for no benefit. Recompute or reload; registers are precious.

---

## Algorithmic

### Two-pass softmax when streaming (online) softmax works

Two-pass writes the attention matrix to global memory, reads it back, normalizes. Streaming doesn't. For any attention-like kernel, streaming is strictly better — bandwidth saved, no intermediate tensor, stays in registers.

### Copying the cost of a wider intermediate type

`float acc = 0; acc += bf16(x) * bf16(y);` — the multiply happens in `float`, which is correct. But `bf16 acc = 0; acc += bf16(x) * bf16(y);` — the add happens in bf16, losing precision. fp32 accumulator is free; use it.

### Re-allocating workspace every kernel call

`cudaMalloc` / `cudaFree` per call costs milliseconds. Cache the workspace, grow it when needed, free when done.

### Adding L2 persistence API without profiling

`cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, ...)` has a per-call cost. For frequently-launched kernels, the overhead eats the cache-residence gain. Only enable after measuring that L2 thrashing is actually hurting.

---

## Process and discipline

### Bundling two changes in one experiment

"I changed block size AND added `__launch_bounds__` — it's 12% faster!" Which change caused it? You don't know. Now you can't tell whether to keep both, one, or either. Always one change per commit.

### Keeping a change because the geo-mean improved but one workload regressed

The average hides the pathology. Always check per-workload deltas. A 2% geo-mean improvement with a 40% regression on one workload is usually a REVERT unless the regressed workload isn't one you care about.

### Editing old perf log entries

The perf log is append-only history. Edit nothing. If an old entry was wrong, write a correction in the *next* entry. Editing history breaks everyone's mental model.

### Relying on a single eval run

GPU timing has noise. 1–3% run-to-run variance is normal. A single run at +1.5% is noise; a 5-run mean at +1.5% with 0.5% stddev is real.

### Skipping profile analysis after a KEEP

Without roofline signal, you're guessing what to try next. Profile every non-trivial KEEP (change of > 5%). Five minutes of NCU analysis beats five experiments of wrong-direction guessing.

### Not writing the hypothesis before running

When you don't commit to a hypothesis, post-hoc you'll rationalize any result. Writing "I expect +10%" before the run calibrates your intuition and surfaces disagreement with reality.

### Ignoring the 3-revert online-search rule

Three consecutive reverts means your mental model of the bottleneck is wrong. More local perturbations won't help. Web-search for fresh ideas, or you will thrash.

---

## Tool and framework

### Printf in production kernels

`printf` serializes across all threads and blocks it's called from. Fine for debugging a handful of launches; catastrophic in hot paths. Remove before committing.

### `cudaDeviceSynchronize()` in the middle of a pipeline

Synchronizes the whole device. Breaks concurrent streams. Use stream-specific sync (`cudaStreamSynchronize`) or event-based sync.

### Compiling for the wrong architecture

`-arch=sm_80` on an H100 works (Hopper runs Ampere PTX), but you miss all Hopper features (TMA, WGMMA, etc.). `-arch=sm_90` on an A100 fails to load. Match `-arch` to the target exactly, or use `-arch=sm_XX -gencode=arch=compute_YY,code=sm_YY` for a fat binary.

### Using `compute-sanitizer` only when things break

Run it occasionally on a clean run to catch latent races and OOB reads that haven't manifested yet. Much cheaper to fix now than after they corrupt results days later.

### Paying JIT compile cost on the hot path

`nvcc` → `.so` → `dlopen` is the standard pattern for vendoring CUTLASS (or any other compiled-on-demand kernel) into a project, and on a *cold* invocation the compile is tens of seconds. If the first benchmark run pays that cost it pollutes your speedup number, and even on warm runs the dynamic-load cache adds noise. Fix: inline the source as a string literal in your wrapper `.cu`, run the compile *once* at module init (not at the first kernel invocation), `dlopen` the resulting `.so` exactly once, and cache the function pointers in `static` globals. Measured at **+7%** when applied to the FuseMoE CUTLASS path on B200. Generalises to any framework that JIT-compiles kernels at runtime (Triton autotune, custom inductor backends, etc.) — get the compile *off* the hot path.

### Calling host-side framework sync inside a CUDA pipeline

After you've eliminated `cudaStreamSynchronize` from the hot path (zero-sync fast path), any *one* `torch.cuda.synchronize()`, `cuStreamWaitEvent` to a host event, mapped-memory CPU spin-wait, or even a Python tensor `.item()` call defeats the whole chain. The bench number reverts to the pre-zero-sync state and you waste rounds wondering why. Audit the wrapper code for *any* host-visible sync once you've moved metadata to the GPU — frequently these hide in `binding.py` argument processing or in a debugging `.cpu()` call left in by mistake. Same applies to mixing PyTorch tensor ops inside the CUDA pipeline: every cross-back into the framework adds a hidden synchronization barrier. Commit to either "pure CUDA" or "framework-tied" for the hot path; hybrids stall.

### Compiling tcgen05 / UMMA without the `_a` suffix

`-arch=compute_100,code=sm_100` accepts code that contains `tcgen05.mma` etc. but silently lowers them to non-tensor-core codegen. You get a correct-but-slow kernel. The arch *must* be `compute_100a,code=sm_100a` (note the `a`) for tcgen05 PTX to actually execute on tensor cores. Verify with `cuobjdump --dump-sass | grep TCGEN05` after a build.

### `cudaGetLastError()` in the launch hot path

`cudaGetLastError()` performs an implicit synchronization with the CUDA driver. Fine in debug builds and at module init; in a hot-loop launcher it adds measurable per-launch latency (a few µs) and can serialize otherwise concurrent streams. Keep it for the first invocation of a kernel (to catch invalid launch configs) and remove from the steady state.

### Writing outputs that no consumer reads

A kernel computes a quantity and writes it to global memory because "we might need it." If no consumer actually reads that buffer, you've paid the bandwidth and the atomic/store latency for nothing. Audit your output buffers; remove writes whose downstream is dead code. The win is in DRAM bandwidth, not in compute, but on memory-bound kernels it's free.

### Over-launching empty CTAs

Launching `gridDim = num_SMs * K` for non-persistent kernels when actual work is `ceil(N / blockSize)` blocks. The empty CTAs still cost scheduler bookkeeping. Right-size the grid to actual work, or commit to the persistent-kernel pattern (one launch per SM, work-pulling loop). Mixing the two patterns is the trap — see `parallelism.md` "Right-sizing grid".

### Allocating a `cuBLAS` handle on the hot path when it isn't used

When your dispatch logic picks CUTLASS or hand-written tcgen05 for the GEMM, the cuBLAS handle and workspace allocation are pure overhead. Guard cuBLAS init behind a "fallback needed" flag — only allocate if the path is actually taken. Goes for any backend init that's optional on the hot path.

### Trusting that the "right" kernel was dispatched without verifying

A particularly insidious failure mode on multi-backend kernel collections: a CUTLASS dispatch silently fails (returns non-success) and the caller falls back to a slower path. The bench number looks plausible (maybe 5–10% slower than expected) but you waste days "optimising" the wrong kernel. **After every change that affects backend selection, verify the NCU kernel name matches what was supposed to dispatch.** Bake a check into the Profiler agent if you're running multi-agent — see `cuda-agent-team`.

---

## Operator-specific dead-ends

This catalogue lists patterns that *generally* regress. Operator-specific measured regressions (e.g. "tried 2-SM cluster on FuseMoE long-seq, −18%") live under `operators/<operator>/dead-ends.md` so they don't pollute the generic catalogue. Check those before retrying anything you saw in a sibling-operator's run.
