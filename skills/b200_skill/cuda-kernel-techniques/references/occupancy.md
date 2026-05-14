# Occupancy and resource management

Occupancy is the ratio of active warps to the hardware maximum. Low occupancy means the GPU has nothing to schedule while waiting on a stall. These techniques fit more warps in flight without changing what the kernel does.

---

## `__launch_bounds__`

**What it does.** Caps the per-thread register count that the compiler targets, in exchange for promising a block size and desired blocks-per-SM. The compiler may spill variables to local memory, but occupancy goes up.

**When it helps.**
- When per-thread registers pin you below your target occupancy (e.g., 72 regs/thread → 1 block/SM on an A100 when you could fit 2 at 54 regs/thread).
- Any time NCU "Occupancy Limiter" says "Registers."

**When it hurts.**
- The spilled variables go to local memory (L1-cached). In hot loops, spills can be slower than having lower occupancy.
- Guessing wrong on the blocks-per-SM number can make things worse.

**How.**

```cuda
__global__ void __launch_bounds__(128, 2) my_kernel(...) {
    // 128 threads per block, aiming for 2 blocks/SM → compiler targets ≤ registers/(128*2) per thread
    ...
}
```

**Tuning.** Don't guess — profile.
1. Start without `launch_bounds`, note registers/thread and achieved occupancy.
2. Add `launch_bounds(threads, 1)` as a baseline.
3. Sweep the second number: 2, 3, 4. Stop when register count forces spills (check ptxas output for "stack frame" > 0).
4. Keep whichever gives best speedup (not highest occupancy — the two don't always correlate).

**Hardware notes.** The `minBlocksPerMultiprocessor` must be achievable given smem per block and threads per block as well — not just registers. The compiler will honor your request only if feasible.

**Field note (DSA).** V5 of the DSA run used `__launch_bounds__(128, 2)` — two 128-thread blocks per SM. Earlier attempts at `__launch_bounds__(256, 2)` failed because 256 threads at ≤ 128 regs was too tight.

**Field note (MoE).** Several measured launch-bounds choices in the FuseMoE kernel — each one specifically tuned, not defaults:
- `tcgen05` GEMM kernel: `__launch_bounds__(192, 1)` — 6 warps × 32 threads, hard-pinned to 1 CTA/SM (that's the *point* — TMEM accumulator + 224 KB smem already cap occupancy, and the persistent warp-spec design needs exactly one block).
- `pull_scatter` tight kernel: `__launch_bounds__(256, 4)` — measured: `(256, 5)` and higher cause register spill to local memory; `(256, 3)` leaves throughput on the table.
- `gather_fp8_and_scales_tight_k`, `swiglu_to_fp8_tight_kernel`: `__launch_bounds__(256, 8)` — high occupancy ceiling, 32 registers/thread is enough.

These are operator-specific; if your kernel doesn't match the workload shape, sweep. The deeper observation: **measure with NCU after every launch-bounds change** — the compiler will silently *not honor* your request if smem/regs disagree, and you only notice when occupancy doesn't move. See `operators/moe/tuning-knobs.md`.

---

## Register pressure management

**What it does.** Proactively keeps the register footprint of a kernel below a target, so occupancy stays high.

**When it helps.** Any kernel where `--ptxas-options=-v` shows registers/thread > 72 (typical A100 sweet spot) or > 128 (H100/B200). Symptom: `nvcc` compile output showing "X registers" and "Y bytes stack frame."

**When it hurts.** If register reductions come at the cost of recomputation, you might net-negative. Measure.

**How.**

1. **Factor hot loops into `__device__` functions.** The compiler often allocates registers more conservatively inside called functions.
2. **Reduce variable live-ness.** Don't hold a tile's worth of data across multiple unrelated compute phases. Use it, write it back, reuse the registers.
3. **Use `__restrict__`** — enables the compiler to overlap loads (cuts the "waiting" registers).
4. **Shrink accumulator types** where correctness allows: `float acc` → `__half2 acc` halves the footprint.
5. **Avoid large constexpr arrays in registers.** A `float foo[128]` is 128 registers. Put constants in `__constant__` memory instead.

**Diagnostic workflow.**

```bash
# Compile with verbose ptxas — tells you per-function register count
nvcc -Xptxas -v -Xptxas -warn-spills kernel.cu

# Output to look for:
# ptxas info: Used 72 registers, 0 bytes stack frame, 0 bytes spill stores
```

If spill stores > 0, the compiler gave up and spilled. Either raise the `launch_bounds` cap (more regs, lower occupancy) or cut live-ness.

---

## Shared memory budget

**What it does.** Caps per-block shared memory so the target blocks-per-SM is achievable. Mirror of register pressure, different knob.

**How.**

```cuda
// Check smem per block
// Per-block smem ≤ smem_per_SM / target_blocks_per_SM
// e.g., on A100 with 164 KB smem/SM, target 2 blocks/SM → cap at 82 KB/block

// Use dynamic smem to make the budget explicit
extern __shared__ __align__(16) char smem_buffer[];
float* tile = reinterpret_cast<float*>(smem_buffer);

// At launch:
int smem_bytes = TILE_M * TILE_K * sizeof(bf16) * 2;  // double-buffer
my_kernel<<<grid, block, smem_bytes, stream>>>(...);
```

**For > 48 KB / block**, you must opt in:

```cuda
cudaFuncSetAttribute(my_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 228*1024);
```

---

## Cached function attributes

**What it does.** Avoids the per-launch overhead of `cudaFuncSetAttribute` by caching whether it's been set.

**When it helps.** Any kernel called frequently (hundreds of times per second) with large smem opt-in. The attribute-setting call takes microseconds — negligible in bulk but measurable when the kernel itself is sub-millisecond.

**How.**

```cuda
void launch_kernel(...) {
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(my_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        attr_set = true;
    }
    my_kernel<<<grid, block, smem_bytes, stream>>>(...);
}
```

**Field note (DSA).** V2 v22 of the DSA run added this pattern (along with static workspace) for a combined +5%.

**Field note (MoE).** Both required and load-bearing: the tcgen05 GEMM uses **224 KB dynamic shared memory** (well above the 48 KB default), so without `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 224*1024)` the launch silently fails or runs at the 48 KB cap. Cache the flag in a `static bool` — fresh `cudaFuncSetAttribute` per launch adds measurable overhead on small-T workloads where the kernel iteration is already ~30 µs.

### Sub-pattern: cache expensive CUTLASS validation queries

`cutlass::Status can_implement()` and `Gemm::get_workspace_size()` are O(ms) on the first call — non-trivial template instantiation work happens inside. Without caching, these run on every kernel invocation.

```cpp
static bool s_validated = false;
static size_t s_ws_bytes = 0;
if (!s_validated) {
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;
    s_ws_bytes = Gemm::get_workspace_size(args);
    s_validated = true;
}
gemm.initialize(args, workspace, stream);
gemm.run(stream);
```

**Caveat.** The cache is valid only as long as the kernel template, tile shape, and *shape regime* don't change. If you switch tile variants at runtime (e.g. the "adaptive dispatch" pattern in `parallelism.md`), each variant gets its own static flag. Marginal per-launch gain (~µs) but matters in cold-start scenarios and for kernels iterated ≥ 100/sec.

---

## Static workspace (avoid `cudaMalloc` per launch)

**What it does.** Pre-allocates scratch buffers (partial-output arrays for Split-K, etc.) once and reuses across launches.

**When it helps.** Any kernel that needs workspace proportional to input size. `cudaMalloc` per launch costs milliseconds — a lot compared to the kernel itself.

**How.**

```cuda
static float* s_partial_out = nullptr;
static int s_max_tokens = 0;

void ensure_workspace(int num_tokens, cudaStream_t stream) {
    size_t needed = num_tokens * SPLIT_K * HEAD_DIM * sizeof(float);
    if (num_tokens > s_max_tokens) {
        if (s_partial_out) cudaFreeAsync(s_partial_out, stream);
        cudaMallocAsync(&s_partial_out, needed, stream);
        s_max_tokens = num_tokens;
    }
}

void launch(int num_tokens, cudaStream_t stream) {
    ensure_workspace(num_tokens, stream);
    kernel<<<grid, block, smem, stream>>>(s_partial_out, ...);
}
```

Use `cudaMallocAsync` / `cudaFreeAsync` so it integrates with the stream's ordering.

**When it hurts.** Multi-stream concurrency — a static workspace is shared across streams and needs external synchronization if you run kernels in parallel. If you need per-stream workspaces, pool them.

**Field note.** V2 v15 of the DSA run added static workspace and saved 1–2 ms per call, worth +5%.

---

## `__ldg` / read-only cache hints (SM35+, meaningful on all modern arch)

**What it does.** `__ldg(ptr)` tells the compiler to route the load through the
read-only data cache (same physical cache as texture on modern arch). On
memory-bound streaming kernels, this hint often produces a measurable speedup
(10–30% on B200 small-batch cases we've observed) even though the compiler
sometimes already does this automatically.

**When it helps.**
- Streaming reads of large inputs where the same data won't be written back.
- Small-batch / few-row kernels that don't saturate HBM — read-only cache reduces
  DRAM pressure and the hardware schedules loads more aggressively.
- When you've already vectorized loads (`uint4` / `float4`) and want another
  bandwidth step.

**When it hurts.**
- Rarely hurts. The compiler can inline `__ldg` into normal `ld.global.nc` PTX
  and ignore it if it won't help.
- If `__restrict__` is already present and the compiler has already chosen the
  read-only path, `__ldg` is a no-op.

**How.**

```cuda
// Scalar
float v = __ldg(&x[idx]);

// Vectorized (treat pointer as uint4 or float4)
uint4 packed = __ldg(reinterpret_cast<const uint4*>(&x[row * D + lane * 8]));
float4 v4    = __ldg(reinterpret_cast<const float4*>(&x[row * D + lane * 4]));

// Typed helpers exist for __half, __nv_bfloat16 — see NVIDIA CUDA math API
```

Pair with `__restrict__` on the pointer type for maximum effect — the two
reinforce each other.

**Field note.** In the iter-2 softmax small-batch run, adding `__ldg` on the X
loads gave +25%, the single largest non-structural improvement. It should be a
default for any bandwidth-bound streaming read.

**When to reach for it.** After you've vectorized loads. Before you try
`cp.async` or TMA (those are for larger/structured copies; `__ldg` is for
scattered-but-streaming reads inside the inner loop).

---

## Block size sweet spots

Empirical defaults by kernel type (start here, then sweep ±1 power of 2):

| Kernel type | Typical block size |
|---|---|
| Matmul (WMMA / WGMMA) | 128 or 256 threads |
| Attention (1 head/block) | 128–256 threads |
| Attention (4 heads/block) | 128 threads (1 warp/head) |
| Reduction | 128 or 256 threads |
| Element-wise / trivial | 256 threads |
| Pointer-heavy (gather / scatter) | 128 threads |

Why these: they balance register + smem budget against minimum-warps-to-hide-latency. 1024 is almost always wrong — register pressure makes occupancy drop to 1 block/SM. 32 is almost always too small — no ILP within the block.

**Field note (MoE) — batch multiple rows per block when a single row underuses the block.** The FuseMoE SwiGLU+quantise kernel was initially launched as 1 row per block × 256 threads. Each row's compute saturates only a fraction of the block's smem and register budget, leaving the rest idle. Re-grouping to **8 rows per block** (still 256 threads, each warp handles one row) improved smem reuse and occupancy by **+0.3–0.5%** (ladder item #20). The generic rule: if an elementwise / per-row kernel has block-level resources sitting idle, batching N rows per block (N chosen so all resources are productive) is a cheap multiplier. Check NCU "Achieved Active Warps Per SM" before and after.

---

## Quick reference

| Symptom | Technique |
|---|---|
| NCU says "Occupancy limited by Registers" | `__launch_bounds__` |
| Register spills in ptxas | Reduce live-ness, `__restrict__`, smaller accumulators |
| Large smem > 48 KB | `cudaFuncSetAttribute` once, cache flag |
| Kernel launched many times with workspace | Static / pooled workspace |
| Not sure what block size to use | Sweep around table above |
