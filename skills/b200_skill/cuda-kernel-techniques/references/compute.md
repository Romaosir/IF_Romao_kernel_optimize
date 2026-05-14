# Compute techniques

Once you've loaded data well and placed it well, the next axis is doing the math efficiently. Tensor cores, specialized reductions, and instruction-level tuning live here.

---

## Tensor cores — WMMA, WGMMA, MMA

**What it does.** Dedicated matrix-multiply-accumulate hardware. One instruction performs a small-matrix MMA (e.g., 16×16×16 at fp16). On modern NVIDIA GPUs, tensor cores dominate FLOPS — on H100, they provide ~10× the throughput of the regular FP pipes for matmul.

**Variants.**
- **WMMA** (SM70+) — per-warp API (`nvcuda::wmma`), 16×16×16 fp16 fragments. Legacy.
- **MMA** (SM75+) — per-thread PTX `mma.sync` instructions, more shape flexibility.
- **WGMMA** (SM90+) — warp-group MMA, asynchronous, larger fragment shapes, TMA-friendly. Best choice on H100/B200.

**When it helps.** Any kernel with a GEMM-shaped inner loop — matmul, linear layers, attention Q@K^T and attn@V.

**When it hurts.**
- Scattered / gather-style access patterns — tensor cores want rectangular tiles, and fetching scattered data into fragments costs more than you save.
- Non-GEMM workloads (reductions, norms) — nothing to multiply.
- Small matrices where the fragment size is bigger than the problem.

**How (WMMA, the simplest API).**

```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// 16x16x16 fp16 GEMM fragment
fragment<matrix_a, 16, 16, 16, half, row_major> a;
fragment<matrix_b, 16, 16, 16, half, col_major> b;
fragment<accumulator, 16, 16, 16, float> c;

fill_fragment(c, 0.0f);
load_matrix_sync(a, a_ptr, lda);
load_matrix_sync(b, b_ptr, ldb);
mma_sync(c, a, b, c);
store_matrix_sync(c_ptr, c, ldc, mem_row_major);
```

For WGMMA on SM90+, use CUTLASS 3.x's Collective MMA — hand-rolling WGMMA is brittle.

**Field note (DSA).** WMMA was tried and reverted multiple times. The KV access pattern (gather via sparse indices) didn't fit rectangular fragments. For dense matmul and standard attention, tensor cores are usually table stakes.

**Field note (MoE).** UMMA — the SM100 5th-generation `tcgen05.mma` instruction with TMEM-backed accumulator — is the foundation of the FuseMoE FP8 path. The CUTLASS collective (`Sm100ArrayTmaUmmaWarpSpecializedBlockwiseScaling`) builds on it; the hand-written kernel uses `tcgen05.mma.cta_group::1.kind::f8f6f4` directly. Switching from CUTLASS-only to a hand-written tcgen05 for *both* GEMMs (with their own TMA descriptor pair each) was worth roughly **+6.8%** cumulatively and is the only path through the 14% CUTLASS occupancy ceiling. **Critical compile flag**: `-arch=compute_100a,code=sm_100a` (not `sm_100`); without the `_a` suffix, tcgen05 PTX is silently dropped to a slower codegen and you get correct-but-slow. See `code-examples/tcgen05-kernel-skeleton.md` and `operators/moe/optimization-ladder.md`.

---

## Online softmax

**What it does.** Computes softmax (and logsumexp) in a single streaming pass over the logits, maintaining running (max, sum_exp, output_accum). Avoids materializing the full attention matrix.

**When it helps.** Every attention kernel, always. This is the FlashAttention core trick.

**When it hurts.** Never, for attention. The alternative (two-pass softmax) wastes bandwidth by writing the attention matrix to global memory.

**How.**

```cuda
// Running state
float running_max = -INFINITY;
float running_sum = 0.0f;
float acc[HEAD_DIM] = {0};

for each KV tile {
    // Compute new logits for this tile
    float logits[TILE_K];
    for k in tile: logits[k] = dot(q_reg, k_smem[k]) * sm_scale_log2e;

    float tile_max = reduce_max(logits);
    float new_max  = fmaxf(running_max, tile_max);

    // Rescale accumulators and sum for the old max
    float correction = exp2f(running_max - new_max);
    running_sum *= correction;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) acc[d] *= correction;

    // Apply new tile's contribution
    for k in tile {
        float w = exp2f(logits[k] - new_max);
        running_sum += w;
        for (int d = 0; d < HEAD_DIM; d++) acc[d] += w * v_smem[k][d];
    }

    running_max = new_max;
}

// Final normalization
for (int d = 0; d < HEAD_DIM; d++) out[d] = acc[d] / running_sum;
float lse = log2f(running_sum) + running_max;  // natural log: + ln(2) factor elsewhere
```

**Numerical notes.** The rescaling of `acc` and `running_sum` when `new_max > running_max` is essential. Skipping it causes catastrophic underflow or overflow. Online softmax is not optional when numerical stability matters.

---

## Base-2 softmax via `exp2f`

**What it does.** Replaces `expf` with `exp2f` in the softmax kernel by pre-multiplying the scale factor by `log2(e) ≈ 1.4427`. The hardware SFU has a dedicated `exp2` instruction; `expf` is emulated on top of it with an extra conversion.

**When it helps.** Every softmax-like kernel. Free +3–10% speedup.

**When it hurts.** Only if your downstream consumer expects a specific precision and the SFU's `exp2` is less precise than your `expf` emulation (rare in practice).

**How.**

```cuda
const float log2e = 1.4426950408889634f;
float sm_scale_log2e = sm_scale * log2e;

// Apply log2-scaled sm_scale once
float logit = dot(q, k) * sm_scale_log2e;  // = original_logit * log2(e)

// Softmax uses exp2 instead of exp
float w = exp2f(logit - running_max);

// If returning lse, account for the base change
float lse_base_e = (log2f(running_sum) + running_max) * /* ln(2) */ 0.6931471805599453f;
// Or keep it in log2 domain if the API allows
```

**Field note.** V2 v5 of the DSA run introduced `exp2f` and got +8%. Once in, every subsequent kernel variant kept it.

---

## FMA and explicit intrinsics

**What it does.** `a*b + c` → `fma(a, b, c)` — one instruction, one rounding step, higher throughput than multiply-then-add.

**When it helps.** Hot inner loops with many multiply-accumulates. The compiler is usually smart enough to fuse them, but explicit intrinsics are insurance when you find it hasn't.

**When it hurts.** Obscures the intent of simple code. Don't use in non-hot paths.

**How.**

```cuda
// Equivalent, but sometimes the compiler produces separate mul + add
float x = a * b + c;
float y = __fmaf_rn(a, b, c);   // Force one instruction, round-to-nearest

// bf16 dot product intrinsics (SM80+)
float acc = __bf16x2_fmaf(__bf16x2_t{a0, a1}, __bf16x2_t{b0, b1}, acc);
```

**Hardware notes.** Always available. Not a revolution — a small, consistent speedup when applied in hot loops.

---

## Operator fusion

**What it does.** Combines multiple elementwise ops (bias add, activation, scaling, mask, residual) into a single kernel epilogue. Avoids round-tripping through global memory between ops.

**When it helps.** Whenever you have N sequential elementwise ops following a compute-heavy kernel (matmul + bias + GELU, attention + mask + dropout + residual, etc.). Easy 10–30% gain.

**When it hurts.**
- When the ops have genuinely different shapes or access patterns.
- When fusion adds too much register pressure and kills occupancy.

**How.** In the kernel's final write-back phase, before storing to global memory, apply all post-processing:

```cuda
// Raw write-back
out[row * N + col] = acc;

// Fused: bias + GELU + scale + write
float biased = acc + bias[col];
float gelu = 0.5f * biased * (1.0f + tanhf(0.7978845608f * (biased + 0.044715f * biased*biased*biased)));
out[row * N + col] = gelu * scale;
```

**Field note (MoE).** Fusion was a major class of wins in the FuseMoE late-stage campaign — five separately measured ones:
- **Dual-prep kernel** (one launch, writes argument arrays for *both* GEMM1 and GEMM2 in a single grid): **+2–3%**
- **Scan-into-scatter fusion** (prefix-scan fused into the scatter kernel, kernel count dropped from 11 → 9): **+2.8%**
- **Memset-into-pull_scatter tail** (the pull_scatter kernel zeros the next-iteration's counters on the way out, eliminating a `cudaMemsetAsync` launch): **+1.0%**
- **Metadata-into-scan fusion** (`threadfence` removal + metadata fusion in the GPU planner): **+5.8%**
- **SwiGLU + FP8 quantize fusion was *attempted and reverted*** — caused correctness failures on 5/19 workloads (the FP8 CUTLASS path is sensitive to input layout assumptions). The general rule "if fusion adds register pressure or breaks a downstream consumer's layout assumption, don't fuse" was the learning. See `operators/moe/fp8-correctness-modes.md` mode 1 for the full failure detail.

### Sub-pattern: fuse next-iteration setup into current-iteration tail

A specific fusion shape worth naming: when a per-iteration pipeline needs to *reset state* (zero a counter, clear a buffer) before the next iteration, do the reset in the **tail of the last kernel** of the current iteration rather than as a separate `cudaMemsetAsync` between iterations. The threads that finished early during the tail do the zeroing for free.

```cuda
__global__ void pull_scatter(..., int* next_iter_counts) {
    if (active_work) {
        do_pull_scatter(...);
    }
    // Tail: threads that have nothing more to do zero the next-iter buffer
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NEXT_ITER_COUNT_SIZE) {
        next_iter_counts[tid] = 0;
    }
}
```

Eliminates one launch per invocation. Generalizes to: scan-into-scatter (do the prefix scan and the scatter in one kernel), routing-into-counting, etc. Worth +0.5–3% per fusion depending on launch overhead share.

---

## `#pragma unroll`

**What it does.** Forces the compiler to fully or partially unroll a loop, enabling ILP by allowing independent iterations to be in flight simultaneously.

**When it helps.** Short loops (< 16 iters) with a constexpr trip count, inside hot inner loops, with independent per-iteration work.

**When it hurts.**
- Over-unrolling blows up the instruction cache (I-cache misses appear as "Imc Miss" in NCU).
- Trip count not constexpr → pragma is silently ignored.
- Per-iteration work has cross-iteration dependencies → unrolling doesn't help ILP, just bloats code.

**How.**

```cuda
#pragma unroll
for (int d = 0; d < HEAD_DIM; d++) {
    acc[d] += w * v_smem[k][d];
}

// Partial unroll when HEAD_DIM is large
#pragma unroll 4
for (int d = 0; d < HEAD_DIM; d++) { ... }
```

**Field note.** V4 v5 of the DSA run got +18% from a `#pragma unroll` on the output-accumulation loop. The inner loop was 512 iterations though, so `#pragma unroll 8` was more appropriate than full unroll.

---

## Quick reference

| Scenario | Technique |
|---|---|
| Matmul-shaped inner loop on SM90+ | WGMMA |
| Matmul-shaped inner loop on SM70–SM89 | WMMA |
| Attention softmax | Online softmax + base-2 via `exp2f` |
| Multiple elementwise ops after compute | Fuse into epilogue |
| Inner-loop ILP underfilled | `#pragma unroll 4` or `8` |
| Manual dot-product | FMA intrinsics |
