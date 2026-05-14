# Numerical techniques

Lower precision saves bandwidth, compute, and storage. But lower precision also exposes catastrophic cancellation, underflow, and overflow bugs. These techniques describe how to use low precision without breaking correctness.

---

## Accumulator precision

**What it does.** Uses `float` (fp32) for running accumulators even when inputs and outputs are bf16/fp16/fp8.

**When to do it.** Always, unless the operator specification explicitly prohibits it. The cost of fp32 acc is ~0 (a handful of registers); the benefit is that softmax sums, dot-product reductions, and norm computations stay numerically stable.

**How.**

```cuda
// Inputs bf16, accumulator fp32
bf16 q_reg[V];  // load as bf16
float acc = 0.0f;
for (int i = 0; i < N; i++) {
    acc += float(q_reg[i]) * float(k_tile[i]);  // promote to fp32 for add
}
// Cast back only at final write-back
out[idx] = bf16(acc);
```

**Anti-pattern.** Accumulating in bf16 — you lose ~3 bits of precision per add. For long reductions (large K dimension), the error compounds to the tolerance floor quickly.

---

## Online softmax rescaling (re-stating for emphasis)

**What it does.** When the running max changes during a streaming softmax pass, rescales previously accumulated sum_exp and output_accum.

**Why it's numerical, not just algorithmic.** Without rescaling, you get a blend of two different exponent bases — a specific failure mode that produces wrong sums without crashing, usually showing up as gradients that are "almost" right but off by a constant.

See the `compute.md` entry for the code — this bullet is here to make sure you treat it as a correctness invariant, not an optimization you could skip.

---

## LSE in log2 base

**What it does.** Stores logsumexp in log2 domain instead of natural log. Matches the base of `exp2f`-based softmax and avoids repeated ln(2) multiplications in downstream consumers.

**When it helps.** When the kernel is the first stage of a chain (partial attention) and downstream (merge kernel, gradient kernel) also wants LSE for further computation.

**When it hurts.** If the downstream consumer expects natural-log LSE, you need to convert — and if only one of the producers converts, you get silent wrong results.

**How.**

```cuda
// Inside online softmax:
float lse_log2 = log2f(running_sum) + running_max;  // max already in log2 base

// When exposed to PyTorch / other APIs expecting natural log:
float lse_natural = lse_log2 * 0.6931471805599453f;  // ln(2)
```

Document in the kernel header which base the LSE output uses.

---

## bf16 partial output

**What it does.** Split-K partial output tensor uses bf16 instead of fp32, halving the memory bandwidth for the merge kernel to read.

**When it helps.** Split-K kernels with large SPLIT_K. For SPLIT_K=32, fp32 partial = 64 KB/token (HEAD_DIM=512); bf16 partial = 32 KB/token. The merge kernel becomes bandwidth-bound less quickly.

**When it hurts.**
- Each partial accumulator now has ~3 bits less precision. For long K with many Split-K chunks, the error compounds.
- If any intermediate step requires fp32 precision (e.g., a per-workload correction factor), the conversion cost eats the gain.

**How.**

```cuda
// Partial kernel writes bf16
// Accumulate in fp32, cast once at write-back
bf16 partial_out[HEAD_DIM];
for (int d = 0; d < HEAD_DIM; d++) partial_out[d] = bf16(acc[d]);
partial_gmem[split * num_tokens * HEAD_DIM + token * HEAD_DIM] = partial_out;

// Merge kernel reads bf16, promotes to fp32
float running_max = -INFINITY, running_sum = 0.0f, merged[HEAD_DIM] = {0};
for (int k = 0; k < SPLIT_K; k++) {
    float max_k = partial_lse[k];  // keep lse fp32
    float correction = exp2f(max_k - max(max_k, running_max));
    // ... standard online merge
    for (int d = 0; d < HEAD_DIM; d++) merged[d] += weight * float(partial_out[k][d]);
}
```

**Field note.** V3 of the DSA run added bf16 partial output for +10%. V4 tried it, found it regressed accuracy below tolerance, reverted. The viability depends on the tolerance window and the number of Split-K chunks. Always re-validate correctness after switching.

---

## fp8 inputs and tensor cores

**What it does.** Uses fp8 (E4M3 or E5M2) for input tensors, leveraging the fp8 matmul throughput on SM89+ / SM90+.

**When it helps.** Compute-bound kernels on SM89+ where fp8 throughput is 2× fp16 throughput. Common for inference of large models trained with fp8-aware quantization.

**When it hurts.**
- Requires a quantization scheme (per-tensor, per-row, per-group scales) — adds complexity.
- Not all models tolerate fp8 — check the downstream eval before committing.
- On SM80 and older, fp8 is emulated or not supported.

**How.** Use the CUTLASS fp8 collective or hand-roll via `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` PTX. This is one place where CUTLASS saves you weeks.

**Field note (MoE).** FP8 is the *entire* compute story for the FuseMoE kernel — both GEMMs run E4M3 inputs with FP32 blockwise (128-block) scales. Lessons that the campaign documented as a separate failure-mode catalog (`operators/moe/fp8-correctness-modes.md`):

| Mode | Triggers `INCORRECT_NUMERICAL` |
|---|---|
| Fused SwiGLU + FP8 quantise in one kernel | The fused output layout doesn't match the downstream CUTLASS scale-alignment assumption |
| Modifying any "named" optimisation on the FP8 CUTLASS path | The path's tolerance margin is narrow; tile/epilogue tweaks frequently regress 5 specific workloads |
| `cublasGemmStridedBatchedEx` with FP8 | Tensor-core dispatch works but a stale-padding bug surfaces on 6/19 workloads |
| FP8 requantisation 128-block → per-tensor | ~12.5% per-element error, far exceeds tolerance — fundamentally incompatible |
| cuBLASLt `BLK128x128_32F` | `CUBLAS_STATUS_NOT_SUPPORTED` on B200 |
| `cublasGemmGroupedBatchedEx` with FP8 | Runtime crash on most workloads |
| CuTe DSL alignment + CUDA graph | All-zero output (DSL + graph-capture interaction) |

The general rule the campaign extracted: **B200's native FP8 wants MXFP8 (32-element block) scaling, not the 128-block float32 scaling that pretrained models use.** Bridging the format gap is what makes the `KernelPtrArrayTmaWarpSpecializedBlockwiseScaling` collective load-bearing — and what makes every "small" change on that path correctness-fragile.

---

## Kahan / compensated summation

**What it does.** Corrects for lost precision in fp32 reductions by tracking a compensation term.

**When it helps.** Very long reductions (millions of elements) where even fp32 acc accumulates rounding error.

**When it hurts.** Always has a small overhead. Don't use unless you've verified the shorter reduction is actually losing precision.

**How.**

```cuda
float sum = 0.0f;
float c = 0.0f;  // compensation
for (int i = 0; i < N; i++) {
    float y = values[i] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}
```

Rarely needed in practice for ML kernels — most reductions are bounded enough that fp32 is fine.

---

## Correctness testing during optimization

**What it does.** Guards against numerical drift during optimization experiments.

**Why.** The keep-revert rule says all workloads must be CORRECT. But CORRECT usually means "within tolerance of reference." You can pass CORRECT while drifting toward the tolerance ceiling — a silent accuracy regression.

**How.**

1. Log absolute error (not just pass/fail) for every workload on every KEEP run.
2. Plot error over versions. If it's creeping up toward the tolerance, investigate the cause.
3. Before switching a tensor to lower precision (bf16 partial, fp8), establish a baseline of current error, then re-measure after the switch. If error jumped by >50% even within tolerance, the next regression will fail.

Example of absolute-error tracking in a perf log:

```markdown
## v25 — bf16 partial output

- Status: KEEP
- Speedup: avg=104.27x (+14%)
- **Max abs error: 0.0028 (prev v22: 0.0017)** — within tolerance 0.01, but trended up. Watch closely.
```

---

## Quick reference

| Scenario | Technique |
|---|---|
| Long reductions | fp32 accumulator |
| Attention softmax | Online with rescaling (not two-pass) |
| Split-K partial tensor, bandwidth-bound merge | bf16 partial — re-validate correctness |
| SM89+ inference, compute-bound matmul | fp8 via CUTLASS |
| Very long reductions (>1M) with error concerns | Kahan summation |
| Switching to lower precision | Log absolute error before/after |
