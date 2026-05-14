# FP8 Correctness Failure Modes

Seven distinct FP8 numerical-correctness failure modes were observed in production B200 MoE work. This is the single highest-value debugging reference — most new FP8 bugs match one of these patterns.

## Shared root cause

B200 SM100 native FP8 instructions require **MXFP8** (32-element block scaling with `e8m0fnu` scales). Most modern pretrained weights use **128-block float32** scaling (Hopper-style). **No B200 library supports the 128-block format natively.**

Every mode below is a consequence of this mismatch, a downstream API limitation, or a fusion that accidentally depends on the mismatch being hidden.

---

## Mode 1: Fused SwiGLU + FP8 quantize

### Symptom
A fixed subset of workloads fails with large numerical-correctness errors; the rest pass cleanly. The same subset fails on every run.

### Affected workloads
Those routed through the FP8 CUTLASS path (typically gated by a token-count threshold). Workloads below that threshold use cuBLAS FP16 and don't fail.

### What triggers it
Fusing SwiGLU activation with the subsequent FP8 quantize kernel. The fused kernel writes FP8 outputs and their scale factors in a layout that CUTLASS FP8 grouped GEMM cannot consume correctly.

### Debugging recipe
1. Disable the fusion; run SwiGLU and FP8 quantize as separate kernels
2. Same subset of workloads now passes → confirms mode 1
3. If failures persist, check mode 2 (could also be a CUTLASS path issue)

### Workaround
- Keep SwiGLU as its own kernel writing BF16
- Run FP8 quantize as a separate kernel on the BF16 output
- Fuse the FP8 quantize with the **next upstream** step if desired (e.g., with GEMM1's epilogue), but not with SwiGLU

---

## Mode 2: Any modification to the FP8 CUTLASS path

### Symptom
The same workload subset as mode 1 fails — even when your change seems unrelated to correctness. E.g., tweaking K-tile from 128 to 64, modifying the prep kernel, adjusting scheduler parameters.

### Why this is a separate mode
The FP8 CUTLASS path has a narrow operating envelope. Many "mechanical" changes break it subtly. The first instinct is "my change broke correctness" — but often it's that the path's tolerance is already at the edge, and your change pushed it over.

### Debugging recipe
1. Revert the change; workloads pass
2. Re-apply just the tile change; workloads fail
3. Check NCU — is the intended collective still being dispatched? (Silent fallback is common.)
4. If dispatch is correct and the change is correct, the path itself is fragile

### Workaround
**T-dependent dispatch** — for workloads where CUTLASS FP8 is known to fail, route through cuBLAS FP16 instead:

```cpp
if (t > 2000) {
    // cuBLAS FP16 path — not as fast, but correct
    call_cublas_gemm(...);
} else {
    // CUTLASS FP8 path — fast, known-good for this T range
    call_cutlass_fp8(...);
}
```

See [`code-examples/t-dependent-dispatch.md`](code-examples/t-dependent-dispatch.md) for the full dispatch code.

---

## Mode 3: `cublasGemmStridedBatchedEx` stale padding

### Symptom
Several workloads fail with very large absolute errors (observed ~28,000–33,000). Correctness is stable on isolated per-workload runs but breaks when workloads are run in sequence.

### What triggers it
Grow-only buffer reuse. The kernel allocates buffer space sized for the largest expert seen so far, and never shrinks. When a large workload runs first, then a smaller one, the smaller workload's padding rows retain data from the earlier large workload. cuBLAS `StridedBatchedEx` reads these padding rows as legitimate input.

### Debugging recipe
1. Run workloads in isolation — each passes
2. Run workloads in the failing sequence — fails at the smaller workload that follows a larger one
3. Dump the GEMM input buffer before the call — you see stale non-zero values in padding rows
4. Add `cudaMemsetAsync` of the GEMM input buffer to zero before each invocation
5. Failure disappears → confirms mode 3

### Workaround

```cpp
// Before gather/scatter into the GEMM input buffer:
CUDA_CHECK(cudaMemsetAsync(ws.b_a_all.data_ptr(), 0,
                           ws.b_a_all.nbytes(), stream));

// Then launch the gather kernel that fills actual rows
gather_kernel<<<...>>>(...);

// Then launch GEMM — padding rows are zero, don't corrupt output
cublasGemmStridedBatchedEx(...);
```

Allocating with `at::zeros()` is not enough — the bug comes from execution reuse, not uninitialized memory. The memset must happen before every GEMM invocation.

---

## Mode 4: FP8 requantization (128-block → per-tensor)

### Symptom
~12.5% per-element error on most workloads — far beyond any realistic numerical tolerance (typical tolerance budgets are ≤ 1% relative or absolute).

### What triggers it
Attempting to use a cuBLAS FP8 path that requires per-tensor scales. To get there from 128-block float32 data, you must requantize.

### Why it can't be fixed
FP8 E4M3 format has 3 mantissa bits. Collapsing 128×128 separate scales into a single scalar destroys precision for any block whose true scale differs significantly from the global. The error ~12.5% is not a bug — it's inherent to the conversion.

### Debugging recipe
Don't debug. Abandon the plan. No quantization tuning recovers from this.

### Workaround
Use CUTLASS with a custom blockwise epilogue, or hand-written tcgen05 that handles 128-block float32 directly. **Never route 128-block data through per-tensor scales.**

---

## Mode 5: `cublasLtMatmul` with `BLK128x128_32F`

### Symptom
API call returns `CUBLAS_STATUS_NOT_SUPPORTED`. `cublasLtMatmulAlgoGetHeuristic` returns **0 algorithms available**.

### What triggers it
Attempting to use cuBLASLt's native 128-block FP8 path on B200.

### Why it can't be fixed
cuBLAS 13.x on B200 does not implement this mode. The header enum exists but there are no backing kernels. Permanent blocker until NVIDIA ships a new cuBLAS version.

### Debugging recipe
None. The API call itself signals the problem.

### Workaround
Use CUTLASS with custom blockwise epilogue — that's the working alternative on B200.

---

## Mode 6: `cublasGemmGroupedBatchedEx` with FP8

### Symptom
Most workloads produce a runtime error (API crash). A few may pass only because they happen to fit the unbroken portion of the API.

### What triggers it
Calling `cublasGemmGroupedBatchedEx` with FP8 data types on B200.

### Why it can't be fixed
API crash on B200 with FP8. Reported internally as a cuBLAS bug.

### Workaround
Use CUTLASS `KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100` for grouped FP8 GEMM.

---

## Mode 7: CuTe DSL alignment + CUDA graph

### Symptom
All workloads produce **all-zero outputs** (total failure, not per-element error).

### What triggers it
Using the CuTe DSL grouped GEMM with:
- Per-expert row count not a multiple of 128 (DSL-required alignment), AND
- CUDA graph capture enabled

### Root cause
Two separate bugs interacting:
1. CuTe DSL requires 128-row alignment per expert; 64-row alignment causes illegal memory access
2. CUDA graph capture doesn't correctly record the CuTe DSL kernel launch parameters

### Debugging recipe
1. Disable CUDA graph capture — outputs become correct but individual workloads may still have illegal accesses
2. Align per-expert row counts to 128 — eliminates the alignment crash
3. Re-enable CUDA graph — outputs become zero again, so leave it disabled

### Workaround
Abandon CuTe DSL for MoE grouped GEMM. Use CUTLASS C++ templates directly, or hand-written tcgen05. CUDA graph is separately unreliable with MoE and not recommended regardless.

---

## Detection: which mode are you in?

When you see numerical-correctness failures:

```
All workloads fail with huge errors → Mode 4 (requantization) or Mode 7 (DSL+graph)
A fixed subset fails, same workloads every time, all above the FP8 CUTLASS threshold
    → Mode 1 or Mode 2 (FP8 CUTLASS path)
A subset fails only when run in sequence → Mode 3 (stale padding)
API call fails with NOT_SUPPORTED → Mode 5 (cuBLASLt BLK128)
API call crashes with RUNTIME_ERROR → Mode 6 (grouped batched)
All zero output, no errors thrown → Mode 7 (DSL+graph)
```

---

## General debugging workflow for FP8 issues

1. **Check dispatch first.** Run NCU, confirm the kernel name matches what you intended. Silent fallback to cuBLAS FP16 is common with CUTLASS FP8 paths.

2. **Compare against cuBLAS FP16 oracle.** If the failing FP8 path is correct at the algorithmic level, cuBLAS FP16 on the same inputs produces the reference output. Diff per-workload.

3. **Isolate by workload.** Run failing workloads one at a time. If failure persists, it's a per-workload correctness bug (likely mode 1, 2, 5, 6, or 7). If failure only appears in sequence, it's mode 3 (stale padding).

4. **Don't relax tolerance to hide a failure.** Loosening absolute/relative tolerance or element-match fraction can mask mode 4 without fixing it. Treat tolerance knobs as triage tools, never as a shipping configuration.

---

## Typical failure signatures by mode

- **Modes 1, 2, 3** — *a few specific rows* of the output are wildly wrong while the rest is correct. These fail element-match-fraction thresholds even if average error looks OK.
- **Mode 4** — *every element* is moderately wrong. Fails both average error and element-match-fraction.
- **Modes 5, 6, 7** — fail at the API level (no output produced, all-zero output, or runtime crash).
