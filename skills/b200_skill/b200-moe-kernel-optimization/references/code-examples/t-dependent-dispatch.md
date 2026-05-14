# T-Dependent GEMM Backend Dispatch (+1.4%)

Dispatch GEMM2 between CUTLASS FP8 and cuBLAS FP16 based on total token count `T`. Workaround for the FP8 CUTLASS path's fragility at large T, while preserving its speedup at small T.

## The rule

```
GEMM2 (smaller K, typically K ≤ ~2k):
  T ≤ ~2000 → CUTLASS FP8 grouped (better small-T occupancy)
  T >  ~2000 → cuBLAS FP16 (COMPUTE_32F_FAST_16BF, better large-T saturation)

GEMM1 (larger K, typically K ≥ ~4k):
  Always FP8 — bandwidth win on large K is too large to give up
```

## Dispatch code

The dispatch is structural:

```
compute total_t = sum of active expert token counts
threshold = ~2000  (see "Threshold tuning" below)

if total_t <= threshold:
    call the CUTLASS FP8 grouped GEMM wrapper (same one used for GEMM1,
    but configured with GEMM2 shapes). On non-zero return, fall through
    to the cuBLAS path.

else:
    for each active expert g with m > 0 rows:
        call cublasGemmEx with:
          - compute type = CUBLAS_COMPUTE_32F_FAST_16BF (activates BF16 tensor cores)
          - A = per-expert row slice of the BF16 intermediate
          - B = per-expert weight slice (BF16 copy — see "BF16 weight copy" below)
          - D = per-expert output slice
          - M = this expert's row count, N = hidden_dim, K = intermediate_dim
```

Both branches produce the same output buffer layout so the rest of the pipeline doesn't know which path was taken.

## The cuBLAS setup (one-time)

Required state for the large-T path:

One-time setup for the large-T path:

1. `dlopen` libcublas (see `cutlass-jit-compile.md` for the dynamic-load pattern)
2. `cublasCreate` → handle
3. `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)` — activates tensor cores for FP16/BF16
4. `cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED)` — deterministic + slight speedup
5. `cublasSetWorkspace` with a **pre-allocated workspace ≥ 32 MB** — cuBLAS default is too small for some shapes and silently falls back to slower algorithms

Call this once at module init (or lazily on first large-T invocation).

## The BF16 weight copy

For the large-T cuBLAS path, you need BF16 weights (cuBLAS FP8 is unusable on B200 — see `hardware-constraints.md`). Two options:

**Option A — one-time dequantize at warmup (preferred):**

At warmup, allocate a `bfloat16` weight tensor of shape `[num_experts, N, K]` and launch a dequantize kernel that reads the FP8 weight + its 128×128-block FP32 scales and writes the BF16 result. The dequantize kernel is routine — one thread per element, multiply FP8 by the matching block scale, convert to BF16.

Keep the BF16 weight tensor for the lifetime of the module. cuBLAS reads from it directly; no per-invocation cost after warmup.

**Option B — dequantize on demand:**

Only dequantize the active expert's weights at each invocation. Higher runtime cost but zero warmup overhead. Not typically worth it — warmup cost amortizes quickly.

## Why not always use cuBLAS FP16?

- Small T: cuBLAS per-expert loop incurs ~10µs per launch × `active_expert_count`. For a MoE batch with several active experts, that overhead accumulates into tens of microseconds — which dominates small-T workload runtime.
- CUTLASS grouped launches one kernel for all experts, avoiding this.

## Why not always use CUTLASS?

- Large T: CUTLASS's static schedule and blockwise scale indirection impose overhead that doesn't fit well with very large tile counts.
- More importantly, the FP8 CUTLASS path has the correctness fragility documented in `fp8-correctness-modes.md` — certain workloads fail only on this path.
- The T-dependent split lets you ship: CUTLASS FP8 for the fast small-T path, cuBLAS FP16 for the correct-and-stable large-T path.

## Threshold tuning

The `T ≈ 2000` threshold works on the MoE workload mix observed in practice. It may differ for:
- Different expert counts (more experts → lower threshold)
- Different `K` dimensions (smaller K → lower threshold — less bandwidth-dominated)
- Different hardware (this is SM100-specific)

To tune, sweep T in your benchmark and plot both backends. Pick the crossover point.

## Gain observed

Going from "CUTLASS always" to "T-dependent split": **+1.4%** observed in practice.

This is a small gain on its own, but it's the *correct shipping configuration* — without it, a subset of workloads fails correctness on the CUTLASS path. The gain comes from choosing the faster backend where both are valid.

## Pitfalls

| Pitfall | Symptom |
|---|---|
| Forgot `COMPUTE_32F_FAST_16BF` | cuBLAS runs slow FP32 path on B200 |
| Used FP8 on GEMM1 large-T path (not always FP8) | Performance drops from losing FP8 bandwidth |
| Launched cuBLAS without workspace | Performance inconsistent; some shapes fail |
| Per-expert loop without host-side active expert list | Re-introduces sync (breaks zero-sync fast path) |
| Hardcoded threshold too high | Small-T gains lost to cuBLAS launch overhead |
| Hardcoded threshold too low | Fragile CUTLASS path runs on workloads it can't handle |

## Environment variable override

Allow runtime control for debugging:

```cpp
static int get_t_threshold() {
    const char* env = std::getenv("T_THRESHOLD");   // name as appropriate for your project
    return env ? atoi(env) : 2000;
}
```

Useful when:
- Disabling the dispatch entirely to isolate a correctness issue (`T_THRESHOLD=0` → always cuBLAS; `T_THRESHOLD=999999` → always CUTLASS)
- Sweeping the threshold during tuning
