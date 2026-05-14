# Dual TMA Descriptors for GEMM1 + GEMM2 (+3.3%)

When tcgen05 is applied to both GEMM1 and GEMM2, each GEMM has different shapes — so each needs its own TMA descriptor pair. This change is responsible for the **+3.3%** "tcgen05 on GEMM1 + GEMM2" item in the optimization ladder (observed going from tcgen05 GEMM1 only to tcgen05 on both GEMMs).

## Why separate descriptors

GEMM shapes in a typical MoE kernel (example dimensions):

| GEMM | A shape (per-group) | B shape (per-expert) |
|---|---|---|
| GEMM1 | `[rows, K1]` — large K (e.g., 7168) | `[E, N1, K1]` — intermediate × hidden |
| GEMM2 | `[rows, K2]` — smaller K (e.g., 2048) | `[E, N2, K2]` — hidden × intermediate |

A single TMA descriptor encodes the global shape, stride, and layout of the tensor. Since the shapes differ, one descriptor cannot be reused — swapping at kernel entry would mean rebuilding the descriptor which is expensive.

## Implementation — specification

Maintain **four** `CUtensorMap` objects (globally) plus a "TMA ready" flag:

- `g_A_tmap_1`, `g_B_tmap_1` — GEMM1
- `g_A_tmap_2`, `g_B_tmap_2` — GEMM2

Build them once in a `setup` function using `cuTensorMapEncodeTiled` (CUDA driver API). Key parameters to get right for each descriptor:

### A descriptor (the per-expert-rows activation tensor)

- `dataType` = `CU_TENSOR_MAP_DATA_TYPE_UINT8` (FP8 is 1-byte — use the byte encoding)
- `rank` = 2 (rows × K)
- `globalAddress` = set to `nullptr` at setup; update per invocation (see below)
- `globalDim` = `{K, MAX_ROWS}` — K first because the tensor is row-major
- `globalStrides` = `{1, K}` in bytes (FP8 = 1 byte per element)
- `boxDim` = `{BK, BM}` — the tile shape you'll load
- `elementStrides` = `{1, 1}`
- `interleave` = `CU_TENSOR_MAP_INTERLEAVE_NONE`
- `swizzle` = `CU_TENSOR_MAP_SWIZZLE_128B` (avoids smem bank conflicts on 128-byte rows)
- `l2Promotion` = `CU_TENSOR_MAP_L2_PROMOTION_NONE`
- `oobFill` = `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`

### B descriptor (per-expert weight tensor — 3D)

Same as A, but:
- `rank` = 3 (K × N × expert)
- `globalDim` = `{K, N, num_experts}`
- `globalStrides` = `{1, K, N * K}` in bytes
- `boxDim` = `{BK, BN, 1}`
- `elementStrides` = `{1, 1, 1}`

### The four descriptors differ in

- `globalDim[0] = K` — `K1` for GEMM1, `K2` for GEMM2
- `globalDim[1] = N` — `N1` for GEMM1, `N2` for GEMM2
- `globalStrides[2]` (for the B tensors) reflects the different `N × K` product

### Storage

Allocate each `CUtensorMap*` on managed memory (`cudaMallocManaged`) so TMA instructions running on the GPU can read the descriptor. Sized as `sizeof(CUtensorMap)` each (128 bytes on current drivers).

## Two entry points

Export `extern "C"` functions `tcgen05_grouped_gemm` and `tcgen05_grouped_gemm2` that the main kernel dlsym's. Each one:

1. Guards on `g_tcgen05_tma_ready` (return non-zero if setup wasn't called)
2. Updates the `globalAddress` field of its two descriptors to the current invocation's A and B pointers (the rest of the descriptor is constant — don't re-encode)
3. Launches the kernel with its descriptor pair

Rough skeleton:

```cpp
extern "C" int tcgen05_grouped_gemm(CutlassBwArgs* a, cudaStream_t stream) {
    if (!g_tcgen05_tma_ready) return -100;

    update_tma_descriptor_address(g_A_tmap_1, a->A);
    update_tma_descriptor_address(g_B_tmap_1, a->B);

    launch_tcgen05_kernel(
        g_A_tmap_1, g_B_tmap_1,     // GEMM1 descriptors
        (nv_bfloat16*)a->D,
        (const float*)a->SFA, (const float*)a->SFB,
        a->m_indptr, a->expert_ids,
        a->num_groups, a->N, a->K,
        stream);
    return 0;
}

extern "C" int tcgen05_grouped_gemm2(CutlassBwArgs* a, cudaStream_t stream) {
    if (!g_tcgen05_tma_ready) return -100;

    update_tma_descriptor_address(g_A_tmap_2, a->A);
    update_tma_descriptor_address(g_B_tmap_2, a->B);

    launch_tcgen05_kernel(
        g_A_tmap_2, g_B_tmap_2,     // GEMM2 descriptors
        (nv_bfloat16*)a->D,
        (const float*)a->SFA, (const float*)a->SFB,
        a->m_indptr, a->expert_ids,
        a->num_groups, a->N, a->K,
        stream);
    return 0;
}
```

## Updating the base pointer in-place

TMA descriptors on managed memory can have their `globalAddress` field updated between invocations — this avoids re-encoding the whole descriptor:

```cpp
static inline void update_tma_descriptor_address(CUtensorMap* tmap, void* base) {
    // The CUtensorMap's global address lives at a fixed offset in the opaque
    // 128-byte structure. For Blackwell, it's typically at bytes 0..7.
    // Safer: re-encode with cuTensorMapEncodeTiled (small overhead).
    //
    // Fastest (but depends on ABI stability):
    *reinterpret_cast<void**>(tmap) = base;
}
```

If the ABI offset isn't stable, just re-encode each invocation. The cost is a single driver API call.

## Dispatcher in the main kernel

Call `tcgen05_grouped_gemm` (descriptor pair 1) for GEMM1. On non-zero return, fall back to the CUTLASS path for both GEMMs (don't try to run GEMM2 on tcgen05 if GEMM1 failed, since the state may be inconsistent).

After SwiGLU + FP8 quantize, if GEMM1 succeeded on tcgen05 and the GEMM2 entry point is available, call `tcgen05_grouped_gemm2` (descriptor pair 2) for GEMM2. Same non-zero-return-means-fallback contract.

## Observed gain

Going from tcgen05 on GEMM1 only to tcgen05 on both GEMM1 and GEMM2 yields **+3.3%**. Run-to-run variance at this level of the stack is ~±0.6×, so multi-run averaging is required.

The gain from adding GEMM2 is almost entirely due to the dual descriptor setup. Without separate descriptors, reusing GEMM1's descriptor for GEMM2 inputs would load the wrong shape and produce incorrect results or crashes.

## Pitfalls

| Pitfall | Symptom |
|---|---|
| Reused single descriptor for both GEMMs | Out-of-bounds loads, correctness failures or crashes |
| Forgot to update `globalAddress` between calls | Kernel loads stale data from the previous invocation |
| Wrong `globalDim` / `globalStrides` | TMA loads wrong shape — correctness failures |
| `CU_TENSOR_MAP_SWIZZLE_NONE` instead of `SWIZZLE_128B` | Shared memory bank conflicts, slower |
| Descriptor on non-managed memory | TMA instruction can't read the descriptor, crash |

## When dual descriptors are worth it

- GEMM1 and GEMM2 have different `N` or `K`
- tcgen05 is already your default GEMM backend
- You've already done single-descriptor GEMM1 and verified correctness

If you're still on CUTLASS for one of the GEMMs, dual TMA descriptors aren't relevant yet. CUTLASS manages its own internal descriptors.
