# PyTorch → CUDA Migration

Direct path from a PyTorch reference MoE implementation to a pure CUDA kernel. Skip intermediate DSLs.

## When to migrate

Migrate to CUDA when the PyTorch reference has hit one of:

- GEMM throughput is below tensor-core peak (PyTorch `torch._scaled_mm` or `torch.nn.functional.linear` cannot access B200-specific FP8 scale formats efficiently)
- Host overhead from per-expert Python dispatch is visible in profiles
- Need fine-grained control over TMA, warp specialization, or Tensor Memory
- Need to fuse operations that framework ops cannot express (e.g., GEMM + custom scale epilogue, or routing + gather + quantize)

## Migration roadmap

The order matters. Each step creates the foundation for the next. Skipping ahead is the most common cause of wasted optimization rounds.

### Step 1: cuBLAS FP16 GEMM as the first baseline

**Do not start with a custom GEMM.** cuBLAS FP16 on B200 produces a strong first baseline and serves as the correctness oracle for every subsequent custom kernel.

**Setup:**

```cpp
cublasHandle_t handle;
cublasCreate(&handle);
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

// CRITICAL: enable BF16 tensor cores on B200 via FAST_16BF compute type
cublasGemmEx(handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    N, M, K,
    &alpha,
    B_fp16, CUDA_R_16F, K,
    A_fp16, CUDA_R_16F, K,
    &beta,
    D_bf16, CUDA_R_16BF, N,
    CUBLAS_COMPUTE_32F_FAST_16BF,   // <-- this activates BF16 tensor cores
    CUBLAS_GEMM_DEFAULT);
```

Without `CUBLAS_COMPUTE_32F_FAST_16BF`, cuBLAS on B200 runs a slower FP32 path and you will underestimate your FP16 baseline.

### Step 2: Grouped GEMM for MoE variable shapes

MoE expert token counts differ per batch. A per-expert loop (one `cublasGemmEx` per expert) incurs per-launch overhead that becomes visible at small T.

**Preferred:**

```cpp
// CUTLASS SM100 grouped FP8 collective
using GemmKernel = typename cutlass::gemm::kernel::GemmUniversal<
    ...
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100
>::CollectiveOp;
```

**Avoid:**
- `cublasGemmGroupedBatchedEx` with FP8 → RUNTIME_ERROR on B200
- Per-expert cuBLAS loop → fine for a first correctness pass, but replace with CUTLASS grouped GEMM before further optimization

### Step 3: Custom kernels for non-GEMM parts first

Before touching GEMM internals, write custom CUDA kernels for the parts that don't require tensor cores:

| Component | Why do it early |
|---|---|
| Top-k routing | Low complexity, removes Python dispatch |
| Gather (FP16) | Prepares quantized inputs; **keep even after moving to FP8** (serves as L2 warmup) |
| SwiGLU activation | Simple elementwise; lines up intermediate for GEMM2 input quantize |
| Quantize-to-FP8 (with 128-block scale) | Required before FP8 GEMM; fuse with SwiGLU output but **not** with GEMM2 input prep |
| Scatter | Final output combine; uint4 load/store gives +2.1% |
| Counting / prefix scan | Needed for GPU-side planner; move these off the CPU early |

**Crucial rule:** *Do not delete the FP16 gather kernel* even after the compute path is entirely FP8. Verified 2× degradation when removed — the gather warms up L2 for subsequent kernels.

### Step 4: Upgrade GEMM backend only after host-GPU sync is eliminated

This is the most commonly violated rule. The biggest single optimization observed was **not** in GEMM — it was eliminating `cudaStreamSynchronize` by moving routing metadata construction to the GPU. Observed gain: **+16.3%**.

Order of attack:

1. Profile with NCU; confirm CPU stalls are visible (look for idle SM windows between routing and GEMM)
2. Move `expert_count` prefix scan, `active_expert` sort, and CUTLASS argument setup onto the GPU (see [`code-examples/zero-sync-fast-path.md`](code-examples/zero-sync-fast-path.md))
3. Use `programmaticStreamSerialization` (PSS) for dependent kernel launches
4. Remove `threadfence` that aren't required
5. **Only now**: move from cuBLAS FP16 to CUTLASS FP8 grouped GEMM

Jumping to step 5 first is correct in isolation but creates a worse measurement baseline — you'll see +60% from the GEMM swap and miss that +16.3% was available for free upstream.

### Step 5: Vendor CUTLASS headers into your code tree

- **Don't** depend on system CUTLASS install — version drift across machines breaks reproducibility
- Embed CUTLASS source in your project
- Compile at runtime with `nvcc -arch=compute_100a` → `.so` cached at `/tmp/`
- After measurements stabilize, consider **static linking with embedded headers** — gave **+7%** by eliminating dynamic-load cache effects. Cost is a larger source file.

See [`code-examples/cutlass-jit-compile.md`](code-examples/cutlass-jit-compile.md) for the JIT compilation pattern.

---

## Anti-patterns in migration

| Anti-pattern | Why it fails | What to do instead |
|---|---|---|
| Start with a custom hand-written GEMM | Matching cuBLAS/CUTLASS throughput is a significant undertaking; without a baseline you have nothing to compare against | cuBLAS FP16 first, then CUTLASS FP8 grouped, then tcgen05 if needed |
| Skip the cuBLAS correctness oracle | When a custom kernel produces wrong results, you have no ground truth to diff against | Always keep the cuBLAS FP16 path compilable, even after custom kernel is default |
| Ignore CUTLASS silent fallback | CUTLASS can silently fall back to cuBLAS internally and look plausibly fast | After every CUTLASS change, verify with NCU that the intended kernel name is dispatched |
| Use CuTe DSL for MoE grouped GEMM | Alignment constraints (128-row) conflict with MoE expert boundaries; CUDA-graph incompatibility causes further failures | Use CUTLASS C++ templates directly, or hand-written tcgen05 |
| Mix Python/PyTorch tensor ops inside the CUDA path | Hybrid pipelines cause debugging confusion and stream-synchronization bugs | Commit to pure CUDA once migrated; the PyTorch reference stays as oracle only |
| Call `torch.cuda.synchronize()` inside the kernel | Defeats the entire zero-sync fast path | Use CUDA events or PSS (`programmaticStreamSerialization`) instead |
| Fuse SwiGLU + FP8 quantize in one kernel | Causes correctness failures on the FP8 CUTLASS path (a fixed subset of workloads) | Keep SwiGLU and FP8 quantize as separate kernels; fuse quantize with *prior* kernel's output write instead |

---

## Structure of a typical MoE CUDA pipeline

After migration, the kernel typically has this structure:

```
1. Routing              — top-k scores, expert assignment (GPU-side)
2. Counting + prefix scan — expert tokens counts, offsets (GPU-side)
3. Active-expert sort    — sort experts by token count (GPU-side)
4. Gather (FP16)         — pack per-expert rows into contiguous buffer; WARMS UP L2
5. Quantize to FP8 (128-block scale) — produces FP8 + scale factors for GEMM1
6. GEMM1                 — [rows × 7168] × [E × 4096 × 7168]ᵀ → [rows × 4096]
   (CUTLASS FP8 grouped, or tcgen05)
7. SwiGLU                — split → silu(gate) * up → [rows × 2048]
8. Quantize to FP8       — produces FP8 + scale factors for GEMM2
9. GEMM2                 — [rows × 2048] × [E × 7168 × 2048]ᵀ → [rows × 7168]
   (CUTLASS FP8 grouped, or tcgen05, or cuBLAS FP16 for large T)
10. Pull-scatter         — combine per-token contributions from k experts into final output
```

Routing, counting, sorting, and CUTLASS argument setup (steps 1–3 + metadata for steps 6, 9) are candidates for the **zero-sync fast path** — move them all onto the GPU to eliminate `cudaStreamSynchronize`.

Gather (step 4) should stay as FP16 even if downstream is FP8. It serves as L2 cache warmup — deleting it regresses.

Quantize (steps 5, 8) must not be fused with SwiGLU or with GEMM input prep — fusion triggers FP8 correctness failures on the CUTLASS path.

---

## Typical performance landmarks during migration

| State | Observed speedup vs. PyTorch reference |
|---|---|
| PyTorch reference (`torch._scaled_mm` + torch ops) | 1× |
| **cuBLAS FP16 CUDA baseline with `COMPUTE_32F_FAST_16BF`** | **~26.8×** |
| + Zero-sync fast path | ~31× |
| + CUTLASS FP8 grouped GEMM | **~43×** |
| + BF16 fast compute + dual-tile dispatch | ~54× |
| + pipeline overlap + metadata fusion | ~57–63× |
| + all subsequent sync-reduction + launch-fusion items | ~81× |
| + static compile + tcgen05 on both GEMMs | ~91× stable |

These are observed landmarks from production MoE optimization. Use them as sanity checks: if after step 1 you're seeing 10× instead of ~26×, something is wrong with BF16 fast compute or the oracle itself.
