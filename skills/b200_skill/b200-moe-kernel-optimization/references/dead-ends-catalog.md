# Dead Ends Catalog

Every entry below was attempted in production MoE optimization on B200 and produced a worse or broken result. **Check this catalog before proposing anything that matches.**

## How to use this catalog

- If your plan matches an entry, the default answer is **don't try it again** — the evidence is already in hand
- If you believe circumstances differ (different problem shape, different data format, different CUDA version), explicitly call out the difference before retrying
- Adding to this catalog is valuable — every future optimization pass benefits

---

## Correctness failures (kernel doesn't pass)

### Fuse SwiGLU + FP8 quantize into one kernel

- **Symptom:** A fixed subset of workloads fails with numerical-correctness errors (large absolute and relative output deviation) while others pass
- **Affected workloads:** those above the FP8 CUTLASS path threshold (workloads below the threshold use cuBLAS FP16 and don't fail)
- **Attempts:** direct fusion, and a rewrite with shared-memory staging — both failed the same workload set
- **Root cause:** FP8 CUTLASS grouped GEMM path is extremely sensitive to input layout. The fused SwiGLU-quantize output doesn't match the scale-alignment assumption CUTLASS makes.
- **What to do instead:** Keep SwiGLU as separate kernel. Fuse quantize with **upstream** write (e.g., directly after GEMM1 epilogue produces BF16) but not with downstream consumption.

### Delete FP16 gather kernel (intending to go "full FP8")

- **Symptom:** 2× performance degradation (verified across multiple iterations)
- **Root cause:** The FP16 gather kernel serves as L2 cache warmup — it pulls expert weights into L2 before subsequent kernels need them. With gather removed, subsequent kernels miss L2.
- **What to do instead:** Always keep FP16 gather, even after the compute path is entirely FP8. It's the cheapest cache prefetch you'll find.

### FP8 requantization to per-tensor scales

- **Symptom:** ~12.5% per-element error on most workloads — exceeds any realistic numerical tolerance
- **Root cause:** Converting 128-block float32 scales to a single per-tensor scalar destroys precision. FP8 mantissa is only 3 bits (E4M3) — without per-block scaling, most values clip or lose precision.
- **What to do instead:** Use CUTLASS custom blockwise epilogue or hand-written tcgen05 that handles 128-block float32 scales directly. Abandon any plan that routes through per-tensor scales.

---

## Performance regressions (kernel is correct but slower)

### 2-SM CUTLASS cluster (M=256 tile variant)

- **Observed delta:** −18%
- **Root cause:** Wave quantization on SM100. The 2-SM cluster produces fewer independent waves across 148 SMs; for typical MoE problem shapes this leaves SMs idle.
- **What to do instead:** Stay at 1-SM with M ∈ {64, 128}. 2-SM was not decisively faster on any workload tested.

### 128×256×128 wide N-tile

- **Observed delta:** −60%
- **Root cause:** N=256 is too wide for typical MoE expert N (4096 or 7168). Wave quantization destroys efficiency.
- **What to do instead:** Keep N=128. Do not widen.

### 64×256×128 wide N-tile

- **Observed delta:** Large regression (similar magnitude to the 128×256 variant)
- **Root cause:** Same wave quantization issue.
- **What to do instead:** Same — N stays at 128.

### Expert chunked pipeline

- **Observed delta:** −8.7%
- **Root cause:** Breaking expert work into pipeline chunks added synchronization and launch overhead that exceeded any parallelism gain.
- **What to do instead:** Keep all experts in one grouped GEMM call. The "parallelism" inside grouped GEMM is already what you need.

### Bitmask `expert_used` (uint8[256] → uint32[8])

- **Observed delta:** −1.5%
- **Root cause:** Bit manipulation (set, test, popcount) had higher instruction cost than the memory-access savings. The original uint8 array was already L1-cached cheaply.
- **What to do instead:** Don't optimize small arrays into bitmasks unless profiling shows the array access is the bottleneck.

### MXFP8 full pipeline (convert all weights at warmup)

- **Observed delta:** Neutral or regression on end-to-end
- **Root cause:** Weight format conversion at warmup adds startup cost that doesn't amortize; MXFP8 tensor-core speedup on individual GEMMs is offset by the conversion overhead and extra memory traffic.
- **What to do instead:** If data is 128-block float32, keep it that way and use CUTLASS/tcgen05 with matching epilogue. MXFP8 is B200-native but not worth the conversion unless data is already in that format.

---

## API / toolchain failures (can't be used at all)

### `cublasGemmGroupedBatchedEx` with FP8

- **Status:** most workloads produce a runtime error (API crash)
- **CUDA version observed:** cuBLAS 13.2.1 on B200
- **What to do instead:** Use CUTLASS `KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100` for grouped FP8 GEMM.

### `cublasLtMatmul` with `BLK128x128_32F` FP8 block-scaling

- **Status:** `CUBLAS_STATUS_NOT_SUPPORTED`
- **Detail:** Header enum exists, but `cublasLtMatmulAlgoGetHeuristic` returns **0 algorithms**
- **What to do instead:** Use CUTLASS with custom blockwise epilogue.

### DeepGEMM (via FlashInfer integration)

- **Observed:** 175 TFLOPS vs CUTLASS 333 TFLOPS on same problem
- **Root cause:** DeepGEMM is optimized for different problem shapes; CUTLASS with blockwise epilogue is a better fit here.
- **What to do instead:** Stick with CUTLASS or tcgen05.

### L2 persistence (`cudaAccessPolicyWindow`)

- **Observed:** Regression (no measurable benefit, adds overhead)
- **Root cause:** MoE expert weights are large and not reused within a single invocation — L2 persistence has nothing useful to hold onto.
- **What to do instead:** The FP16 gather kernel already provides effective L2 prefetch; don't add another mechanism.

### CUDA graph capture

- **Observed:** Unstable — captures fail with dynamic shapes, or produce wrong results (all zeros) when used with CuTe DSL kernels
- **Root cause:** Dynamic expert token counts per invocation + per-invocation memory allocation breaks graph capture assumptions
- **What to do instead:** Use individual kernel launches with PSS for chaining. CUDA graph is not viable for dynamic MoE.

### CuTe DSL for MoE grouped GEMM

- **Status:** Abandoned after repeated correctness failures
- **Root cause:** 128-row alignment requirement conflicts with per-expert token boundaries; interacts poorly with CUDA graph
- **What to do instead:** Use CUTLASS C++ templates directly, or hand-written tcgen05 with manual alignment.

---

## Neutral (no benefit either way — don't bother)

### Tanh-sigmoid substitution for SwiGLU sigmoid

Attempted replacing `1/(1+exp(-x))` with `0.5 + 0.5*tanhf(0.5*x)`. Neutral result — B200 SFU handles both paths with similar throughput. Not worth the numerical risk.

### Expert-sort removal

Skipping the active-expert sort step was tested. Marginal effect — the L2 locality benefit from sorted experts was small. Keep the sort.

### CUTLASS EVT SwiGLU epilogue fusion

Investigated but not shipped. Effort was high and the expected gain was uncertain. In principle this could fuse SwiGLU into GEMM1's epilogue, saving a kernel, but the EVT machinery for blockwise-scaled FP8 is complex and the correctness risk is non-trivial. Revisit only after every item in the optimization ladder is exhausted.

### Warp specialization attempts in Triton (pre-migration)

Warp specialization in Triton was attempted multiple times in production. No stable gain was found.

### Routing micro-opts (`__restrict__` / `#pragma unroll`)

Adding these annotations to the routing kernel had no effect — the compiler was already handling the equivalent optimization. Not harmful, just don't expect a speedup.

---

## CUTLASS-specific crashes and regressions (additional)

### `StageCount<3>` for CUTLASS grouped GEMM

- **Observed:** −20% regression
- **Root cause:** Insufficient pipeline depth for the tile shape; SM stalls waiting for loads
- **What to do instead:** Use `StageCountAutoCarveout<sizeof(EC)>` (the default) — it picks the right depth from shared-memory availability

### `-maxrregcount=128` compiler flag

- **Observed:** Catastrophic regression
- **Root cause:** Forces register spill on kernels that need more than 128 registers per thread; spill is much slower than extra occupancy is worth
- **What to do instead:** Let the compiler pick, or tune via `__launch_bounds__` per-kernel

### `sm_count=74` hardware-info override

- **Observed:** Catastrophic regression
- **Root cause:** CUTLASS scheduler uses the reported SM count to partition tiles; mismatching the real count (148) produces a bad schedule
- **What to do instead:** Always query `cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0)` — don't hardcode

### CUTLASS K-tile `128 → 64`

- **Observed:** Crash
- **Root cause:** Breaks the `ScaleGranularityK=128` invariant required by the blockwise FP8 epilogue
- **What to do instead:** Keep K-tile at 128 for blockwise FP8

### `kMPad = 1` instead of the default 16

- **Observed:** Crash
- **Root cause:** Hard alignment requirement for grouped blockwise FP8 path; M must be padded to multiple of 16
- **What to do instead:** Keep default `kMPad = 16`

### Mapped-memory CPU spin-wait for `total_tight_rows`

- **Observed:** −7.5% regression
- **Root cause:** Page-fault cost on mapped memory is worse than a regular `cudaStreamSynchronize`, which itself is worse than the zero-sync fast path
- **What to do instead:** Apply the zero-sync fast path properly; don't try shortcuts via mapped memory

### CUTLASS 4.1 Example-92 path (`blockscaled_rcgrouped`, `MoEProblemShape`)

- **Status:** Blocked on CUTLASS 4.1 or earlier
- **Root cause:** Templates introduced only in CUTLASS 4.2
- **What to do instead:** Upgrade CUTLASS, or stick with the `KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100` path which works on 4.1+

### `KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100` with 256×128 tile

- **Observed:** Neutral-to-broken. `can_implement()` falsely accepts for small M but produces wrong output at runtime
- **Root cause:** 2-SM path isn't validated for the small-M problem shapes MoE produces
- **What to do instead:** 1-SM path with 64 or 128 tile (see `tuning-knobs.md`)

### `cutlass_fused_moe` with `use_deepseek_fp8_block_scale=True`

- **Observed:** `NotImplementedError` at runtime on SM100
- **Root cause:** Path implemented only for SM90
- **What to do instead:** Hand-roll the CUTLASS grouped-GEMM path as the skill recommends

### DeepGEMM UE8M0 scale conversion

- **Observed:** Up to 2× per-block error for non-power-of-2 scales
- **Root cause:** Conversion to UE8M0 discards mantissa; scales that aren't exact powers of 2 are quantized to the nearest representable scale
- **What to do instead:** If your scales aren't power-of-2 blockwise float32, don't use DeepGEMM on this path

---

## CUDA-runtime / compiler traps (additional)

### Streaming loads/stores (`__ldcs` / `__stcs`)

- **Observed:** Regression
- **Root cause:** Bypasses L2 — but in this pipeline, L2 hit-rate is load-bearing (gather kernel warms L2)
- **What to do instead:** Don't use streaming variants here; the default L1/L2-caching loads are what you want

### Cooperative kernel launch

- **Observed:** Fails to launch
- **Root cause:** Cooperative launch has a lower grid-size limit than the MoE working set requires
- **What to do instead:** Use regular launch + PSS for dependency

### `cudaMemPrefetchAsync`

- **Observed:** N/A — API not applicable
- **Root cause:** Requires managed (UVM) memory; the pipeline doesn't allocate managed memory for performance buffers
- **What to do instead:** Rely on the FP16 gather kernel for L2 prefetch; don't introduce UVM

---

## CuTe DSL failure modes (expanded)

### 64-row per-expert alignment

- **Observed:** Illegal memory access (crash)
- **Root cause:** CuTe DSL grouped GEMM requires 128-row alignment per expert; 64-row alignment causes out-of-bounds access during MMA
- **What to do instead:** CUTLASS C++ templates, or pad per-expert rows to 128

### 128-row alignment combined with CUDA graph capture

- **Observed:** All-zero output (total failure)
- **Root cause:** CUDA graph capture doesn't correctly record CuTe DSL kernel launches; the captured graph executes but produces zeros
- **What to do instead:** Don't mix CuTe DSL and CUDA graphs; see `fp8-correctness-modes.md` mode 7

---

## MXFP8 warmup-conversion path (expanded)

The main-catalog entry lists "MXFP8 full pipeline" as neutral. Additional detail on why it's not worth pursuing: MXFP8 requires a swizzled `LayoutSF` (`Stride<Stride<_16,_4>, Stride<_0,_1>>`) for the 32-element block scales, and the conversion-plus-swizzle cost offsets the native tensor-core speedup. Only worth the effort if your data is already in MXFP8 layout upstream.

---

## Retired: Triton as an intermediate

A Triton prototype phase was historically used as a stepping stone from PyTorch to CUDA. In retrospect, the Triton phase is not needed — direct PyTorch → CUDA via cuBLAS FP16 reaches the same state faster. **This skill documents direct PyTorch → CUDA migration;** Triton is not part of the recommended path.

If you find yourself considering Triton:
- For rapid prototyping of a novel algorithm — fine, use it temporarily
- For an end-state production kernel on B200 — skip it, go straight to cuBLAS FP16 + CUTLASS FP8

---

## Summary: patterns to avoid

| Pattern | Why it's a dead end |
|---|---|
| Start with a custom GEMM | Nothing to compare against; cuBLAS/CUTLASS is the baseline |
| Reach for tcgen05 first | Only +3–5%; misses the +60% CUTLASS swap and +16% zero-sync upstream |
| Fuse anything with FP8 quantize | Correctness failures on the CUTLASS path |
| Remove "redundant" gather kernels | L2 warmup loss |
| Widen tiles to "saturate" | Wave quantization kills efficiency on SM100 |
| Use CUDA graphs for MoE | Dynamic shapes incompatible |
| Use CuTe DSL for MoE grouped GEMM | Alignment + graph issues |
| Try any cuBLAS FP8 path | All four fail on B200 |
