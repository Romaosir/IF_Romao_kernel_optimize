# Tuning Knobs — Measured Sweet Spots

Every value below was measured in production. The sweet spot is the default; the "regression at" column names the boundary where things break. Values that changed by ≤ 0.5% are marked **noise** and should not be tuned speculatively.

## CUTLASS / GEMM dispatch

| Knob | Sweet spot | Regression at |
|---|---|---|
| CUTLASS tile M | **64** for `max_M ≤ 256`, **128** for `max_M > 256` | `M = 256` (needs 2-SM cluster, −18%) |
| CUTLASS tile N | **128** | `N = 256` ⇒ **−60%** (wave quantization) |
| CUTLASS tile K | **128** | `K = 64` ⇒ crash (breaks `ScaleGranularityK=128`) |
| 1-SM vs 2-SM cluster | **1-SM** | 2-SM ⇒ −18%; `can_implement()` also silently fails for small M |
| `max_swizzle_size` | **4** | smaller values hurt L2 locality; larger values noise |
| CUTLASS `StageCount` | **auto-carveout** | `StageCount<3>` ⇒ −20% |
| `kMPad` | **16** | `kMPad=1` ⇒ crash (hard alignment requirement) |
| Prep kernel | **dual (fused) for GEMM1+GEMM2** | separate preps ⇒ one extra launch per invocation, −2–3% |

## tcgen05

| Knob | Sweet spot | Regression at |
|---|---|---|
| `TCGEN05_MIN_T` | **~500** | default 0 ⇒ regresses small-T where CUTLASS wins |
| `TCGEN05_MAX_T` | **~10 000** | 16 000 ⇒ regresses very-long-seq workloads |
| `NUM_STAGES` (pipeline depth) | **7** | 5 ⇒ hurts small-T |
| Tile shape | **BM = BN = BK = 128** | — |
| Grid size | **full SM count (148 on B200)** | 110 ⇒ −4.5% |
| `__launch_bounds__` | **`(192, 1)`** — 6-warp | tighter → compiler can't fit required regs |
| Tile-decode search | **linear scan** for `G ≤ 32` | binary search ⇒ more branch mispredictions |
| Tile swizzle | **S = 4** for L2 B-matrix locality | — |

## Other kernels

| Knob | Sweet spot | Regression at |
|---|---|---|
| `pull_scatter` `__launch_bounds__` | **`(256, 4)`** | ≥ 5 ⇒ register spill |
| `swiglu` rows per block | **8** | — |
| Scatter load/store width | **`uint4` (128-bit)** | narrower gives up bandwidth |
| Grid estimate for launch | **`t × 1.25` (tight)** | `t × kTopK` over-launches |

## Bucketing / long-seq dispatch

| Knob | Sweet spot | Regression at |
|---|---|---|
| Bucket threshold | **128** | 64 and 256 both regress |
| Max buckets | **6** | 1 ⇒ correctness failure |
| Short-chunk size | **512** | 1024 fails shortest seq; 256 neutral |
| GEMV fast-path trigger | **M ≤ 2** (dedicated kernel, 148-CTA grid) | — |
| `GEMM_N_ALIGN` | **1** | 8–32 supported; 16 neutral |

## Compilation flags

| Flag | Required / recommended | Effect if missing |
|---|---|---|
| `-arch=compute_100a,code=sm_100a` | **required** for tcgen05 | silent PTX fallback, no tensor-core speedup |
| `-O2` | required | — |
| `-Xcompiler -fPIC` | required for `--shared` | linker error |
| `-lcuda` | required for driver-API symbols | `dlopen` failures on `cu*` functions |
| `--expt-relaxed-constexpr` | required in static-compile + embedded-CUTLASS builds | build failure |
| `-cudart=static` | optional | — |
| `--use_fast_math` | optional, minor | — |
| `-maxrregcount=128` | **do not use** | catastrophic regression |
| `-sm_count=74` override | **do not use** | catastrophic regression |

## Host-side

| Knob | Sweet spot | Notes |
|---|---|---|
| `cublasSetMathMode` | `CUBLAS_TF32_TENSOR_OP_MATH` | activates tensor cores |
| `cublasSetAtomicsMode` | `CUBLAS_ATOMICS_NOT_ALLOWED` | deterministic + slight speedup |
| cuBLAS workspace | pre-allocate **≥ 32 MB** | default is too small for some shapes |
| `cudaFuncCachePreferL1` | **on non-GEMM kernels only** | neutral or harmful on GEMM |
| Pinned host buffers for `h_offsets` | **yes**, if any H2D/D2H remains | only matters if the zero-sync fast path isn't complete |

## How to use this table

1. When proposing a knob change, check it against the "regression at" column first — many "obvious" directions are already known-bad.
2. If your workload differs substantially from a typical MoE setup, verify the sweet spot empirically — but change one knob at a time.
3. Treat the `compute_100a` flag as the single most common silent failure — every "tcgen05 didn't help" investigation should verify this flag is present.
