---
name: b200-moe-kernel-optimization
description: Optimize Mixture-of-Experts (MoE) CUDA kernels on NVIDIA B200 (Blackwell, SM100). Use when writing, migrating, debugging, or optimizing FP8/FP16 MoE operators that involve grouped GEMM, per-expert routing, SwiGLU activation, gather/scatter, or Blackwell-specific features (TMA, tcgen05, TMEM). Covers PyTorch→CUDA migration, an optimization ladder ordered by measured ROI, FP8 correctness failure-mode catalog, cuBLAS/CUTLASS/tcgen05 backend selection, and agent-team patterns for multi-round optimization.
---

# B200 MoE Kernel Optimization

## Mission

Take a PyTorch-reference MoE implementation to a competitive B200 CUDA kernel by:

1. **Correctness before optimization** — cuBLAS FP16 is the oracle baseline
2. **ROI-ordered changes** — the biggest wins are outside GEMM internals; don't start with GEMM tuning
3. **Avoid known dead ends** — the catalog lists what regressed and by how much
4. **Agent-team discipline** — role separation, GPU isolation, anti-cheating rules

MoE-specific, B200-specific. Not a generic CUDA authoring guide.

## Skill dependencies

Load these companions before significant optimization work:

- **`cuda-b200-skill`** — Blackwell CUDA reference (TMA, WGMMA, tcgen05, warp specialization, PTX)
- **`ncu-cuda-profiling-skill`** — NCU metric selection, bottleneck attribution, run workflow

When this skill refers to a Blackwell primitive or NCU metric without full detail, the companion skill is the canonical reference.

---

## Before you start

Before proposing any change, read the following in full:

- `references/optimization-ladder.md` — the ordered list with measured gains and expected landmarks
- `references/dead-ends-catalog.md` — what has been tried and regressed; don't re-explore
- `references/pytorch-to-cuda-migration.md` — if starting from a PyTorch reference

Load on-demand when relevant:

- `references/hardware-constraints.md` — B200/SM100 facts and compile-target gotchas
- `references/fp8-correctness-modes.md` — debugging FP8 numerical failures (required reading whenever you see wrong output)
- `references/decision-trees.md` — dispatch logic (backend, tile, T-threshold, tcgen05 trigger)
- `references/agent-team-patterns.md` — team configs and role rules
- `references/tuning-knobs.md` — table of measured sweet spots and hard limits
- `references/non-gemm-optimizations.md` — routing / gather / scatter micro-ops

Before implementing any item from the ladder, read the matching file under `references/code-examples/`. The ladder rows explicitly call out when this is required.

## Core operating rules

1. **Correctness against cuBLAS FP16 oracle before anything else** — enable `CUBLAS_COMPUTE_32F_FAST_16BF` to activate BF16 tensor cores. Diff custom kernel outputs against this baseline.
2. **Measure before optimizing** — N ≥ 2 runs averaged; re-measure baseline periodically to catch drift; same GPU for baseline and variant. B200 run-to-run noise is ~3–4%.
3. **GPU selection** — unless the user names a device, agent picks an idle GPU via `nvidia-smi` and holds it for the full baseline+variant batch.
4. **No benchmark-specific shortcuts** — reject any gain depending on sample-specific caching or fixed shape shortcuts. If it depends on a legitimate deployable cache, explicitly quantify cold-path vs warm-path.
5. **Default to an agent team** — Planner / Implementer / Profiler (only the Implementer writes code). Use a single agent only for short focused debugging. See `references/agent-team-patterns.md`.
6. **Revert discipline** — when a named ladder item regresses, do **not** silently revert. Re-read the matching `code-examples/` file, confirm your implementation matches the pattern exactly, look for an orthogonal bug (pointer alignment, stale buffer) that might be the real cause, and re-apply after fixing. Every named ladder item has been confirmed to work on this hardware — first-try regressions usually mean an implementation bug, not a bad idea.
7. **Plateau = climb the ladder, don't tweak** — if speedup hasn't improved > 1% in 3 consecutive rounds, stop tweaking the current path. The next-untried ladder item is your move. If you're below 75× and haven't attempted zero-sync fast path or tcgen05, those are mandatory next steps.

---

## References

All detailed content lives under `references/`. Load the ones relevant to the current phase.

| File | When to load |
|---|---|
| [`references/hardware-constraints.md`](references/hardware-constraints.md) | FP8 format compatibility, cuBLAS FP8 path status, CUTLASS SM100 constraints, tcgen05 basics |
| [`references/pytorch-to-cuda-migration.md`](references/pytorch-to-cuda-migration.md) | First phase — moving from PyTorch reference to CUDA |
| [`references/optimization-ladder.md`](references/optimization-ladder.md) | Ordered optimization techniques with observed ROI and implementation notes |
| [`references/dead-ends-catalog.md`](references/dead-ends-catalog.md) | Documented failures with investigation context — check before trying anything here |
| [`references/fp8-correctness-modes.md`](references/fp8-correctness-modes.md) | 7 distinct FP8 numerical-correctness failure modes with debug recipes |
| [`references/decision-trees.md`](references/decision-trees.md) | FP8 backend selection, tile shape, T-dependent dispatch, when to move to tcgen05 |
| [`references/agent-team-patterns.md`](references/agent-team-patterns.md) | Team configurations, role rules, prompt templates, failure modes |
| [`references/tuning-knobs.md`](references/tuning-knobs.md) | Measured sweet-spot values for every tunable knob (tile, stages, launch bounds, compile flags, thresholds) |
| [`references/non-gemm-optimizations.md`](references/non-gemm-optimizations.md) | Routing, gather, SwiGLU, scatter micro-ops that cumulatively add ~5–8% after the main ladder is done |
| [`references/code-examples/`](references/code-examples/) | Working code snippets: JIT compile, dual-tile dispatch, zero-sync fast path, tcgen05 skeleton, dual TMA descriptors, T-dependent dispatch |

---

## Quick-reference: what to try first

For a fresh MoE optimization starting from a PyTorch reference:

1. **cuBLAS FP16 grouped GEMM baseline** with `COMPUTE_32F_FAST_16BF`. Pass correctness.
2. **NCU profile** — identify dominant kernel, occupancy, DRAM throughput.
3. **Eliminate host-GPU sync** — move routing metadata to GPU. *(+16% observed)*
4. **Swap to CUTLASS FP8 grouped GEMM** with blockwise epilogue if data uses 128-block float32 scales. *(+60% cumulative)*
5. **Dual-tile dispatch** for variable per-expert M. *(+13%)*
6. **Pipeline overlap** and metadata fusion after sync is clean. *(+5–6% each)*
7. **Only after all the above**: consider hand-written tcgen05 if GEMM is still > 60% of runtime and CUTLASS is at the occupancy wall. *(+3–5%)*
8. **Never**: start with tcgen05, fuse SwiGLU+FP8 quantize, or delete the FP16 gather kernel.

### Sanity landmarks during the climb

| State | Expected speedup range |
|---|---|
| Step 4 done (CUTLASS FP8 grouped GEMM) | ~43× |
| + Step 3 (zero-sync) | ~50× |
| + Step 5 (dual-tile) + static compile | ~57–65× |
| + Step 6 (pipeline + metadata fusion) | ~70–78× |
| + Step 7 (tcgen05 on both GEMMs) | ~85–90× |
| + non-GEMM micro-ops | ~90–93× |

If you're behind the landmark after "completing" an item, either the item isn't really in place (silent CUTLASS fallback, missing `sm_100a` compile target) or the implementation has a bug. Verify with NCU kernel names before moving on.

---

## Top observed gains (see `references/optimization-ladder.md` for full list)

| # | Technique | Typical gain |
|---|---|---|
| 1 | cuBLAS FP16 → CUTLASS FP8 grouped GEMM | **+60%** |
| 2 | Zero-sync fast path (metadata → GPU) | **+16.3%** |
| 3 | Dual-tile CUTLASS dispatch | **+13%** |
| 4 | Static compile + embedded CUTLASS | **+7.0%** |
| 5 | threadfence removal + metadata fusion | **+5.8%** |
| 6 | Pipeline overlap | **+5.7%** |
| 7 | tcgen05 on GEMM1 (hand-written PTX) | **+3.5%** |
| 8 | tcgen05 on GEMM1 + GEMM2 (dual TMA desc) | **+3.3%** |

## Top dead ends (see `references/dead-ends-catalog.md` for full list)

| Attempted | Observed |
|---|---|
| 2-SM CUTLASS cluster | **−18%** |
| 128×256×128 wide N-tile | **−60%** |
| Expert chunked pipeline | **−8.7%** |
| Fuse SwiGLU + FP8 quantize | Correctness failures on FP8 CUTLASS path |
| Delete FP16 gather kernel | 2× degradation |
| FP8 requantization to per-tensor | ~12.5% per-element error |
| `cublasGemmGroupedBatchedEx` FP8 | RUNTIME_ERROR |
| cuBLASLt BLK128×128 FP8 | `CUBLAS_STATUS_NOT_SUPPORTED` |
| CuTe DSL for MoE grouped GEMM | Correctness + alignment issues |
| CUDA graph | Unstable with dynamic shapes |

---

*Derived from production MoE kernel optimization work on NVIDIA B200 (observed 1× → ~91× stable over a reference MoE pipeline).*
