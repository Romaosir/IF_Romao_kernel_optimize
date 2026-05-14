# MoE (FuseMoE / DeepSeek FP8 blockwise) — operator-specific findings

This folder holds the **measured** evidence from the FuseMoE optimisation campaign on B200. The catalogue files in `references/*.md` give the *generic* techniques; this folder records *what proved out on this specific operator*.

## Files

| File | Contents |
|---|---|
| [`pipeline.md`](pipeline.md) | The MoE pipeline shape: routing → counting → gather → GEMM1 → SwiGLU+quantise → GEMM2 → pull-scatter. Plus the Blackwell features the kernel actually uses (UMMA / TMEM / TMA / mbarrier / PSS / persistent warp specialisation). |
| [`optimization-ladder.md`](optimization-ladder.md) | 21 measured optimisations ordered by observed ROI on the all-19 average — from cuBLAS FP16 baseline (~26.8×) through CUTLASS FP8 (~43×), zero-sync fast path (+16.3%), tcgen05 (+6.8%), and the final non-GEMM micro-opts that push to ~92×. |
| [`dispatch-decisions.md`](dispatch-decisions.md) | Six runtime dispatch trees — FP8 backend selection, tile shape, T-dependent GEMM2 backend, when to move from CUTLASS to tcgen05, scale-argument setup, when to add PSS. Each threshold is the measured sweet spot. |
| [`fp8-correctness-modes.md`](fp8-correctness-modes.md) | Seven distinct FP8 `INCORRECT_NUMERICAL` failure modes — what triggers each, debug recipe, workaround. The shared root cause is B200 wanting MXFP8 scaling vs the dataset's 128-block float32 scaling. |
| [`tuning-knobs.md`](tuning-knobs.md) | Measured sweet-spot values for every tunable on the B200 MoE kernel — CUTLASS tile shapes, tcgen05 stages, launch-bounds, bucket thresholds, compile flags. With the regression edge of each. |
| [`dead-ends.md`](dead-ends.md) | Catalogue of measured regressions — what was tried, what the delta was, what the root cause turned out to be. Check this before retrying anything that matches. |
| [`manager-failures.md`](manager-failures.md) | Three "agent gets stuck" patterns observed during the campaign and the human-in-the-loop interventions that unblocked them. |

## Code patterns

The MoE-specific code patterns live in `../../code-examples/` with the `moe-` prefix:

- `moe-dual-tile-dispatch.md` — dual-tile dispatch by `max_M` (the +13% lever)
- `moe-dual-tma-descriptors.md` — two TMA descriptor pairs (GEMM1 vs GEMM2)
- `moe-t-dependent-dispatch.md` — backend selection by total-token count T

Generic patterns reusable across operators (also in `code-examples/`):

- `cutlass-jit-compile.md`, `tcgen05-kernel-skeleton.md`, `zero-sync-fast-path.md`

## How this folder cross-references the generic catalogue

Each MoE finding refers back to a generic technique by name. Example trail:

- `optimization-ladder.md` says "dual-tile dispatch, +13%" → points at `parallelism.md` "Adaptive dispatch" → which has an MoE field-note pointing back here.
- `fp8-correctness-modes.md` documents the bridge from B200's native MXFP8 to the dataset's 128-block float32 scaling → that's a `numerical.md` field-note topic.

If you're adding evidence from a new MoE run, the convention is: update the operator-specific file here (e.g. add a row to `optimization-ladder.md`), *and* append a one-paragraph **Field note (MoE).** to the relevant catalogue file (e.g. `parallelism.md`). That keeps both surfaces useful.

**Timing of these updates: end-of-campaign, NOT mid-campaign.** This catalogue exists to serve the *next* campaign on this operator. Updating it while the current campaign still has untried heavy lifts on the menu is one of the failure modes documented in `cuda-kernel-autodev/references/experiment-loop.md` Step 12 ("wrap-up mode mid-campaign"). If the current speedup curve is still climbing, do not break out of the experiment loop to update this file — keep running experiments. The catalogue update is for when the campaign ends (user-signalled stop, or plateau-bias forcing function exhausted with all heavy lifts skipped for hardware-or-prerequisite reasons).

### Cross-trail of measured items → catalogue field notes (current)

| Ladder item | Catalogue Field note location |
|---|---|
| #1 cuBLAS FP16 → CUTLASS FP8 grouped GEMM | `numerical.md` FP8 failure-mode table, `hardware-b200.md` cuBLAS FP8 status |
| #2 Zero-sync fast path | `parallelism.md` "Fast-path / slow-path", `code-examples/zero-sync-fast-path.md` |
| #3 Dual-tile dispatch | `parallelism.md` "Adaptive dispatch", `code-examples/moe-dual-tile-dispatch.md` |
| #7–8 tcgen05 on GEMM1/2 | `compute.md` UMMA field note, `hardware-b200.md` tcgen05 section |
| #9 Dual prep kernel | `compute.md` "fuse next-iteration setup into current-iteration tail" |
| #13 Warp-parallel routing scores | `parallelism.md` "Warp-level shuffle reductions" Field note (MoE) |
| #14 Routing intermediates in regs | `data-placement.md` "Register-centric data path" Field note (MoE) |
| #15 Memset fusion into pull_scatter tail | `compute.md` "fuse next-iteration setup" |
| #16–17 PSS / griddepcontrol | `parallelism.md` "Programmatic Dependent Launch" |
| #18 Skip cuBLAS allocs on fast path | `anti-patterns.md` "Allocating a cuBLAS handle on the hot path" |
| #19 Routing sync-barrier removal | `anti-patterns.md` "Redundant `__syncthreads` whose invariant…" |
| #20 SwiGLU 8 rows per block | `occupancy.md` "Block size sweet spots" Field note (MoE) |
| #21 Grid tightening to actual work | `parallelism.md` "Right-sizing grid" Field note (MoE) |

## Cross-skill dependencies

- For the workflow / experiment-loop / perf-log format: see [`cuda-kernel-autodev`](../../../../cuda-kernel-autodev/SKILL.md)
- For roofline-driven "what to try next": see [`cuda-roofline-strategy`](../../../../cuda-roofline-strategy/SKILL.md)
- For multi-agent orchestration on multiple GPUs: see [`cuda-agent-team`](../../../../cuda-agent-team/SKILL.md)
