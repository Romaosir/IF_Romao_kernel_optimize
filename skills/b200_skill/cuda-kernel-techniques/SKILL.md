---
name: cuda-kernel-techniques
description: Catalog of proven CUDA kernel optimization techniques, broken by sub-topic (memory access, data placement, parallelism, compute, control flow, occupancy, numerical, anti-patterns). Use this skill whenever you need to look up *how* a specific technique works or *when* it applies — e.g., "how does Split-K FlashDecoding work", "when should I use cp.async vs TMA", "how do I implement online softmax", "what is register-centric data path", "what bank-conflict padding should I use", "when does warp specialization help", or before committing to a technique suggested by `cuda-roofline-strategy`. The main SKILL.md is an index; each sub-topic reference has the implementation details, when-it-helps, when-it-hurts, and code sketches. Each technique includes hardware-general guidance with B200 (SM100) and older-arch examples where they differ.
---

# CUDA Kernel Optimization Techniques — Catalog

This is a reference skill, organized by sub-topic. The SKILL.md is deliberately short — its job is to route you to the right sub-topic file. Go there for the actual content.

When you read a technique file, you will find, for each technique:
- **What it does** — one-paragraph description
- **When it helps** — the bottleneck / roofline position / hardware it fits
- **When it hurts** — the failure modes and anti-indications
- **Code sketch** — minimal CUDA or PTX snippet (or link to CUTLASS / cuda-samples)
- **Hardware notes** — SM80 / SM90 / SM100 differences if relevant
- **Field notes** — observations from real runs (e.g., "contributed +24% in one step during V5 of the DSA run")

---

## What to read first, by bottleneck (problem → sub-topic)

If you have an NCU profile in hand, jump directly to the sub-topic that addresses the dominant bottleneck. (For the full position × phase decision, use `cuda-roofline-strategy/references/strategy-matrix.md` — the table below is the fast lookup.)

| If NCU shows… | First read | Then look at |
|---|---|---|
| DRAM > 70%, Long Scoreboard top stall | `memory-access.md` (vectorize, `cp.async`, double buffer, TMA) | `data-placement.md` (shmem tiling, bank conflicts) |
| SM throughput > 70%, low DRAM | `compute.md` (tensor cores, online softmax, FMA intrinsics, fusion) | `control-flow.md` (branchless, predication) |
| Occupancy < 30% with low SM and DRAM | `occupancy.md` (`__launch_bounds__`, register pressure, smem budget) | `data-placement.md` (register-centric data path) |
| Occupancy > 60%, throughputs middling, Short Scoreboard / Wait stalls | `parallelism.md` (warp specialization, persistent kernels, shuffles) | `compute.md` (ILP, unroll) |
| Bank conflicts present | `data-placement.md` (XOR swizzle, padding) | `anti-patterns.md` |
| Variable-shape benchmark, some workloads catastrophic | `parallelism.md` (adaptive dispatch, Split-K, right-sizing grid) | strategy-matrix Step 4 (per-workload refinement) |
| Recurrent state operator (token-by-token state evolution) | `data-placement.md` (register-centric state) + `parallelism.md` (V-split / row-decomposition / chunked) | `compute.md` (WMMA on chunk-internal matmuls) |
| FP8 / mixed precision suspicions | `numerical.md` (precision strategies, online rescaling) + `operators/moe/fp8-correctness-modes.md` | `hardware-b200.md` (FP8 path status, scale formats) |
| Anything regressed unexpectedly | `anti-patterns.md` (check before debugging deeper) | the sub-topic of the technique you just added |

**A regression search-path note**: if you just landed a regression, the *first* file to read is `anti-patterns.md` — many regressions are reinventions of known dead-ends (excess `__syncthreads`, premature atomics, TMA on scattered access, over-unrolling, JIT-cost on hot path, etc.). Read that before opening a debugger.

---

## Sub-topics

| File | Covers |
|---|---|
| [`references/memory-access.md`](references/memory-access.md) | Coalescing, vectorization (`float4`, `__ldg`), `cp.async` + double buffering, TMA / `cp.async.bulk`, L2 cache locality & tile swizzle, `__restrict__` |
| [`references/data-placement.md`](references/data-placement.md) | Register-centric data path, shared memory tiling, bank-conflict avoidance (padding, swizzle), cluster-shared memory (SM90+) |
| [`references/parallelism.md`](references/parallelism.md) | Split-K / FlashDecoding, adaptive dispatch, multi-heads-per-block, block/grid sizing, persistent kernels, warp-level shuffles |
| [`references/compute.md`](references/compute.md) | Tensor cores (WMMA / WGMMA / MMA), online softmax, base-2 softmax via `exp2f`, FMA + dot-product intrinsics, operator fusion, `#pragma unroll` |
| [`references/control-flow.md`](references/control-flow.md) | Branchless masking (`-INFINITY` trick), predication, warp divergence avoidance, hoisting invariants |
| [`references/occupancy.md`](references/occupancy.md) | `__launch_bounds__`, register pressure management, smem budget, cached function attributes, static workspace |
| [`references/numerical.md`](references/numerical.md) | Lower-precision strategies (bf16 partial output, fp8), accumulator precision, LSE in log2 base, online softmax rescaling |
| [`references/anti-patterns.md`](references/anti-patterns.md) | 20+ things that *sound* clever but usually regress: excess `__syncthreads`, premature atomics, TMA without need, WMMA on scattered access, over-unrolling, etc. |
| [`references/hardware-b200.md`](references/hardware-b200.md) | B200 / SM100 hardware reference: compute capability flags, memory hierarchy, cuBLAS FP8 path status, tcgen05 PTX reference, CUTLASS SM100 tile constraints, measurement noise |

## Operator-specific findings

The catalog above is hardware-general. When you're optimizing a specific operator, the operator-specific subfolder collects the measured findings, dispatch rules, and code patterns that proved out on real runs:

| Folder | Operator | What's there |
|---|---|---|
| [`references/operators/moe/`](references/operators/moe/) | Fused MoE (DeepSeek-style FP8 blockwise) | Pipeline structure, measured optimization ladder with %-gains, FP8 correctness failure modes, B200 tuning-knob sweet spots, dead-end catalog, manager-failure modes |
| `references/operators/attention/` | DSA sparse attention | *(placeholder — DSA-attn agent fills this)* |
| `references/operators/topk/` | DSA top-K indexer | *(placeholder — DSA-topk agent fills this)* |

Operator-specific files reference back into the catalog, e.g. an MoE ladder item like "dual-tile dispatch" points at [`parallelism.md`](references/parallelism.md) for the generic mechanics. Each catalog technique that has measured operator evidence ends with a **Field notes** section naming the operator, the run, and the magnitude. Add new field notes when you finish a campaign.

### Code patterns

Reusable code sketches that are too long to inline in a technique file live under [`references/code-examples/`](references/code-examples/). Generic ones (CUTLASS JIT compile, tcgen05 persistent warp-spec skeleton, zero-sync fast path) are usable across operators; operator-specific ones are prefixed (e.g. `moe-dual-tile-dispatch.md`).

---

## How to use this catalog

### During planning (Phase B step 9 of `cuda-kernel-autodev`)

1. `cuda-roofline-strategy` told you which tier to try (e.g., "Bandwidth-bound × Mid → `cp.async` + double buffer").
2. Open `references/memory-access.md`, jump to the `cp.async` section.
3. Read "when it helps" and "when it hurts." If your scenario matches the hurts column, pick a different technique in the same tier.
4. Use the code sketch as a starting point. Adapt to your kernel's shapes.

### During debugging

If your experiment regressed and you don't know why, check `references/anti-patterns.md` — the specific pattern is often there. Then look at the technique file for the thing you *added* and re-read the "when it hurts" section.

### During onboarding a new operator

Read the sub-topics in this order for a balanced picture:
1. `memory-access.md` (usually the first bottleneck)
2. `data-placement.md` (usually the second)
3. `parallelism.md` (algorithmic leverage)
4. `compute.md` (later-phase wins)
5. `occupancy.md` (when the first three don't explain the low perf)
6. `control-flow.md` + `numerical.md` (late-phase tuning)
7. `anti-patterns.md` (mandatory — saves you from reinventing dead ends)

Then check `references/operators/<your-operator>/` if a folder exists — the measured evidence there will be much more specific than the catalog, and the catalog field notes give you the cross-references back.

### When you finish an optimization campaign

For each *named* technique you used that already lives in the catalog: append a one-paragraph field note at the bottom of the relevant technique file (3–6 lines: which technique, what % gain, on which operator/run, what caused it to win or to revert). This is how the catalog stays useful for the next team.

---

## Hardware coverage note

Each technique is described in a hardware-general way, with notes where the behavior diverges across SM80 (A100), SM89 (Ada), SM90 (H100/H200), and SM100 (B100/B200). If you're on a different arch (consumer RTX, Grace Hopper, older Volta/Turing), the general principles hold but specific thresholds may differ — confirm via a quick run or the NVIDIA Programming Guide appendix for your compute capability.

When a technique requires a specific SM version, the file says so clearly (e.g., "TMA requires SM90+"). Don't try to port those to older arch — you'll hit undefined behavior or compile errors.

---

## What's *not* in this catalog

- **Framework-specific details** (FlashInfer / KernelBench / PyTorch custom op conventions) — those belong in your repo's docs.
- **Triton-specific techniques** (`tl.dot`, `num_stages`, `@triton.autotune`) — this skill is CUDA C++ focused. A separate Triton skill would be appropriate if the user asks.
- **cuBLAS / cuDNN / CUTLASS usage guides** — when those libraries are the right answer, use them; this catalog is for when you're writing custom kernels.
- **Numerical-accuracy debugging** for production training — covered partially in `numerical.md` but not in depth.
