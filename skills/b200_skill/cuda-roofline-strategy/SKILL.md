---
name: cuda-roofline-strategy
description: Roofline-driven strategy selection for CUDA kernel optimization. Use this skill during the "what do I try next" moment in a kernel optimization loop — when the user has a profile (NCU report or similar) and needs to decide which class of technique to try, or when experiments are plateauing and the user asks "what now?", or when diagnosing whether a kernel is compute-bound / bandwidth-bound / occupancy-limited / latency-limited. Maps (roofline position, iteration phase) → technique tier so the next experiment has a principled justification rather than being a guess. Complements `cuda-kernel-autodev` (which drives the loop) and `cuda-kernel-techniques` (which details each technique).
---

# Roofline-driven strategy selection

The roofline model asks: is this kernel limited by **compute throughput**, **memory bandwidth**, or **occupancy / latency**? The answer changes which optimizations will help and which will be wasted effort.

This skill turns a profile into a decision: *given the current bottleneck and how far we are into the run, which tier of techniques should we try next?*

Use it at two points in the loop from `cuda-kernel-autodev`:
1. **After every profile** (step 7 of the experiment loop) to classify the current kernel.
2. **When planning the next experiment** (step 9) to pick a technique tier matching (position, phase).

---

## Step 1 — Classify the current roofline position

From an NCU report (see `ncu-cuda-profiling` for collection), read four numbers:

| Metric (NCU section) | What you're looking at |
|---|---|
| **Compute throughput** (`sm__throughput.avg.pct_of_peak_sustained_elapsed` or Speed-of-Light compute %) | % of SM peak FLOPS achieved |
| **Memory throughput** (`dram__throughput.avg.pct_of_peak_sustained_elapsed` or SOL DRAM %) | % of HBM peak bandwidth achieved |
| **Achieved occupancy** (`sm__warps_active.avg.pct_of_peak_sustained_active`) | Active warps / max warps |
| **Top warp stall reason** (Warp State Statistics) | What the warps are waiting on |

Classify using this table:

| Compute % | Memory % | Occupancy | Classification | What this means |
|---|---|---|---|---|
| > 70% | any | any | **Compute-bound** | You're near the FLOPS ceiling. Future wins come from doing less work or using tensor cores. |
| any | > 70% | any | **Bandwidth-bound** | You're near the HBM ceiling. Future wins come from loading less data or raising arithmetic intensity. |
| < 40% | < 40% | < 40% | **Occupancy-limited** | Too few warps in flight. Reduce register pressure, reduce smem usage, or raise block count. |
| < 40% | < 40% | > 60% | **Latency-bound** | Enough warps, but they're stalled on dependencies. Software pipelining, async copies, warp specialization. |
| 40–70% | 40–70% | any | **Balanced** | No single bottleneck. Future wins are small and come from parallel attack on multiple fronts. |

Top stall reason refines the call:
- **"Long scoreboard"** → global memory latency → add pipelining / `cp.async` / prefetch
- **"Short scoreboard"** → shared memory / register dependency → rework inner loop, ILP
- **"Wait"** → barrier (`__syncthreads`) → reduce sync frequency or rebalance loads
- **"Not selected"** → kernel is being issued, just not by this warp → usually means occupancy is enough; check other stalls
- **"MIO throttle"** → memory I/O pipe saturated → vectorize loads, reduce memory traffic
- **"Tex throttle"** → texture/LDG pipe saturated → vectorize, use ldg.128

Write the classification into `profile/profile_vN.md`:

```
Classification: bandwidth-bound
  compute=52%  dram=84%  occupancy=34%  top_stall=Long Scoreboard (38%)
  Note: achieved 6.7/8.0 TB/s HBM; L2 hit rate 24% — surprisingly low.
  Implication: next experiment should either reduce load volume (vectorize,
  compress, reuse from smem/L2) or increase arithmetic intensity per byte.
```

See `references/analysis-steps.md` for the full diagnostic walkthrough.

---

## Step 2 — Locate yourself in the iteration phase

The same bottleneck calls for different responses depending on where you are in the run. An "aggressive rewrite" move at experiment 5 is natural; at experiment 40 it's probably desperation.

| Phase | Typical experiment count | Default attitude |
|---|---|---|
| **Early** | 0–10 | Aggressive — try large structural changes, different algorithms, different block-size regions |
| **Mid** | 10–30 | Focused — systematic sweeps over a few promising parameters, one at a time |
| **Late** | 30+ | Incremental — fine-tuning, combining previously-successful techniques |
| **Plateau** | 5+ consecutive KEEPs under 2% each, or 5+ consecutive REVERTs | Radical — fundamental rethink, likely a new direction |

"Plateau" overrides the experiment-count rule. If you're on experiment 42 and just broke out of a plateau with a 24% jump from an architectural rewrite (like V5 v32 in the DSA run), treat yourself as "early" again w.r.t. the new architecture — you just opened new ground to explore.

---

## Step 3 — Pick a tier from the strategy matrix

Cross the classification (rows) with the phase (columns) to get the technique tier. See `references/strategy-matrix.md` for the full matrix with rationale; this is the summary:

| Position \ Phase | Early | Mid | Late | Plateau |
|---|---|---|---|---|
| **Bandwidth-bound** | Restructure memory access (layout, vectorization, coalesce) | Async copies + double-buffer; L2 reuse | Fine-tune tile sizes / stages | Fundamentally reduce data loaded (compression, recomputation, algorithmic change like Split-K) |
| **Compute-bound** | Tensor cores if applicable; algorithmic simplification | Instruction mix tuning; hoist invariants | Micro-tune inner loop, unroll | Try lower-precision algorithm or a different math formulation |
| **Occupancy-limited** | Reduce register pressure (`__launch_bounds__`, smaller tiles) | Smaller smem footprint; split across more blocks | Tune `launch_bounds` numbers | Rewrite data layout to cut register/smem needs |
| **Latency-bound** | Software pipeline + prefetch | Warp-level reorg, more stages | Unroll + ILP | Warp specialization or producer/consumer model |
| **Balanced** | Sweep block sizes widely | Attack the 2nd-highest stall reason | Micro-tune + combine prior wins | Algorithmic change (the 3–5× "new architecture" move) |

Then pick a specific technique from the chosen tier in `cuda-kernel-techniques`.

---

## Step 4 — Per-workload refinement

When you're in the Late or Plateau phase of a benchmark with multiple workloads, the aggregate metric hides per-workload pathologies. Example (from the DSA run): average speedup looked flat, but the 2-token workload had 6.21% occupancy — a full order of magnitude below the 8-token workload. Fixing that one case bumped the geo-mean.

When to do this:
- Aggregate speedup plateaus but variance across workloads is high
- The min-speedup workload is materially worse than the rest
- Iteration count > 30 and you've exhausted the "treat all workloads the same" ideas

How:
1. Sort workloads by current speedup. Identify outliers (the bottom 10–20%).
2. Run NCU on just those workloads (NCU supports `--launch-skip` and `--launch-count` for targeting specific launches).
3. Classify each outlier separately using the table above. They often have a *different* bottleneck than the happy path.
4. Consider workload-aware dispatch: a single kernel that uses different code paths based on input shape (e.g., single-pass vs Split-K, different SPLIT_K values per token count, different block sizes).

The "adaptive dispatch" pattern is how a kernel crosses from "fast on the common case" to "fast on all cases."

---

## Step 5 — Sanity-check before running the experiment

Before you commit and run, mentally check:
- **Is this in the right tier for my (position, phase)?** If not, why am I overriding?
- **Have I already tried this?** Scan `notes/perf_log.md` — if you reverted this technique at v12, don't redo it at v25 without naming what's different.
- **What's the predicted magnitude?** If the tier usually gives 5–15% and my hypothesis promises 40%, one of them is wrong.
- **What would disprove the hypothesis?** (From the Phase B checklist.)

If any answer is uncomfortable, spend 5 more minutes sharpening the plan before running.

---

## Escape hatches

The matrix isn't a cage. Override it when:

- **The user has domain knowledge.** If they say "split-K is the move here, skip ahead," believe them.
- **A recent KEEP changed the picture.** Re-profile and reclassify. The whole matrix resets.
- **An online search (per the 3-revert rule) surfaces a technique that matches the bottleneck class**, even if it's from a different tier. Cite the source in the plan file.

The goal isn't matrix adherence — it's making the *why* of each experiment explicit and checkable.

---

## References

- `references/analysis-steps.md` — walking through an NCU report to a classification
- `references/strategy-matrix.md` — full (position, phase) × tier matrix with rationale and worked examples
