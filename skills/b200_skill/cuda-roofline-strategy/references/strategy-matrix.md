# The (position × phase) strategy matrix

Each cell lists techniques to try, ordered by typical payoff first. Cross-reference with the sub-topic files in `cuda-kernel-techniques/references/` for implementation detail.

---

## Bandwidth-bound

You're near the HBM ceiling. Every byte you don't load is free work.

### Early phase
- Coalesce global loads — all threads in a warp hit consecutive addresses. [memory-access]
- Vectorize loads (`float4` / `__ldg` 128-bit) — cut instruction count 4–8×. [memory-access]
- Pick a saner layout (transpose one operand if it makes both loads stride-1).
- Tile sizing first-pass — block sizes that make one tile fit in L1 / L2.

### Mid phase
- `cp.async` + double-buffered tiles — overlap load with compute. [memory-access]
- L2 swizzle / tile schedule — tiles reading shared KV land on the same SM. [memory-access]
- Bank-conflict padding on shared memory tiles. [data-placement]
- TMA (SM90+) / `cp.async.bulk` for large contiguous loads. [memory-access]

### Late phase
- Tune tile sizes and number of pipeline stages (3 vs 4 vs 5).
- Swap a smem staging buffer for direct global → register when it fits.
- Ensure `__restrict__` on all input pointers — lets the compiler overlap loads.

### Plateau
- Algorithmic: can you *load less*? This is where Split-K / FlashDecoding lives — instead of streaming all of K through one block, partition across blocks and merge. Pay more compute, consume less bandwidth per block. [parallelism]
- Can you recompute instead of storing intermediate? (e.g., FlashAttention's fused softmax).
- Lower precision of a hot tensor if correctness allows (bf16 partial output, fp8 weights). [numerical]

### Tactical sweep (after canonical techniques converge)

This is a distinct **phase**, not a loose "always try." When the canonical
technique stack (vectorization + shuffle reduction + register residency +
streaming pipeline) has been applied and you've hit 2–3 same-category reverts,
switch modes: run through this tactical checklist one item at a time. Each
item is a single experiment. Items are ordered by empirical hit rate.

1. **XOR swizzle or padding for shared-memory bank conflicts.** Profile for
   bank conflicts first (NCU "Shared Memory Bank Conflicts" metric); if any
   warp shows conflicts, add either `[M][N+1]` padding or a row-indexed XOR
   on the column. Observed 10–20% on matmul-style kernels. [data-placement]
2. **`__launch_bounds__` variants.** Try `(threads, 1)`, `(threads, 2)`,
   `(threads, 3)`, and compare ptxas register counts + achieved occupancy.
   Sometimes the default compiler choice is off by one block-per-SM. [occupancy]
3. **Right-size the register cache.** If you have a `reg[K]` array, confirm
   K is no larger than the reduction scheme requires. Halving is often
   possible and often doubles occupancy. [data-placement — and see its
   "validate each cache" counter-rule]
4. **`__ldg` on every streaming read site.** One-line change per load;
   10–30% on bandwidth-bound kernels near HBM peak. [occupancy]
5. **Block-schedule / L2 swizzle.** Reorder `blockIdx` → tile mapping so
   adjacent blocks share L2 residency. [memory-access]
6. **Per-workload block-size dispatch.** For variable-shape benchmarks, pick
   block size per input shape via the host-side launcher. [parallelism]

These are *not* substitutes for algorithmic/structural work — they're the
mechanical hygiene that's usually worth a combined 10–30% on bandwidth-bound
kernels and 5–15% on compute-bound ones. Do them in this order; after each,
measure and keep/revert normally. Do NOT collapse all six into one big commit
— the principled-rewrite exception is for algorithmic changes, not for tactical
tuning (which needs per-move attribution).

**When to switch from canonical phase to tactical sweep:** after the canonical
technique stack is applied AND the last 2–3 experiments were same-category
reverts (e.g., three variants of a register-cache change all reverted).
"Tactical sweep" is the different-category move that breaks you out of the
local-optimum loop.

---

## Compute-bound

You're near FLOPS peak. Doing less work or using the tensor cores are your levers.

### Early phase
- Tensor cores — if the op is matmul-shaped, WMMA / WGMMA is usually a 4–16× jump. [compute]
- Fuse elementwise ops into the epilogue (bias + activation + scaling in one pass).
- Avoid recomputing invariants in the inner loop. [compute]

### Mid phase
- Base-2 softmax (`exp2f` not `expf`) — lets the hardware SFU do the work in one instruction. [compute]
- Warp-level reduction (`__shfl`) for sums/maxes. [parallelism]
- Dot-product intrinsics / FMA explicit use.

### Late phase
- Unroll the inner loop (`#pragma unroll`) only as far as ILP gains exceed register pressure.
- Factor out common subexpressions in the inner loop.
- Audit PTX — sometimes the compiler inserts redundant casts.

### Plateau
- Reformulate the math. Online softmax instead of two-pass softmax. LSE in log2 base. Welford vs naive variance.
- Drop precision where you can tolerate it (`tf32`, `bf16` accumulate, `fp8` inputs with `fp32` accum).
- **Hand-write the next-gen tensor-core primitive when the library hits a hardware wall.** Library collectives (CUTLASS, cuBLAS) target portability and conservative register budgets; on a kernel that's *plateaued at the library's occupancy ceiling*, the move is a hand-rolled kernel using the architecture's newest MMA family (e.g. WGMMA on H100, **tcgen05 / UMMA on B200**) with a more aggressive accumulator-placement strategy (TMEM on B200). Risky — typically a 1–2 week effort — but unlocks the part of the roofline the library cannot reach. See `cuda-kernel-techniques/references/compute.md` "Tensor cores" + the `code-examples/tcgen05-kernel-skeleton.md` for the SM100 case.

### When to pull this trigger

Use **all** of these as the green-light test:
1. The library kernel is at the occupancy wall (achieved occupancy is dictated by per-block reg/smem; no other ladder item can lift it).
2. The kernel is > 60% of total iteration time (so any % gain on it matters).
3. All upstream non-GEMM optimisations are exhausted (zero-sync, pipeline overlap, scatter fusion, etc.).
4. You have a reference template (open-source kernel like gau-nernst `matmul_v7`, or a CUTLASS example you can hard-fork).

If any of (1)–(4) is missing, the expected-value of the hand-write is negative.

---

## Occupancy-limited

Too few warps in flight. The GPU is idle because there's nothing to run.

### Early phase
- `__launch_bounds__(threads, min_blocks_per_sm)` — forces the compiler to keep registers below the threshold. [occupancy]
- Smaller tile sizes — less smem, less registers per block.
- Sanity-check block size (typical sweet spot: 128, 192, or 256).

### Mid phase
- Profile with `--ptxas-options=-v` to see per-function register usage; spill hotspots into functions.
- Reduce live-ness of variables — don't hold tile A while computing tile B if you don't need it.
- Consider whether you're over-using shared memory as a staging area for small data; some fits in registers.

### Late phase
- Tune the two numbers in `__launch_bounds__` empirically — the sweet spot is often not the compiler's default.
- Swap a 256-thread block for two 128-thread blocks doing the same work.

### Plateau
- Rewrite data layout to shrink per-block smem (e.g., interleave instead of padding).
- Split a monolithic kernel into two smaller ones — the extra launch overhead is often paid back by doubled occupancy.

---

## Latency-bound

Enough warps, just stalled on their own dependencies.

### Early phase
- Software pipelining — overlap load(tile N+1) with compute(tile N) via `cp.async` + `__pipeline_wait_prior`. [memory-access]
- Unroll inner loop 2–4× to let ILP fill the stall. [compute]
- Hoist loop-invariant values out of the inner loop.

### Mid phase
- Add one more pipeline stage (2 → 3 → 4) if smem budget permits.
- Reorder inner-loop instructions to break dependency chains.
- Use warp-level shuffles instead of shared-memory round-trips for reductions. [parallelism]

### Late phase
- Minimize `__syncthreads()` — every sync flushes ILP. Merge sync points.
- Precompute offsets / masks on the host or in a warmup if they're loop-invariant.

### Plateau
- **Warp specialization**: dedicate some warps to loading, others to computing. Producer/consumer pipeline. This is how FlashAttention 3 and persistent kernels get their final wins.
- Consider a **persistent kernel** pattern — launch one block per SM and let each loop over tiles.

---

## Balanced

No dominant bottleneck. This is actually a hard regime — gains are small and come from attacking multiple fronts.

### Early phase
- Sweep block sizes (16, 32, 64, 128, 256, 384, 512) — balanced kernels are the most sensitive to this.
- Pick the 2nd-highest warp stall category and attack it as if it were the dominant one.

### Mid phase
- Incremental improvements across memory *and* compute; don't expect single big wins.
- Look for fusion opportunities with adjacent kernels in the pipeline. [compute]

### Late phase
- Micro-tune the kernel you have. Unroll factors, `launch_bounds`, swizzle patterns.
- Tune input-aware parameters (different block sizes for different shapes).

### Plateau
- This is where algorithmic changes pay the biggest. Historical example: DSA run v32 switched from 1-head/block to 4-heads/block and got +24% in one commit, breaking out of a plateau around 128x.
- "New architecture" moves: Split-K dimension, adaptive dispatch, heads-per-block, merge-kernel shape.

---

## Cross-cutting: sequencing rules

These rules cut across all five (position × phase) cells. They're about *which kernel in the pipeline to attack first*, not *what technique to use*.

### Fix host stalls and inter-kernel gaps before tuning the dominant kernel

When the wall-clock iteration contains both kernel compute and host/launch gaps, the single-kernel SOL number is misleading. A 50 µs `cudaStreamSynchronize` between routing and the "dominant" GEMM is 25% of a 200 µs iteration — and no amount of GEMM tuning recovers it.

Default ordering:

1. **Look at the NCU timeline first**, not just the per-kernel SOL section. Are there visible gaps between kernels? Host-side stalls (red bars on the host row)? Empty CTA windows?
2. **Kill host stalls** — move metadata construction to the GPU, eliminate `cudaStreamSynchronize` from the hot path, replace D2H/H2D round-trips with GPU-resident arguments.
3. **Chain dependent kernels** — `programmaticStreamSerialization` (PSS) for stream-level, `griddepcontrol` PTX for grid-level. See `cuda-kernel-techniques/references/parallelism.md` "Programmatic Dependent Launch".
4. **Right-size grids** — non-persistent kernels launching `gridDim = num_SMs × K` for non-existent work are pure scheduler overhead.
5. **Only then** turn to the dominant kernel's compute-vs-memory tuning.

**Field note (MoE).** The FuseMoE campaign measured **+16.3%** from eliminating host sync vs **+60%** from the GEMM backend swap. They aren't comparable — the +16.3% is achievable only because the +60% wasn't running yet. If you did the backend swap first, you'd see a +60% kernel-throughput win on top of a still-stalling pipeline, and the +16.3% would shrink (the host overhead is now a smaller fraction). Order matters because gains *compound multiplicatively from the unstalled baseline*, not additively from the original.

**Counter-example.** If the kernel pipeline is already zero-sync (no host stalls visible in NCU, no `cudaStreamSynchronize` in the hot path, kernels chained back-to-back), this rule is satisfied — go directly to dominant-kernel tuning. The rule is only load-bearing when host stalls exist *and* the agent is tempted to skip them.

### Plateau bias: planning under-proposes heavy lifts

At plateau (≥ 3 rounds of < 1% gain), the planner has a measured bias toward proposing another small tweak — block size, padding, launch-bounds variant — rather than the heavy lift the Plateau cell of this matrix actually calls for (hand-roll a next-gen tensor-core kernel, write a custom collective, restructure the pipeline). Small tweaks have positive, well-calibrated expected value; heavy lifts have high variance and are easy to over-estimate cost on. Both forces push toward "another tweak" even when the matrix is pointing at a heavy lift.

This bias *is the failure mode* documented as Pattern 1 in `cuda-kernel-techniques/references/operators/moe/manager-failures.md`. In the FuseMoE campaign, plateau-stage planning kept proposing CUTLASS scheduler tweaks instead of hand-rolling tcgen05 PTX — the run stalled near 80× until external direction broke the loop. The agent's plan and implementation are correct once the *direction* is chosen; the bias is in direction-selection, not in execution.

**Forcing function.** The countermeasure lives in `cuda-kernel-autodev/references/experiment-loop.md` Step 9: when in plateau phase, enumerate every untried heavy lift in this matrix's Plateau cell and the operator's measured ladder, and write an explicit reason for *not* picking each before falling back to a small tweak. Acceptable skip reasons are prerequisites (not done), missing reference templates, hard hardware constraints. "Looks expensive" or "uncertain payoff" are not — those are the bias itself talking. If no skip reason holds for a heavy lift, that lift is the next plan.

Important: this rule does not move plan or code authorship to a human. The agent still proposes the plan and writes the code. The rule only forces the agent to argue *against* heavy lifts before defaulting to easy ones, which is enough to defeat the bias in practice.

---

## Using the matrix — examples

### Example A: Experiment 4, bandwidth-bound

You're early. The current kernel is naive, HBM % is 80, compute % is 30. Don't reach for TMA yet.

Pick from **Bandwidth-bound × Early**: coalesce first, then vectorize, then tile. Three experiments' worth of roadmap. Don't jump to "plateau" techniques — you haven't earned them.

### Example B: Experiment 28, compute-bound

You're mid-to-late. Compute is at 72%, occupancy is good. You've already done tensor cores and base-2 softmax.

Pick from **Compute-bound × Late**: audit PTX, factor CSE, tune unroll. Small gains but principled.

### Example C: Experiment 40, plateau, balanced

Last 6 KEEPs were each under 2%. The top stall distribution is flat — no single pipe saturated. Classic plateau.

Pick from **Balanced × Plateau**: algorithmic change. Consult the user on whether a rewrite is in scope. Candidates: Split-K, persistent kernel, heads-per-block reorg, adaptive dispatch, new fusion. Offer 2–3 directions; let the user pick.

### Example D: The 2-token outlier

Aggregate is fine but the small-batch workload is at 6% occupancy. You're in **Occupancy-limited × (any phase)** for *that workload specifically*.

Move: adaptive dispatch. Make the kernel use Split-K for small batches (even though large batches don't need it). The per-workload refinement step in SKILL.md covers this pattern.

### Example E: The FuseMoE long-seq GEMM (Compute-bound × Plateau, MoE field note)

NCU on the slowest workload (seq_len = 14107): GEMM kernel is **77% of iter**, SM throughput **50.7%**, DRAM **24.9%**, **achieved occupancy 14.1%** capped by 168 reg/thread + 218 KB shared memory → 1 CTA/SM. Last 4 KEEPs on this kernel were all in the "tweak CUTLASS scheduler" category (each < 1%). Plateau, compute-bound, at the library's occupancy wall.

Move applied: hand-write a **tcgen05** persistent warp-spec kernel (TMEM accumulator, 7-stage pipeline, BM=BN=BK=128, 6 warps × 32 threads = 192 thread CTA). This is the "Plateau → Hand-write the next-gen primitive" bullet in the Compute-bound section above. Combined wins from tcgen05 on GEMM1 (+3.5%) and tcgen05 on both GEMMs with dual TMA descriptors (+3.3%) reached the speedup the campaign needed to ship. See `cuda-kernel-techniques/references/operators/moe/optimization-ladder.md` items 7–8.

The all-four-conditions test from the bullet above held: (1) at the occupancy wall; (2) GEMM > 60% of iter; (3) all non-GEMM ladder items done (zero-sync, dual-tile dispatch, pipeline overlap); (4) `gau-nernst/matmul_v7` was the reference template. Without all four, this move regresses or wastes weeks.
