# MoE manager-mode failure patterns

This file documents three specific "agent gets stuck" patterns observed in the FuseMoE optimisation campaign, and the human-in-the-loop interventions that unblocked them. **For the general multi-agent protocol** (roles, GPU pinning, merge cadence), see the sibling skill [`cuda-agent-team`](../../../../cuda-agent-team/SKILL.md). This file is the operator-specific addendum.

The three patterns combined unblocked > 30 percentage points of speedup in the MoE campaign that the agent team would otherwise not have reached.

---

## Pattern 1 — Agents prefer the easy direction

**Symptom.** Given a plateau, the planner keeps proposing another small tweak (block size, padding, launch bounds) rather than picking up the heavy lift (hand-roll tcgen05 PTX, write a new CUTLASS collective).

**Why it happens.** The expected-value of a small tweak is positive and well-calibrated; the expected-value of a heavy lift is high-variance and easy to over-estimate cost on. Risk-averse plan generation defaults to the small tweak.

**Where the generic countermeasure lives.** This file is the *field evidence*; the generic rule and its forcing function now live in:
- `cuda-roofline-strategy/references/strategy-matrix.md` "Plateau bias: planning under-proposes heavy lifts"
- `cuda-kernel-autodev/references/experiment-loop.md` Step 9 "Counter the easy-direction bias"

If the agent reads either of those before each plan at plateau, it self-corrects without a human nudge — which is the intended state.

**Magnitude.** Before the forcing function was codified, the FuseMoE run stalled around 80×. After a *human-issued* direction nudge ("stop tweaking; go look at tcgen05 — read the gau-nernst matmul_v7 source and adapt it for FP8 blockwise + grouped semantics"), the agent autonomously researched references, wrote the PTX-level kernel, debugged the multi-barrier pipeline, and reached 90+×. The same self-correction is now what the forcing function above is designed to trigger — the agent enumerates heavy lifts and forces itself to argue against each before defaulting to a tweak.

**When the human still has to intervene.** Even with the forcing function, two situations leave a human role: (a) the heavy-lift menu is empty in the agent's current view (no reference template surfaced by web search) — the human supplies the URL or the example; (b) the agent enumerates heavy lifts but rationalises away each — usually a sign the agent's "uncertain payoff" reasoning needs to be challenged externally. The intervention is still about *direction*, not plan or code.

**When to apply the forcing function.** After ≥ 3 consecutive rounds of < 1% gain on the same kind of change. (Same trigger as the "tactical sweep" in `cuda-roofline-strategy/strategy-matrix.md`, but the move here is "switch optimisation *class*", not "switch within a class".)

---

## Pattern 2 — Agents revert good ideas after spurious failures

**Symptom.** A correct, named optimisation lands and then regresses an apparently-unrelated workload. The agent attributes the regression to the new code and reverts. Days later the same optimisation re-appears in the plan because nothing in the catalog warned against it.

**Why it happens.** The agent's hypothesis-test loop is sound (KEEP/REVERT on speedup), but it doesn't *differentiate* between "this optimisation is wrong" and "this optimisation is fine; there's an unrelated bug". On the FuseMoE campaign this happened twice with the zero-sync fast path: the optimisation was correct, but an orthogonal pointer-alignment bug surfaced because zero-sync triggered an allocator code path. The agent reverted the +16.3% zero-sync win instead of fixing the alignment bug.

**Fix.** Human watches for: a *named* ladder item regresses → first hypothesis is *orthogonal bug*, not "the named item is wrong". Re-read the matching `code-examples/` file. Compare implementation to the reference. Only revert if the implementation diverges from the spec. We codified this as a "revert discipline" rule.

**Magnitude.** +16.3% recovered on the first hit (zero-sync), several other smaller items on subsequent hits.

**Detection signal.** Look at the perf-log: if the same optimisation appears in `plan_v*.md` more than twice and gets KEEP'd then REVERT'd, the regressor is almost certainly orthogonal.

---

## Pattern 3 — Agents under-search the literature

**Symptom.** The agent's web search misses a key reference (CUTLASS Blackwell example, an open-source matmul template, an obscure NVIDIA blog post). Without the reference, the agent produces a strictly-worse implementation from first principles.

**Why it happens.** Search ranking on these topics is weak; the right reference is often three pages deep in GitHub README hierarchy. Generic LLMs without browse-and-summarise loops fall back on their training-time corpus, which under-represents the latest CUDA tooling.

**Fix.** Human injects the URL. Concretely on the FuseMoE campaign, two interventions of this kind: pointing the agent at `https://github.com/NVIDIA/cutlass/tree/main/examples/75_blackwell_grouped_gemm/` and at `gau-nernst/matmul_v7` source. Once each URL was in context, the implementation followed correctly.

**Magnitude.** Hard to quantify directly — without the references the campaign would have produced a less-optimal CUTLASS configuration and very likely never reached tcgen05.

**Note.** This is also the trigger condition for `cuda-roofline-strategy`'s "3-revert online-search rule" — if the agent has reverted three times in the same category, force a `WebSearch` / `WebFetch` round before the next experiment.

---

## Generalisation note for other operators

These patterns are not unique to MoE — they generalise to any long-horizon kernel optimisation. DSA-attention and DSA-topk runs are likely to see analogous failures (e.g. for DSA-attn: agents avoid writing custom WGMMA kernels and stick to CUTLASS examples). When porting this skill to a new operator, expect to observe these three patterns again, and write operator-specific intervention notes here.

See also: [`cuda-agent-team/references/merge-protocol.md`](../../../../cuda-agent-team/references/merge-protocol.md) for how parallel-worker results merge back; this file complements that with "what to do when one worker is stuck".
