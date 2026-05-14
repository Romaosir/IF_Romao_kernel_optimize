# Agent Team Patterns

Multi-round kernel optimization benefits from an agent team with clear roles, rules, and a measurement protocol. The configurations below were used successfully in production MoE optimization work. Patterns that didn't work are also documented.

## When a team helps vs. hurts

| Use a team when | Use a single agent when |
|---|---|
| Exploring multiple hypotheses in parallel | Tight iteration on one well-defined approach |
| Multi-round optimization (many iterations expected) | Short focused debugging session |
| Work has clean plan / implement / profile stages | Single well-defined change to make |
| Multiple isolated GPUs available | Only one GPU available |

Team overhead is real — role handoffs, synchronization, and merge phases cost time. If the task is small, a single agent is faster.

---

## Team configurations observed

| Configuration | Best for | Notes |
|---|---|---|
| 3-role (Planner + Implementer + Profiler) | Multi-round optimization with clear handoffs | Most common configuration; balance of discipline and speed |
| 4-role (add Evaluator) | Correctness-critical phases | Separate agent owns correctness verification |
| 2-agent parallel | A/B comparing two directions | Each agent on isolated GPU, compare results at sync point |
| 4-agent parallel | Broad exploration across hypotheses | Highest overhead; only justified for initial brainstorming phases |
| Single agent + team-internal subagents | Fine-grained sequential work | One driver agent spawns short-lived helpers |

### When each configuration breaks down

- **3-role** gets stuck if the Profiler isn't rigorous — silent CUTLASS fallback has slipped past Profilers repeatedly in practice, causing bad data to drive bad decisions
- **4-agent parallel** suffers from GPU contention noise (see failure modes below)
- **Single agent** misses opportunities for clean role separation on long runs

---

## Rules that make teams effective

### 1. Role separation

Only the **Implementer** agent modifies kernel source. Every other agent operates read-only on the code.

**Why:** Prevents race conditions on code state; forces clean Plan → Implement → Profile handoff; makes every change reviewable.

**How to apply:** Put this in the initial prompt. The Planner proposes changes in text; the Profiler reports findings in text; only the Implementer touches files.

### 2. GPU selection

Unless the user names a device, the agent autonomously picks an idle GPU before any performance measurement:

- Query `nvidia-smi` for utilization and free memory
- Pick a device with 0% utilization and sufficient free memory for the kernel
- Hold that device for the full baseline + variant measurement batch
- Do not switch devices mid-measurement (comparisons become invalid)

**With parallel multi-agent runs:** each agent claims a distinct idle GPU and reports which one. Contention destroys measurement fidelity — see failure mode below.

### 3. Anti-cheating rule

From the initial prompt:

> Reject any branch that depends on disallowed sample-specific caching or benchmark-specific shortcuts. If a gain depends on a legitimate deployable cache, quantify cold-path vs warm-path.

**Why:** Agents will find "gains" that depend on the evaluation harness's structure (same inputs repeated, fixed tensor shapes, deterministic seeds) and propose them as real optimizations. These don't survive deployment.

**How to apply:** State the rule at team setup. When reviewing a proposed change, explicitly ask "does this depend on sample-specific caching?" If unclear, run the change with inputs permuted and check it still wins.

**Legitimate deployable cache exception:** Warmup-time weight format conversion is OK if explicitly framed as cold-path vs warm-path. Quantify both.

### 4. Baseline re-measurement

- Test each kernel change **N ≥ 2 runs**, average the speedup
- Periodically re-test the baseline (every few rounds, or after any major refactor)
- Use the same GPU for baseline and variant

**Why:** B200 run-to-run noise is ~3–4%. Single-run comparison is unreliable. Baseline drifts silently as cumulative changes accrue.

**Pattern:** When a proposed change shows < 3% gain, re-run both baseline and variant three times before deciding. Many "small wins" are noise.

### 5. Snapshot contract

Every round saves:
- A code snapshot with the measured speedup in the filename (e.g., `kernel_R<round>_<speedup>x.cu`)
- A short summary document describing: what changed, hypothesis, measurement, result, decision (keep / revert / investigate further)

**Why:** Enables regression traceback — when performance drops, bisect through snapshots. Enables cross-round comparison — compare code A at 54× against code B at 67× side-by-side.

**File naming convention:** include round number AND measured speedup. Both are needed — round number alone forces a lookup, speedup alone can collide.

---

## Failure modes observed

| Failure | Symptom | Mitigation |
|---|---|---|
| **GPU contention noise** | 4 agents on 4 GPUs show ±30% variance | Isolate to one GPU for final eval; use multi-agent only for independent exploration phases |
| **Silent fallback not detected** | "Gains" come from unintended code path (e.g., CUTLASS fell back to cuBLAS) | Profiler must verify intended kernel name dispatched via NCU |
| **Baseline drift** | Cumulative changes cause hidden regressions | Periodic baseline re-test mandatory every few rounds |
| **Code-state confusion** | "Where is your newest code?" | Snapshot naming with round + speedup suffix resolves this |
| **Multi-agent divergence** | Two agents optimize to different local maxima | Sync point + merge phase at the end; don't let them run independently to the end |
| **Agent in wrong direction** | Wasted optimization rounds on unpromising path | Explicit course-correction prompt (ship-vs-explore signal) |
| **Implementer makes breaking refactor mid-round** | Correctness test fails for reason unrelated to the change being measured | Require Implementer to commit changes incrementally; don't bundle refactor + optimization in one round |
| **Planner proposes from stale information** | Plan references kernel that was removed two rounds ago | Planner reads current state before each plan; doesn't work from memory |

### GPU contention: the detail

With 4 agents on 4 physical GPUs on the same host, measurement noise jumped from ±3–4% to **±30%**. Root cause: host-side resource contention (CPU for launching kernels, PCIe bandwidth, memory bandwidth in multi-tenant setups).

**Fix:** For final measurement, use a single agent on a single idle GPU. Use parallel agents only for *independent exploration* where measurement precision doesn't matter, then consolidate the winning approach onto a single agent for definitive measurement.

### Silent CUTLASS fallback: the detail

In practice, agents have repeatedly reported speedups that turned out to be measuring cuBLAS FP16, not CUTLASS FP8, because the CUTLASS dispatch had silently failed and fallen back. This gets caught only when the Profiler checks kernel names in NCU output.

**Fix:** After any CUTLASS change, the Profiler runs an NCU profile and verifies the intended kernel name appears in the timeline. If it's missing (e.g., you see `cublas*` where `...Blockwise1SmSm100...` should be), the optimization didn't actually take effect.

---

## Recommended initial prompt template

```
Goal: <specific, measurable — e.g., "optimize MoE kernel to break Xx baseline
       using CUTLASS FP8 grouped GEMM">

Team:
  agent1 (Planner): research, propose plan, no code changes
  agent2 (Implementer): kernel source changes only; uses cuda-b200-skill
  agent3 (Profiler): NCU analysis, no code changes; uses ncu-cuda-profiling-skill
  # optional:
  # agent4 (Evaluator): correctness verification; no code changes

Rules:
  - Only agent2 modifies kernel source
  - Before any performance measurement, pick an idle GPU via nvidia-smi
    (unless the user specified a device). Hold that device for the full
    baseline + variant measurement batch.
  - Test N ≥ 2 runs on each change, compute avg speedup, periodically retest
    baseline
  - Save every round's code snapshot with the round number and measured
    speedup in the filename (e.g., kernel_R<round>_<speedup>x.cu); save a short
    summary describing change + hypothesis + result + decision
  - Reject any optimization depending on sample-specific caching or
    benchmark shortcuts. If a gain depends on a legitimate deployable
    cache, quantify cold-path vs warm-path.
  - After any CUTLASS change, Profiler verifies the intended kernel name
    dispatches (no silent fallback to cuBLAS)
  - Don't stop until I stop you (esc)
```

### Adapting the template

- **Just correctness work (bug hunt, not optimization):** remove the measurement/snapshot rules; keep role separation and GPU selection
- **Exploration phase (no shipping target):** replace measurement rule with "report trends, don't require N≥2 precision"
- **Production shipping phase:** tighten measurement to N ≥ 3, require cold-path and warm-path measurements separately

### Rules the template intentionally does not include

- **Specific GPU device IDs** — let the agent pick
- **Specific file paths** — path conventions belong to the project, not this skill
- **Framework-specific evaluation commands** — those belong to the project's benchmark harness, not this skill

---

## Prompts that work mid-run (post-setup corrections)

These are verbatim patterns that have produced useful course-corrections in practice:

| Purpose | Prompt |
|---|---|
| Continue with unchanged plan | "continue 50 rounds, and no stop" |
| Sanity check baseline measurement | "Before starting, measure the baseline 3× on an idle GPU and report the variance" |
| Break optimistic mindset | "the baseline code have used the fp8 gemm, review carefully" |
| Introduce anti-cheating mid-round | "there is another rule: Reject any branch that depends on disallowed sample-specific caching or benchmark-specific shortcuts. If a gain depends on a legitimate deployable cache, quantify cold-path vs warm-path." |
| Shift direction (explore→ship) | "what is your best solution, how can I run it?" |
| Shift direction (ship→explore) | "try the DeepGEMM approach or implement a new CUTLASS version. If still not working, search on the web to find some idea and move on." |
| Cross-reference another direction | "<other_team>'s best result was N×, look at their approach and see if anything applies here" |

---

## What to log per round

Each round, the Implementer should produce (or the Planner should capture):

1. **Hypothesis** — what change is being tested, and why it should help
2. **Code delta** — the actual change as a diff or short code snippet (not the full kernel)
3. **Measurement protocol** — GPU device, N runs, baseline retest status
4. **Result** — speedup (old → new), PASS/FAIL on correctness
5. **Decision** — keep, revert, or investigate further (with reason)
6. **NCU snippet** (from Profiler, for significant changes) — dominant kernel, occupancy, DRAM throughput, any silent-fallback check

Short is fine. Consistency matters more than length. A one-paragraph summary per round compounds into a useful history.

---

## When the team gets stuck

If the team has run N rounds with no stable gain:

1. **Re-measure the current baseline rigorously.** Noise may be hiding a small gain, or the baseline may have drifted upward.
2. **Check the dead-ends catalog.** Has the team already re-tried something documented as a dead end?
3. **Pull back to the optimization ladder.** Are earlier items (1–6) actually done? If the team skipped ahead, go back.
4. **Run NCU on the dominant kernel.** What metric is saturated? What isn't?
5. **Consider the floor.** Some workloads are at 82× and the kernel is fundamentally memory-bound at that point. Further gain requires tcgen05 or architectural changes, not tuning.
6. **Stop.** Not every optimization converges. A stable 78–82× that survives scrutiny is better than an unstable 93× that only shows up on specific runs.
