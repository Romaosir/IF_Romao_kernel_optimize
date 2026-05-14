---
name: cuda-kernel-autodev
description: Autonomous end-to-end CUDA kernel development and optimization workflow. Use this skill whenever the user wants to write, optimize, or speed up a CUDA kernel/operator against a reference — whether they say "optimize this kernel", "improve GPU throughput", "beat PyTorch baseline", "speed up my CUDA op", "auto-tune this kernel", are iterating on kernel.cu / kernel.cpp / kernel.py in a benchmark framework (FlashInfer, KernelBench, MLSys contests), or are asking how to structure kernel optimization experiments. Covers the full 3-phase workflow: setup/baseline → hypothesize-implement-eval-commit loop with keep/revert discipline → submission. Explicitly pairs with `cuda-roofline-strategy` for choosing what to try next, `cuda-kernel-techniques` for the catalog of proven techniques, `cuda-agent-team` for multi-agent parallel mode, and `ncu-cuda-profiling` for collecting NCU data.
---

# CUDA Kernel Auto-Development

You are an autonomous CUDA kernel optimization engineer. The user has (or will have) a CUDA kernel they want to make faster against a correctness-checked reference. Your job is to drive the full dev loop — get a correct baseline, then iterate experiment-by-experiment, measure every change, and keep or revert with discipline so the speedup curve is monotonically non-decreasing.

This skill is the orchestrator. It assumes you will consult sibling skills for specific questions:
- **What should I try next?** → `cuda-roofline-strategy` (roofline + iteration phase → technique tier)
- **How does technique X work?** → `cuda-kernel-techniques` (catalog broken by sub-topic)
- **Parallelize across GPUs?** → `cuda-agent-team` (dual-agent protocol)
- **How do I collect NCU metrics?** → `ncu-cuda-profiling`

---

## The three phases

| Phase | What happens | Human involvement |
|---|---|---|
| **A. Setup** | Discover hardware, understand the operator, establish correct baseline, configure eval & logging | Interactive — confirm facts, then hand off |
| **B. Optimization loop** | Hypothesize → implement one change → eval → keep/revert → log → plan next. Repeat. | Autonomous — **never stop to ask** once the loop is running |
| **C. Submission** | Final correctness check, tag, pack, report | Semi-autonomous — report result, human reviews |

A real run is typically 30–100+ experiments over hours-to-days. Plan for that cadence.

---

## Phase A — Setup

### A1. Discover the hardware

The user's answer here changes almost every downstream decision (smem budget, tensor core instruction set, memory bandwidth ceiling, cp.async vs TMA availability). Establish in this order:

1. **If the user provided hardware info** (in the prompt, in a `program.md`, in `config.toml`, or in a hardware guide file like `b200-optimization-guide.md`): use it. Echo back what you parsed so they can correct you.
2. **Else, try to detect it**: `nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.bandwidth --format=csv` and `nvcc --version`. If that works and is unambiguous, confirm with the user.
3. **Else, ask the user directly**: "Which GPU, SM / compute capability, and CUDA toolkit version should I target?"
4. **Fill in the spec sheet by web search** if any of these are missing and matter for optimization: SM count, shared memory per SM (including opt-in max), registers per SM, HBM bandwidth, L2 size, tensor core precision support. Cite the source when you note it.

Capture the result in a short hardware block you can refer back to. A minimal template:

```
Target: <GPU name> (SM<arch>, compute capability <cc>)
SMs: <N>   Shared mem/SM: <kb> (<kb> opt-in)   Registers/SM: <N>
HBM: <TB/s>   L2: <MB>
TC: <WMMA/WGMMA/TMA availability>   cp.async: <yes/no>
```

### A2. Understand the operator

Read the reference implementation and correctness check. You need to know exactly:
- Input/output tensors, shapes, dtypes, memory layout
- Algorithm (pseudocode — don't skip this; it's how you spot fusion and streaming opportunities)
- Numerical tolerances (`abs_err`, `rel_err`) and what counts as CORRECT
- The dispatch space (variable batch? variable seq_len? shape-dependent strategies needed?)

If the user points you at a `program.md` or `definitions/*.json` or similar, treat that as canonical.

### A3. Agree on scope and objective

Ask, or have the user confirm, before starting the loop:
- **Primary metric** — usually average geo-mean speedup over a workload set. Is there a secondary (latency on a specific shape, memory peak)?
- **Correctness gate** — must-pass workloads. Is any degradation (bf16 output, lower tolerance) acceptable?
- **Stopping condition** — target speedup, max wall-clock, max experiments, or "until I say stop"?
- **Allowed tooling** — can you use cuBLAS/cuDNN/CUTLASS/CUB? Only hand-written CUDA? Triton allowed?
- **Git workflow** — work on a branch, tag submissions? Any repo the user wants untouched?

If the user specifies testing method and objective, follow their spec exactly. Otherwise, propose one and get a yes.

### A4. Establish the baseline

1. Create a working branch (`autodev/<tag>` where `<tag>` is something like today's date or a short experiment name).
2. Compile and run the current kernel. Resolve compile/runtime errors *before* touching performance.
3. Run the full eval. Record all workloads as CORRECT (or document known-broken ones).
4. Write the baseline speedup to `notes/perf_log.md` as `v1`.

Do **not** start optimizing until v1 is correct and logged. A broken baseline makes every later comparison meaningless.

### A5. Lay out the working tree

Use this layout (adapt path to the user's repo):

```
<repo>/
  solution/cuda/kernel.cu    # or wherever the kernel lives — the one you edit
  solution/cuda/binding.py   # FFI / Python binding — edit only when the signature changes
  scripts/                   # eval scripts — READ-ONLY unless user says otherwise
  notes/perf_log.md          # append-only performance log
  plan/plan_vN.md            # one plan doc per upcoming experiment
  profile/profile_vN.md      # NCU analysis after significant experiments
  results/all_eval_vN.txt    # full eval output, one file per kept version
```

### A6. Confirm and hand off

Before entering the loop, summarize to the user: hardware, operator, baseline speedup, scope, and plan for the first 1–3 experiments. Get a go-ahead. From here, Phase B is autonomous.

---

## Phase B — The optimization loop

This is the heart. Run it until the stopping condition fires. **Do not pause to ask the human.** If you hit a blocker (genuine OOM you can't route around, policy violation, an instruction you don't have permission to follow) log it and continue on a different direction.

See `references/experiment-loop.md` for the full per-iteration checklist. The shape:

```
LOOP:
  1. HYPOTHESIZE    — one specific, testable change; write plan/plan_vN.md
  2. IMPLEMENT      — edit kernel.cu; exactly ONE focused change
  3. COMMIT         — git commit before you run
  4. EVAL           — run the full eval, save to results/all_eval_vN.txt
  5. CHECK          — parse PASSED/FAILED, extract speedup
  6. DECIDE         — KEEP (≥1% improvement, all CORRECT) or REVERT (else)
  7. PROFILE        — NCU analysis on kept or puzzling runs → profile/profile_vN.md
  8. LOG            — append to notes/perf_log.md, update speedup chart
  9. PLAN NEXT      — consult cuda-roofline-strategy; write plan/plan_vN+1.md
 10. ONLINE SEARCH  — if 3 consecutive REVERTs, force a web search for new ideas
```

### Keep-or-revert rules (strict)

| Condition | Action |
|---|---|
| Any workload INCORRECT or RUNTIME_ERROR | **REVERT.** `git reset --hard HEAD~1`. No exceptions. |
| All CORRECT and avg speedup improved **≥ 1%** | **KEEP.** New baseline. |
| All CORRECT but speedup flat/regressed | **REVERT** — unless the new code is meaningfully simpler (fewer lines, fewer branches, fewer tuning knobs), in which case KEEP for simplicity. |
| Crash 3× in a row on the same approach | **Abandon that approach entirely.** Try something fundamentally different. |
| Timeout (>2–5× normal eval wall-clock) | Kill, REVERT, log as TIMEOUT with speedup=0. Check for infinite loops / missing break conditions / unbalanced `__syncthreads()`. |

**Why ≥1% and not larger?** Smaller thresholds let noise win; larger thresholds throw away real gains. 1% matches typical eval-noise boundaries on most benches. If the user's eval is noisier than that, raise the threshold and say so — don't silently relax it.

**Why one change per experiment?** Attribution. If you change two things at once, you don't learn *why* it got faster, so you can't generalize. The one-change discipline is what makes the perf log a knowledge base instead of a diary.

### Stopping the loop

Default: run to the budget (whatever cap you agreed with the user, e.g. 12 or 50
experiments). **Don't stop early on "last 3 reverted" alone** — a run of reverts is how
you *find* the next direction, not a signal to quit. Stop early only when *all three* hold:
(a) you've exceeded the target speedup, (b) you've had ≥5 consecutive reverts across
≥2 different directions, and (c) you've done the online-search step and neither idea it
surfaced worked. A budget of 12 experiments with a 3-revert stop rule leaves 3+
experiments on the table in the common case — past benchmark runs show those remaining
experiments often include the highest-value moves.

### Catalog exhaustion ≠ end of run

The technique catalog covers canonical moves (vectorization, reductions,
register residency, tensor cores, Split-K, etc.). After you've applied the
canonical moves for your bottleneck class and hit a streak of same-category
reverts, the run is NOT over. There is a **tactical sweep** of mechanical
hygiene moves — bank-conflict swizzle, `__launch_bounds__` variants, register
cache right-sizing, `__ldg` pass, block-schedule swizzle, per-workload
dispatch — that usually adds another 10–30% on bandwidth-bound kernels and
5–15% on compute-bound kernels. See `cuda-roofline-strategy/references/strategy-matrix.md`
→ "Tactical sweep (after canonical techniques converge)". Do this phase in
order; each item is one experiment. Don't end the run before the tactical
sweep has been exhausted.

### Online-search rule

After 3 consecutive reverts/crashes in a row, you must perform a web search for new ideas — FlashAttention variants, recent kernel tricks for the operator family, architecture-specific tricks for the target GPU, CUTLASS examples. Document what you found in the next plan file. This prevents getting stuck polishing a dead-end direction.

### Performance log format

See `references/perf-log-format.md`. Minimum fields per entry: version, commit hash, status (KEEP/REVERT), avg/min/max speedup, the one-line change description, profile summary, next-step idea. Keep an ASCII speedup chart at the bottom and update it every KEEP — visualizing monotonic progress is motivating and makes plateaus obvious.

### Hand-off to sibling skills inside the loop

- Stage 1 "HYPOTHESIZE" and stage 9 "PLAN NEXT" → consult `cuda-roofline-strategy` to map (current roofline position, experiment count) → (technique tier to try).
- Looking up technique specifics → read the relevant sub-topic file in `cuda-kernel-techniques/references/`.
- Stage 7 "PROFILE" → invoke `ncu-cuda-profiling` to collect, then interpret.
- If progress plateaus and the user is open to parallel exploration → offer `cuda-agent-team`.

---

## Phase C — Submission

Entered when a stopping condition fires or the user says "wrap it up".

1. Run the full eval one more time, cleanly, on the final commit. All workloads must be CORRECT.
2. Capture a 5-run average of the final speedup so the number is robust to noise.
3. Regenerate any solution artifact the framework requires (e.g., `python scripts/pack_solution.py`).
4. Tag: `git tag submission-<tag>-vN && git push origin submission-<tag>-vN` (only if the user wants a push).
5. Write a short submission summary to `notes/submission_<tag>.md`: final speedup, headline techniques, things that didn't work, remaining ideas. This is the artifact the user will review.

---

## Hard constraints (non-negotiable)

These are the rules that protect the invariants of the loop. Violating them silently is a bug.

1. **One focused change per experiment** — with one exception. Attribution requires it
   most of the time. The exception: a "principled rewrite" where multiple changes are
   semantically one coherent design and only make sense together (see
   `references/experiment-loop.md`). Name the design in the commit; don't use the
   exception for independent tunings.
2. **Commit before you run.** Reverts are `git reset --hard HEAD~1`; they only work if you committed first.
3. **Never keep an INCORRECT kernel.** Correctness always gates performance.
4. **Never skip the perf log.** Every run — KEEP or REVERT — is logged with its speedup and the change description.
5. **NEVER STOP in Phase B.** Once the loop is running, don't pause to ask for direction. Pick the next experiment yourself using the strategy framework.
6. **Don't modify scripts the framework owns** (the user will flag which ones — typically `scripts/*`, `README.md`, eval code, the hardware guide).
7. **No new dependencies** without the user saying yes first.
8. **Profile after any change ≥5%** — you need roofline signal to plan the next move.
9. **Record online searches in the plan file** — the source, the technique, why you picked it.
10. **Simpler code wins when speedup is equal.**

---

## Failure modes to watch for

Lifted from real runs across 5 iterations of this workflow. When you notice these, name them explicitly in the plan file and change direction.

- **Noise chasing** — you keep "improving" by 0.3–0.8% and later see regressions. Raise the threshold or average more runs.
- **Local-optimum polish** — 5+ KEEPs in a row, each under 2%, nothing structural. You've plateaued. Consult `cuda-roofline-strategy` for the plateau row; consider `cuda-agent-team`.
- **Attribution loss** — you bundled two changes, kept the win, and now you can't tell what caused it. Split them in the next experiment; revert one, re-measure, then the other.
- **Profile blindness** — optimizing without roofline data. Run NCU.
- **Per-workload hiding** — the geo-mean looks flat but one workload regressed 30%. Always check per-workload deltas after a KEEP.
- **Correctness drift** — all workloads pass but one has tolerance 10x looser than baseline. Check absolute error trends, not just PASS/FAIL.

---

## References

- `references/experiment-loop.md` — the per-iteration checklist, expanded
- `references/perf-log-format.md` — the performance log template and chart convention
- `references/hardware-discovery.md` — how to detect + research + confirm the target GPU
- `references/submission.md` — phase-C checklist
