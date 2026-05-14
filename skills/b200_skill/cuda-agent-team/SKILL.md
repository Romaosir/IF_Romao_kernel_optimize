---
name: cuda-agent-team
description: Multi-agent parallel CUDA kernel optimization. Use this skill when the user wants to run two (or more) optimization agents in parallel — typically on separate GPUs — to explore different optimization directions concurrently, or when the single-agent loop has plateaued and the user asks for parallel exploration. Also use when the user says things like "spawn agents on GPUs 6 and 7", "have two workers try different directions", or "Agent 1 focus on X, Agent 2 focus on Y". Extends `cuda-kernel-autodev` (which defines the single-agent loop) with orchestration protocol: isolated working directories, per-agent GPU pinning, sync/merge cadence, direction assignment, and main-agent role. Typically activated after you've already been running `cuda-kernel-autodev` for a while and the curve has flattened.
---

# CUDA agent team — parallel optimization orchestration

This skill extends the single-agent optimization loop (see `cuda-kernel-autodev`) into a parallel multi-agent setup: one orchestrator agent coordinates two or more worker agents running independent experiment loops on separate GPUs (or the same GPU in isolated environments).

Use this when:
- The single-agent loop has plateaued (≥5 small-gain KEEPs in a row, or ≥5 REVERTs).
- The user has multiple GPUs available and wants parallel exploration.
- There are two *clearly distinct* directions worth exploring simultaneously — e.g., "memory restructure" vs "compute restructure."
- The kernel is mature enough that workers can hypothesize on a stable baseline.

Do **not** use this when:
- The baseline isn't stable yet. Agents working on different broken starting points waste their runs.
- Only one GPU is available and you'd be time-slicing — parallelism is the whole point.
- The next experiment is obvious. Spawning two agents to run sequential steps is overhead, not parallelism.
- The agents would be evaluating on the same GPU with contention — their speedup numbers become unreliable.

---

## Roles

### Orchestrator (main agent — you)

Does **not** write kernel code. Responsibilities:

1. **Setup**: establish working directories, per-agent branches, eval scripts, GPU pinning. Snapshot the shared baseline.
2. **Direction assignment**: tell each worker what to focus on this round. Directions should be orthogonal enough that their results are independently useful.
3. **Progress monitoring**: pull each worker's latest `perf_log.md` summary. If a worker is stuck (3+ reverts) or silent, decide whether to reroute or sync.
4. **Synchronization decisions**: when one worker finds a big win, decide whether to sync the other worker onto the same baseline or let them finish their direction first.
5. **Merge**: when both workers have independent wins, produce a merged kernel that combines them, verify correctness, measure, and KEEP if better.
6. **Round boundaries**: decide when to call "end of round" — everyone sync, pick new directions.
7. **NCU / Roofline analysis** for the team — workers may not have bandwidth to re-profile deeply.

### Workers (subagents)

Each worker runs a `cuda-kernel-autodev` loop independently, constrained to:
- Its own working directory (e.g., `agent1_dir/`, `agent2_dir/`).
- Its own GPU via `CUDA_VISIBLE_DEVICES`.
- The direction assigned by the orchestrator.
- Its own `perf_log.md` (append-only, per-worker).
- Shared reference files (hardware guide, the reference kernel, the eval dataset) — read-only.

Workers should **not** communicate directly. Always route through the orchestrator. This keeps attribution clean and avoids silent coupling.

---

## Setup protocol

See `references/setup-protocol.md` for the full checklist. Summary:

1. **Pick an anchor commit** on the shared branch. Both workers start from this. Tag it: `git tag team-baseline-<date>`.
2. **Create per-worker directories** that contain a copy of (or symlink to) the needed files:
   ```
   agent1_dir/
     solution/cuda/kernel.cu     # worker 1's copy to edit
     solution/cuda/binding.py
     eval.sh                     # worker-1-specific eval script (sets CUDA_VISIBLE_DEVICES, output paths)
     notes/perf_log.md           # worker 1's log
     plan/
     profile/
   agent2_dir/  (same layout)
   ```
3. **Worker-specific eval scripts** must pin the GPU:
   ```bash
   #!/bin/bash
   # agent1_dir/eval.sh
   export CUDA_VISIBLE_DEVICES=6  # or whichever GPU is assigned
   cd agent1_dir && bash ../scripts/run_all_eval.sh
   ```
4. **Orchestrator workspace**: keep summaries, direction assignments, and merge notes in a top-level `team_notes/` directory.
   ```
   team_notes/
     round1_plan.md             # what each worker is doing this round
     round1_results.md          # end-of-round recap
     merge_log.md               # history of merges
   ```
5. **GPU isolation check**: run `nvidia-smi` to confirm the target GPUs are actually idle. Running workers on contended GPUs produces garbage speedup numbers.

---

## Round structure

A "round" is a period (typically 30–90 minutes of wall-clock, or until a natural stopping point) during which workers run their assigned directions, then the orchestrator syncs/merges.

```
Round N:
  1. Orchestrator assigns directions (team_notes/roundN_plan.md)
  2. Workers run their autonomous loop for the round duration
  3. Orchestrator pulls worker perf logs, summarizes
  4. Orchestrator decides:
     a. One worker wins outright → adopt that as the new baseline for round N+1
     b. Both workers have independent wins → merge, validate, adopt
     c. Nobody has a clear win → workers continue OR orchestrator assigns new directions
  5. Tag the new baseline: git tag team-baseline-rN
  6. Go to round N+1 until stopping condition
```

### Direction assignment heuristics

Good direction pairs are orthogonal — a worker's win can be combined with another worker's win. Examples from the DSA V5 run:

| Round | Worker 1 direction | Worker 2 direction |
|---|---|---|
| R1 | Software pipelining / `__ldg` / index prefetch | Softmax optimization / batch max |
| R2 | Architectural: 4-heads/block, `cp.async` | L2 locality: block remap |
| R3 | Split-K tuning | Hybrid merge kernel |
| R4 | Bank-conflict XOR swizzle | Templated SPLIT_K |

Bad direction pairs produce conflicts (both modify the same smem layout) or are redundant (both tune block size).

**When the workers agree a direction is promising** but can't both explore it without overlap — let one worker do it, assign the other to something adjacent. Don't waste parallelism on duplication.

---

## Synchronization decisions

After each round, the orchestrator picks one:

### Option A — One-sided win

Worker 1 got a +20% KEEP. Worker 2's experiments were flat.

- Adopt worker 1's kernel as the new baseline.
- `cp agent1_dir/solution/cuda/kernel.cu agent2_dir/solution/cuda/kernel.cu`
- Commit: `git commit -m "team: adopt agent1 v<N>, +20%"`
- Tag: `git tag team-baseline-r<N>`
- For round N+1, worker 2 starts fresh from the new baseline.

### Option B — Both have wins on different fronts

Worker 1 added `cp.async`. Worker 2 changed the softmax formulation. Both +10% independently.

**Merge**:
1. Start from the shared pre-round baseline.
2. Apply worker 1's diff (cleanly).
3. Apply worker 2's diff (may need manual conflict resolution).
4. Compile, run full eval.
5. If merged is correct AND speedup ≥ either worker alone, KEEP merged as new baseline.
6. If merged regresses below both workers, something is interacting badly — investigate. Often one of the changes assumed the old state of the other.
7. If merged only matches the better of the two, the changes don't compose — adopt the winner.

See `references/merge-protocol.md` for the detailed checklist.

### Option C — Nobody has a clear win

Both workers reverted 3+ times or had only noise-level improvements.

- Orchestrator reads both perf logs. Are they stuck on similar blockers?
- Issue the online-search rule team-wide: both workers web-search for new ideas, return findings.
- Orchestrator picks two *fresh* directions based on the search results.
- If this happens twice in a row, consider that the kernel is near its ceiling — shift to submission mode.

### Option D — One worker has a win, other worker's direction is still promising

Worker 1 found +15%. Worker 2's experiments are flat but the direction hasn't been fully explored yet.

- Adopt worker 1's win as the new baseline for worker 1.
- Let worker 2 continue on the old baseline for one more round — keep exploring the direction.
- Sync at the end of the next round.

This lets you avoid forcing worker 2 to abandon a direction mid-exploration.

---

## When to end the team phase

End the team phase and collapse back to single-agent (or submit) when:

- **Clear winner emerges**: one kernel dominates and the other direction is exhausted. Kill the losing worker, run single-agent on the winner.
- **Convergence**: 3+ rounds with no KEEP bigger than 2%. You're in team-wide plateau; the submission is probably the best you'll get.
- **User says to wrap up**.

Always leave a clean final state: one kernel on the shared branch, all per-worker directories tagged so they can be revived if needed, `team_notes/summary.md` explaining what was tried.

---

## Common failure modes

- **Unbalanced GPUs** — worker 1 on B200 and worker 2 on A100. Speedups aren't comparable. Always run team on matched GPUs.
- **Worker 1 silently affects worker 2** — shared file edited, shared workspace reused. Enforce the per-agent-dir isolation strictly.
- **Duplicate work** — both workers converge on the same idea independently. The orchestrator should notice and reassign one.
- **Merge conflicts that hide bugs** — a 3-way merge that compiles doesn't mean the changes compose correctly. Run the full eval after every merge, not just correctness.
- **Orchestrator does the work itself** — the orchestrator's job is coordination, not coding. If you find yourself writing kernel changes in the orchestrator role, that's a sign something went wrong with worker assignment.
- **No clear direction** — workers floundering because the orchestrator didn't give a focused direction. Be specific: "Worker 1: explore XOR swizzle for `k_tile`; do not touch block size" is better than "Worker 1: reduce bank conflicts."

---

## References

- `references/setup-protocol.md` — initial directory setup, GPU pinning, per-worker eval scripts
- `references/merge-protocol.md` — how to combine independent wins into a merged kernel
