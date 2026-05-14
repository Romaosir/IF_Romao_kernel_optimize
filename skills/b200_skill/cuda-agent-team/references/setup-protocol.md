# Setup protocol — team mode

Before you spawn any workers, set up the environment so the runs are isolated and comparable.

## 1. Confirm the baseline is stable

Team mode inherits the starting state. If the starting state is broken, both workers waste their runs.

- Checkout the branch you want the team to work from.
- `bash scripts/eval_solution.sh` — all workloads must be CORRECT.
- Record the baseline speedup; every worker's "delta" is measured against this.
- Tag: `git tag team-baseline-<tag>-r0`.

## 2. Confirm GPU availability

```bash
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv
```

You need:
- At least as many visible GPUs as workers.
- Matched architecture and SKU across workers. Running worker 1 on a B200 and worker 2 on an A100 makes the speedups non-comparable.
- GPUs idle (< 5% utilization, > 90% memory free). Running on a contended GPU makes speedup numbers unreliable.

If GPUs are busy, wait or ask the user. Don't just start — the data will be bad.

## 3. Per-worker working directories

Each worker edits its own copy of the solution files:

```bash
# From the repo root
mkdir -p agent1_dir agent2_dir
cp -r solution/ agent1_dir/solution/
cp -r solution/ agent2_dir/solution/
mkdir -p agent1_dir/{notes,plan,profile,results} agent2_dir/{notes,plan,profile,results}
```

Some files stay shared and read-only — reference kernel, eval dataset, hardware guide, `scripts/*`. Workers must not edit those.

## 4. Per-worker eval scripts

Each worker needs an eval script that (a) pins the correct GPU and (b) reads the worker's own kernel.

```bash
# agent1_dir/eval.sh
#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=6   # or whichever GPU this worker owns
cd "$(dirname "$0")"            # agent1_dir/
# Point the eval at the local solution dir
bash ../scripts/run_all_eval.sh --solution ./solution > eval_output.txt 2>&1
# Print summary
grep -E "PASSED|FAILED|speedup|CORRECT|INCORRECT|error" eval_output.txt
```

Adapt the invocation to whatever the framework expects. The two non-negotiables:
- `CUDA_VISIBLE_DEVICES` pins the GPU.
- The solution path points at the worker's own copy.

## 5. Worker git workflow

Each worker needs to commit changes somewhere but shouldn't interfere with the other's history. Options:

**Option A — per-worker branches**:
```bash
# Worker 1
git checkout -b team/agent1/<tag>

# Worker 2
git checkout -b team/agent2/<tag>
```
The orchestrator merges from each into `team/<tag>` at round boundaries.

**Option B — commits under per-worker directories only**:
```bash
# Worker 1 commits
git add agent1_dir/solution/
git commit -m "a1 vN: <hypothesis>"
```
The orchestrator can cherry-pick or copy files across directories as needed.

Option B is simpler for small teams (2 workers). Option A scales better but has more merge overhead. Default to B.

## 6. Launching workers

Via the `Agent` tool with `subagent_type` set to a coding-capable agent:

```
Agent 1 prompt:
  You are worker 1 of a CUDA kernel optimization team. Read:
  - ../team_notes/round1_plan.md  (your direction for this round)
  - ./notes/perf_log.md           (your own history)
  - ../notes/perf_log.md           (shared baseline log)

  Loop per `cuda-kernel-autodev`'s Phase B. Only edit files under `agent1_dir/`.
  Use `./eval.sh` for evaluation. Commit to agent1_dir on each experiment.

  STOP after <N> experiments or <X> minutes, whichever comes first.
  Report: speedup at end, your best version, brief notes.

Agent 2 prompt:
  (Same structure, different directory and direction.)
```

Spawn them in parallel (same tool-use turn, multiple Agent calls) so they run concurrently.

## 7. Orchestrator state

Keep these files current as the rounds progress:

- `team_notes/roundN_plan.md` — directions assigned this round
- `team_notes/roundN_results.md` — per-worker summaries at round end
- `team_notes/merge_log.md` — running log of merges and their outcomes
- `team_notes/current_baseline.md` — which tag is the current shared baseline

When spawning the next round, the workers read `current_baseline.md` + `roundN_plan.md` to know where to start and what to do.

## 8. Failure mode: workers silently share state

Watch for:
- Workers accidentally writing to `../solution/` (outside their own dir).
- Shared cache files (`__pycache__`, build artifacts) causing non-determinism.
- Shared temp directories for NCU reports clobbering each other.

Fix: ensure every write path is inside the worker's own directory. Review each worker's eval script for paths leaking outside.
