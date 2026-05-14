# Merge protocol

When both workers produce independent wins, you want to combine them. The output is either: (a) a merged kernel that beats both, (b) the single better one (when changes don't compose), or (c) a regression (investigate).

## When to merge

Only merge when:
- Both workers have KEEP-grade results (all CORRECT, ≥1% over their own baseline at the start of the round).
- The two changes touch largely different parts of the kernel — otherwise it's just picking one.
- The hypotheses are compatible on their face (e.g., "add `cp.async` pipelining" and "switch softmax to `exp2f`" don't conflict).

Don't merge when:
- Both changes touch the same code section deeply (e.g., both rewrote the softmax loop differently). Pick one.
- Worker 2's change depends on a state worker 1 eliminated. Pick the more principled direction.

## The merge procedure

### 1. Identify the base

The shared baseline from the start of this round. E.g., `team-baseline-r3`.

### 2. Capture the two diffs

```bash
# Worker 1's diff (from the round start to worker 1's best commit)
git diff team-baseline-r3 team/agent1/<tag> -- agent1_dir/solution/ > /tmp/a1.patch

# Worker 2's diff
git diff team-baseline-r3 team/agent2/<tag> -- agent2_dir/solution/ > /tmp/a2.patch
```

If you're using the "commits under per-worker directories" pattern, edit the patches to reference the merged file path (since both workers' patches refer to their own directory; the merged kernel will be in the shared `solution/` dir).

### 3. Apply to a merge branch

```bash
git checkout -b team/merge-r3 team-baseline-r3
git apply /tmp/a1.patch  # may need --directory=solution or similar
git apply /tmp/a2.patch
```

If patch 2 fails to apply cleanly, there's a textual conflict. Inspect and resolve manually — but be careful: a clean textual resolution doesn't mean semantic compatibility.

### 4. Compile

```bash
# Build the solution
bash scripts/build.sh  # or whatever the framework uses
```

If compilation fails, the changes are incompatible at the type / signature level. Usually means one worker renamed or resignatured something the other depended on. Pick one.

### 5. Correctness check

```bash
bash scripts/eval_solution.sh > merge_eval.txt 2>&1
grep -E "CORRECT|INCORRECT" merge_eval.txt
```

All workloads must be CORRECT. If any fail, the changes interact subtly — one worker's assumption is violated by the other. Investigate; don't just pick one without understanding why.

### 6. Measure

Run the eval 5 times, compute the mean speedup and stddev. Compare to:
- Worker 1 alone: `S1`
- Worker 2 alone: `S2`
- Merge: `SM`

### 7. Decide

| Outcome | Action |
|---|---|
| `SM > max(S1, S2)` by ≥ 1% | **KEEP merge.** Tag as new baseline. |
| `SM ≈ max(S1, S2)` (within 1%) | The changes don't compose. **Adopt the better of the two** (S1 or S2). Log that the merge was flat. |
| `SM < max(S1, S2)` | The changes interfere. **Investigate** before deciding. Don't just throw out the better single change. |
| `SM < min(S1, S2)` | Interference is severe. Adopt the better single. Write a careful note on the interaction — it's likely to repeat. |

### 8. Tag and update state

```bash
git tag team-baseline-r<N+1>
# Update team_notes/current_baseline.md
# Copy new baseline into both agent dirs so workers start round N+1 from merged state:
cp solution/cuda/kernel.cu agent1_dir/solution/cuda/kernel.cu
cp solution/cuda/kernel.cu agent2_dir/solution/cuda/kernel.cu
```

Both workers now start round N+1 from the merged kernel.

## Common interaction failures

### "Changes don't compose"

Worker 1 moved Q to registers, worker 2 added `cp.async` for Q. Merged kernel tries to do both — the `cp.async` is redundant and the register hold dominates, so you get worker 1's win plus a small overhead. `SM < S1`.

**Fix**: pick the more principled direction (usually the one with the larger single-run win). Note the interaction for future reference.

### "Conflicting smem layout"

Worker 1 used `[128][129]` padding for bank conflicts. Worker 2 re-shaped the tile to `[64][256]` and didn't need padding. Merged layout is inconsistent — compile error or wrong results.

**Fix**: one worker's smem layout is right for the new architecture. Pick that one. If worker 2's tile re-shape is the bigger win, drop the padding — it may not even be needed in the new layout.

### "Accumulator precision mismatch"

Worker 1 added bf16 partial output (lower precision). Worker 2 added Split-K with 32 chunks. Merged: 32 bf16 partials = accumulated rounding error over 32 summations. Fails correctness.

**Fix**: if Split-K=32 is the bigger win, revert the bf16 partial (fp32 partial has 2× bandwidth but passes). If bf16 partial is the bigger win, cap Split-K lower.

### "Occupancy collapse"

Worker 1 added `__launch_bounds__(128, 2)` — tight register budget. Worker 2 added a new register-resident accumulator. Combined exceeds the budget → register spills → speedup regresses.

**Fix**: retune `launch_bounds` for the new register count. Or scale back worker 2's accumulator to fit the budget.

## Merge log format

Keep a running log in `team_notes/merge_log.md`:

```markdown
## Round 3 merge

- Base: team-baseline-r2 (avg 157.8x)
- Worker 1: +3.2% — `__launch_bounds__(128, 2)` tuning
- Worker 2: +1.8% — MERGE_DIM_SPLIT=2 in the merge kernel
- Merge: +4.9% → 165.5x
- **Result: KEEP merge.** Tagged team-baseline-r3.

Notes: Changes touched different files (partial kernel vs merge kernel) — clean merge. No interaction issues.
```

This log is the team's institutional memory. When a similar interaction comes up three rounds later, the log saves you rediscovering the lesson.
