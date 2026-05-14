# Phase C — submission checklist

Phase C runs when a stopping condition fires or the user says wrap it up. The goal is a clean, reproducible artifact plus a compact summary of what worked.

## 1. Final eval — clean room

Before packaging anything:

1. `git status` — working tree must be clean.
2. `git log --oneline -10` — confirm the HEAD is the kept version you think it is.
3. Re-run the full eval, fresh:
   ```bash
   bash scripts/eval_solution.sh > eval_output_final.txt 2>&1
   grep -E "CORRECT|INCORRECT|speedup" eval_output_final.txt
   ```
4. All workloads must be CORRECT. If any fail, you cannot submit — fix the regression, loop back into Phase B.

## 2. Measure with noise control

Single-run numbers lie. Run the eval 5 times and record:

- Per-run avg speedup
- 5-run mean
- 5-run stddev
- Min of the 5 runs (worst realistic case)

If stddev > 2% of mean, the benchmark is noisy; increase the run count or note it in the submission. If mean dropped vs. what you had logged, something regressed between your last KEEP and now — investigate before submitting.

## 3. Package the artifact

Follow whatever the user's framework requires. Common forms:

- `python scripts/pack_solution.py` → `solution.json`
- `bash scripts/build_submission.sh` → `submission.tar.gz`
- A container or PR

If the framework wants specific files (`kernel.cu`, `binding.py`, metadata), double-check each is present and references the correct paths.

## 4. Git tag and push

Only push when the user has said they want a remote push (it's a shared-state action — ask if unclear). The tag convention:

```bash
git tag submission-<tag>-v<N>
git push origin <branch>
git push origin submission-<tag>-v<N>
```

Include the speedup in the tag message:

```bash
git tag -a submission-0416-v38 -m "avg 161.3x, min 19.0x, max 317.0x, 5-run stddev 0.8%"
```

## 5. Write the submission summary

`notes/submission_<tag>.md` — short, scannable, focused on what a reviewer needs to know.

Template:

```markdown
# Submission: <tag> v<N>

## Headline

avg **161.3x** (5-run mean, stddev 0.8%), min 19.0x, max 317.0x across 23 workloads. All CORRECT.

## What worked

1. **Split-K FlashDecoding with adaptive K** (v25–v31) — single biggest gain. Varying K per token count lets small workloads avoid merge overhead while large workloads saturate SMs. Contributed +64% (v9 → v31).
2. **Register-centric data path** (v3–v9) — moving Q and output accumulator from shared memory to registers. Contributed +78% (v1 → v9).
3. **4 heads per block** (v32) — architectural change from the reference kernel. Enables KV reuse across heads within a CTA. Contributed +24% in one step.
4. **Per-workload NCU** (v36) — profiling the 2-token outlier revealed 6.21% occupancy; targeted Split-K fix. +7%.

## What didn't work (in this run)

- TMA — setup cost larger than benefit for this access pattern
- L2 persistence API — per-call overhead ate the gain
- WMMA tensor cores — scattered access pattern doesn't fit the fragment shapes
- HEADS_PER_BLOCK=8 — register pressure > 68 regs/thread, occupancy crashed

## Remaining ideas

- cp.async.bulk (TMA-via-ptx) on the KV load path
- bf16-in-bf16-out merge kernel
- Warp-specialized producer/consumer pipeline

## Speedup curve

[copy the latest ASCII chart from perf_log.md]

## Files of record

- Kernel: `solution/cuda/kernel.cu` @ <hash>
- Eval output: `results/all_eval_v38.txt`
- Full perf log: `notes/perf_log.md`
- Final profile: `profile/profile_v38.md`
```

This is the one place to editorialize. The perf log is chronological ground truth; the submission summary is the narrative you'd tell a colleague.

## 6. Report back to the user

Post a 4–8 line summary in the conversation:
- Final speedup (with stddev if noisy)
- Number of experiments total / KEEPs / REVERTs
- 2–3 techniques that drove most of the gain
- Path to the submission summary file
- Whether you pushed anything

## Don'ts

- **Don't pack an untested commit.** The final eval runs on HEAD, not on the last KEEP.
- **Don't "clean up" the perf log** before submitting. The messy log, including REVERTs, is the valuable artifact.
- **Don't fabricate a 5-run average** from a single run. If the user's eval is slow, say so and run fewer.
- **Don't push without permission** unless the user has pre-authorized pushes for this run.
