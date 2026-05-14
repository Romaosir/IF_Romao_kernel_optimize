# The experiment loop — per-iteration checklist

One iteration = one hypothesis tested. Each iteration should produce exactly one of: a KEEP (new baseline), a REVERT (nothing lost, knowledge gained), or a CRASH/TIMEOUT (log and move on).

## 1. Hypothesize

Before you touch code, write a short plan to `plan/plan_vN.md`:

- **Current bottleneck.** From the latest profile: compute / bandwidth / occupancy / latency / synchronization. If you don't know, stop and profile first.
- **The change.** One sentence. E.g. "Switch ckv-cache loads from scalar to float4 via `__ldg`." Specific enough that you could hand it to someone else to implement identically.
- **Expected mechanism.** Why should this help? "Reduces global-load instructions 8×, should lift DRAM throughput from 60% to 80%."
- **Expected magnitude.** Rough — "+5–10%". Writing this down calibrates you. If reality is an order of magnitude off, that's diagnostic.
- **What would disprove the hypothesis.** "If speedup is unchanged, bandwidth wasn't the bottleneck." "If it regresses, register pressure increased — check ptxas output."
- **Fallback.** "If this reverts, try half-wave reorder of the tile schedule next."

Writing the fallback *before* you run prevents panic-driven experimentation after a revert.

## 2. Implement

Edit **only** the files that need to change. In practice that's `kernel.cu` and occasionally `binding.py` (if the signature changes) or `config.toml` (if a tuning knob is exposed). Resist the urge to "fix a small thing" in an unrelated file — it breaks attribution.

Keep the change focused. Examples of a single focused change:
- Change `BLOCK_SIZE_M` from 64 to 128.
- Add `__launch_bounds__(128, 2)`.
- Swap scalar loads for `float4` vectorized loads on one tensor.
- Add `cp.async` double-buffering for the K-tile.
- Fuse two kernels into one.
- Add bank-conflict padding to the shared-memory layout.

Anti-examples (multiple changes bundled):
- "Retuned block size *and* added vectorization" — if it wins, which one won?
- "Fixed a bug *and* added pipelining" — bug fixes go in their own commit.

**The "principled rewrite" exception.** Sometimes multiple changes are *semantically one
change* — they only make sense together. Example: a new kernel architecture that moves Q
from smem to registers *requires* switching the reduction from smem-tree to warp-shuffle,
*requires* vectorized loads to fill the registers, *requires* a different block shape.
These aren't independent tunings; they're one coherent design. Bundle them, name the
*design* in the commit ("v5: register-held query + warp-shuffle reduction + 4 uint4 per
thread"), and in the perf_log note what the *design* is doing differently — not the
individual ingredients. Independent tunings (block size, `__launch_bounds__`, padding)
still go in their own commits. If you can't describe the bundle as a single coherent
design in one sentence, it's not a principled rewrite — split it.

Empirically (from our benchmark runs), bundling a principled rewrite like this often
produces the single largest jump in the whole run. Attribution is preserved because you
named the design; future-you can still understand what happened even if you can't
isolate which ingredient contributed how much, because the ingredients aren't separable
anyway.

## 3. Snapshot

Save a pre-eval snapshot of the kernel so reverts are a one-line `cp`. Two equivalent mechanisms; pick whichever the launch prompt specifies:

**File-based (default for these experiments):**
```bash
cp solution/cuda/kernel.cu snapshot_code/kernel_v<N>_pending.cu
```
On KEEP, rename to `kernel_v<N>_<speedup>x.cu`. On REVERT, `cp snapshot_code/kernel_v<prev>_<speedup>x.cu solution/cuda/kernel.cu` to restore the last known-good kernel.

**Git-based (only if the kit has `.git` and the prompt opts in):**
```bash
git add solution/cuda/ && git commit -m "vN: <one-line hypothesis>"
```
On REVERT: `git reset --hard HEAD~1`.

Snapshot *before* you run the eval. If the eval blows up or you need to revert, the previous kernel is one operation away. If you didn't snapshot, you can't revert, and now you have to hand-unwind changes.

## 4. Eval

Run the framework's full eval, capturing the complete output:

```bash
bash scripts/eval_solution.sh > eval_output.txt 2>&1
```

(Or whatever the user's framework uses. If the user has a one-shot script, use it.)

Save the full output: `cp eval_output.txt results/all_eval_vN.txt`. Future you will want to diff workload-level results across versions.

## 5. Check results

Parse for status and speedup:

```bash
grep -E "PASSED|FAILED|CORRECT|INCORRECT|speedup|error|Timeout" eval_output.txt
```

If grep returns nothing, something catastrophic happened — read `tail -n 80 eval_output.txt` for a traceback. Don't assume "no errors found" means success.

Extract three numbers: **avg speedup** (usually geo-mean), **min speedup**, **max speedup**. The min matters a lot: a 2× avg with a 0.3× min is a regression for whoever gets the worst workload.

**N = 2 runs minimum on noisy hardware.** Single-run comparison is unreliable when run-to-run variance is in the 3–4% range (typical for B200; check your chip with `nvidia-smi` and a couple of baseline reruns). Run the eval **twice** for any change, average the speedup, and use the average to decide. Re-measure the *baseline* periodically (every several rounds, or after any structural refactor) so cumulative drift doesn't silently inflate or deflate later gains. Always use the same GPU for baseline and variant — switching devices mid-comparison invalidates the delta.

## 6. Decide

**Anti-cheating rule — reject benchmark-specific shortcuts.** Speedups that depend on sample-specific caching, fixed input shapes that the eval harness happens to use, hardcoded constants tied to the test set, or any other artefact of *the benchmark* rather than the operator are not real optimizations. They will not survive deployment and will reverse the moment the input distribution shifts. Before you KEEP, explicitly ask: *would this win survive if the inputs were permuted or shape-shifted slightly?* If a gain depends on a legitimately deployable cache (e.g., warmup-time weight format conversion), that's fine — but quantify cold-path vs warm-path separately so the gain is honest.



See the keep/revert table in `SKILL.md`. In short:

- Any INCORRECT → REVERT.
- All CORRECT and avg improved ≥ 1% → KEEP.
- All CORRECT but flat/regressed → REVERT (unless code is meaningfully simpler).
- Crash or timeout → REVERT, log, possibly switch direction after 3 in a row.

When you REVERT: restore the prior kernel via whichever snapshot mechanism is in use (`cp snapshot_code/kernel_v<prev>_<speedup>x.cu solution/cuda/kernel.cu`, or `git reset --hard HEAD~1` if the kit uses git). Log the revert *with the speedup you got* — failed experiments are data.

**Revert discipline for previously-validated techniques.** Some techniques you'll try are *named, measured, and validated* — they have an entry in a reference catalog or in an operator's measured optimization ladder, with a known % gain. When one of these regresses on your first attempt, do **not** silently revert. The first hypothesis should be an *orthogonal bug* (pointer alignment, stale buffer, missing `__threadfence`, wrong `sm_*a` compile target, an unrelated kernel introduced in a recent commit) — not "the named technique is wrong on this hardware." Procedure: (a) re-read the matching code example or reference, compare your implementation line-by-line to spec; (b) check whether some recent unrelated change could be the regressor; (c) only revert *after* you've named what's different from the known-working pattern. Log the hypothesis in your snapshot doc so the next attempt has breadcrumbs. The catch-all "this didn't help" revert is what burns the most ground in long-horizon runs — it abandons known-good territory because of an orthogonal bug nobody traced.

## 7. Profile (after any significant change)

"Significant" = any KEEP, or any REVERT that was surprising enough you want to understand why. See the `ncu-cuda-profiling` skill for how to collect. Save analysis to `profile/profile_vN.md` with:

- Compute throughput % of peak
- DRAM throughput % of peak
- L2 hit rate
- Achieved occupancy
- Top 3 warp stall reasons
- Any bank conflict counts
- Notable differences from the previous version's profile

The profile → strategy step is what `cuda-roofline-strategy` is for.

## 8. Log

Append to `notes/perf_log.md` in the format documented in `perf-log-format.md`. Every run gets an entry — KEEPs, REVERTs, crashes, timeouts. The log is append-only; never edit old entries.

Update the ASCII speedup trend chart at the bottom of the log on every KEEP.

## 9. Plan next

**Mandatory: look up the strategy-matrix cell before writing the plan.** Open `cuda-roofline-strategy/references/strategy-matrix.md`, find the cell matching your current (roofline position × iteration phase) from the latest profile, and **name the cell verbatim in `plan/plan_vN+1.md`** along with the candidate techniques it lists. Pick one specific technique with a one-sentence reason for choosing it over its siblings. Then open the matching `cuda-kernel-techniques/references/<topic>.md` for implementation detail (when it helps / when it hurts / code sketch / field notes).

If you can't name the cell, you don't have enough NCU data — re-profile before planning. Citing the cell forces the plan to be *consequential of the profile* rather than a guess. Plans that don't reference a specific strategy-matrix cell tend to drift toward repeating the same category of tweak; cite the cell to break that loop.

**Switch categories when you're stuck.** If the last 2–3 experiments were all
in the *same category* and all reverted (e.g., three register-cache-size
tweaks, or three block-size variants), do not pick a fourth variant of the
same category. Switch to a *different category* — the specific checklist is
the "Tactical sweep" section in `cuda-roofline-strategy/references/strategy-matrix.md`.
Register-cache tweaks → try a bank-conflict swizzle next, or a `__launch_bounds__`
sweep. Block-size tweaks → try an `__ldg` pass next, or a block-schedule/L2
swizzle. Same-category retries are how optimization runs get stuck in local
minima; different-category moves are how they get unstuck.

**Counter the easy-direction bias (forcing function at plateau).** Once
you're in plateau phase (≥ 3 consecutive rounds of < 1% gain, or the run is
near a known landmark and stalled), planning has a well-documented bias: it
favours another small tweak — block size, padding, a launch-bounds nudge —
over a heavy lift like hand-rolling a next-gen tensor-core kernel, writing
a custom CUTLASS collective, or restructuring the pipeline. Small tweaks
have positive, well-calibrated expected value; heavy lifts have high
variance and are easy to over-estimate cost on. Both forces push toward
"another tweak" even when the heavy lift is what the roofline matrix calls
for.

Forcing function — apply *before* writing the next plan when at plateau:

1. Open the matching **Plateau** cell in
   `cuda-roofline-strategy/references/strategy-matrix.md` for your
   (position, phase) pair. List every item in that cell.
2. Also list every untried *heavy lift* in your operator's measured ladder
   (e.g. `cuda-kernel-techniques/references/operators/<op>/optimization-ladder.md`)
   — entries marked as "structural" or "rewrite" or with explicit warnings
   that they're hard.
3. For each item on either list: *explicitly write why you are NOT picking
   it this round*. Acceptable reasons: prerequisite not yet met (e.g. you
   haven't done zero-sync yet), reference template missing, hard hardware
   constraint. Unacceptable reasons: "looks expensive", "uncertain payoff",
   "next round". If you can't articulate a hardware-or-prerequisite reason
   to skip a heavy lift, that's the next plan.
4. Only after the heavy-lift menu is exhausted may you fall back to another
   small tweak — and if you do, name *which* heavy lift you're punting and
   *what would unblock it next round*.

The agent owns the plan and the implementation. This rule does not have a
human dictate either. It exists because, without it, plateau-stage planning
silently converges on safe tweaks and the heavy-lift menu never gets
attempted — the failure mode documented in
`cuda-kernel-techniques/references/operators/moe/manager-failures.md`
Pattern 1, where a campaign stalled near 80× and required external nudging
to attempt tcgen05 (which then delivered +6.8% combined and unlocked the
final ~93× state).

## 10. Online-search rule

Keep a running counter of consecutive reverts/crashes. When it hits 3:

1. Stop trying variations of the current direction.
2. Web-search for new ideas. Useful queries:
   - `"<operator name>" CUDA kernel optimization <target GPU>`
   - `FlashAttention <variant> CUDA implementation`
   - `<recent paper> CUTLASS example`
   - `sm_<XX> <bottleneck> technique`
3. Write the top 2-3 promising leads into the next plan file, with URLs.
4. Reset the counter when you get a KEEP from a search-inspired experiment.

This rule exists because three in a row usually means your mental model of the bottleneck is wrong. Fresh external input helps more than another local perturbation.

## 11. When external input is needed: ask for *direction*, not code

If after web-search (Step 10), category-switch (Step 9 tactical sweep), and the plateau-bias forcing function (Step 9 again) you still genuinely need outside input from the user, the *form* of the ask matters more than people expect.

**Do** ask for direction:
- "Should I try tcgen05 next, or finish item N on the ladder first?"
- "Are we in ship mode or explore mode at this point?"
- "Is there a specific paper / repo / reference I should consult?"
- "Is this regression worth chasing, or should I revert and move on?"
- "Does the user care about the small-T workload, or is the long-seq case the one we're optimising?"

**Do NOT** ask for:
- Code, templates, or "what should the kernel look like"
- The exact implementation of a technique you've already chosen
- Line-by-line hand-holding through a CUTLASS / tcgen05 / PTX snippet
- Pre-written diffs to apply

**Why this rule exists.** The bias documented at Step 9 (plateau / easy-direction) is in *direction-selection*, not in plan or code generation. The agent's planning and implementation are competent; a human nudge at the direction layer is enough to break the bias. A human code hand-off, by contrast, *removes the agent's ability to recover from later regressions* because the agent no longer owns the design — it's stenographing. The first time something downstream needs to adapt to that handed-over code, the agent has no model of why the code is shaped that way, and the regression cascades.

**Concrete contrast.** Instead of "can you write me the tcgen05 grouped-GEMM kernel?", ask "I've been avoiding tcgen05 because <X>. Should I attempt it next, and is there a reference template?" The user's reply ("yes, look at gau-nernst matmul_v7") is *direction*; the agent still writes and debugs the code. This matches the field evidence in `cuda-kernel-techniques/references/operators/moe/manager-failures.md` Pattern 1 — every successful human intervention in the FuseMoE campaign supplied a direction or a URL, not code; every attempt to hand over code shortcut the agent's debugging loop and produced worse outcomes.

**Special case — when you genuinely lack a reference.** If you've web-searched (Step 10), the top references don't match your workload, and you're about to write something from first principles in a domain where good references exist somewhere but aren't surfacing: it's fine to ask the user *"do you know a reference for X?"* — that's still direction. Receiving the URL and then reading and adapting it yourself preserves ownership.

## 12. Special cases

**First experiment after a KEEP that came from an architectural rewrite.** Re-profile before planning. The whole roofline picture may have shifted.

**You find a bug during optimization** (e.g., wrong LSE formula). Fix it in a standalone commit with a clear "fix:" prefix; don't bundle it with a performance change. Re-establish baseline speedup, then resume.

**The eval takes >2× normal wall-clock.** Something is hung or degenerate. Kill, revert, log as TIMEOUT.

**Speedup improved but correctness passes with wider tolerances.** This is numerical drift. Before KEEPing, compare the absolute error distribution to the previous baseline. If it's genuinely worse, REVERT even if technically CORRECT.

**A skill claim contradicts observed reality.** A skill says the install behaves one way; the install obviously behaves another (you read the source, you ran the binary, you saw the metric). **Flag the discrepancy to the user before acting on it.** Do not silently override the skill — the discrepancy means either the skill is stale or the environment has drifted, and the optimisation strategy depends on knowing which. Acting silently on an "outdated" claim means you're working from an unverified hypothesis about how the system behaves; the user is the only person who can decide whether to update the skill or restore the environment. A 30-second flag preserves the user's ability to course-correct; silent override loses it.

**The temptation to enter wrap-up mode mid-campaign.** Symptoms: writing detailed snapshot docs at length, drafting "final" summaries, updating the operator catalogue under `operators/<op>/`, recording memories, "let me consolidate" / "let me finalise" turns. These are all *real* end-of-campaign activities — but they pull the agent out of the experiment loop. **Before doing any of them, check: is the speedup curve still climbing? Have I exhausted the heavy-lift menu? If the answers are "yes, climbing" / "no, still options", these are not the right turns to spend on right now — run another experiment.** A campaign ends when the user says it ends, or when the plateau-bias forcing function (Step 9) has been applied AND every heavy lift on the menu has a hardware-or-prerequisite reason for being skipped. "I've shipped enough" or "I've documented enough" are not stopping conditions. Field evidence: the GDN-prefill with-skill run (2026-05-12) under-performed the no-skill control by stopping at v4-final and entering catalog/memory-writing turns while the no-skill control kept iterating, hit the plateau check, and broke through with `cp.async` (R9, +1.7%). The agent that documented less, optimised more, won.

**Stopping check — required before any "final summary" / "campaign complete" turn.** When you reach what feels like a stopping point, before writing the final summary, you MUST first produce an explicit list of untried catalogue techniques relevant to the current (position, phase) cell, with a hardware-or-prerequisite skip reason for each. Format:

```
## Heavy lifts NOT attempted, with skip reasons
1. <technique name> — <hardware-or-prerequisite reason for skipping>
2. <technique name> — <reason>
...
```

Acceptable skip reasons: *prerequisite not met* (e.g. "no TMA descriptor setup yet"), *reference template missing* (e.g. "no GDN-shaped tcgen05 skeleton in `code-examples/`"), *hard hardware constraint* (e.g. "MMA already emits the same PTX for this shape"). Unacceptable: "looks expensive", "uncertain payoff", "next round", "I think I'm done". **If you cannot articulate an acceptable skip reason for one of the techniques, that technique is your next round — not your final state.** This is the no-skill GDN-prefill 2026-05-12 final perf_log pattern: it enumerated tcgen05, TMA, cluster mode, inline-PTX with skip reasons before declaring 2.86× done. With-skill stopped at 2.07× without producing that enumeration. The check is what closes that gap.
