# Performance log format

`notes/perf_log.md` is append-only and must be readable at a glance months later. One entry per experiment — KEEP, REVERT, crash, or timeout.

## Entry template

```markdown
## v<N> — <one-line change description>

- **Status**: KEEP | REVERT | CRASH | TIMEOUT
- **Date**: YYYY-MM-DD HH:MM
- **Commit**: <short hash>
- **Speedup**: avg=<X.XX>x  min=<X.XX>x  max=<X.XX>x  (workloads: N CORRECT / M total)
- **Change**: <2-3 sentence description — what you did and where>
- **Hypothesis**: <why you expected this to help>
- **Outcome**: <what actually happened; any surprising workload-level behavior>
- **Profile summary**: <1-2 lines from NCU: compute %, DRAM %, occupancy, top stall>
- **Next**: <the specific thing you plan to try next, or "see plan_v<N+1>.md">
```

Keep entries tight — 10–15 lines each. Long rationale goes in `plan/plan_vN.md`, not here.

## Why include REVERTs

REVERTs are the highest-information-density entries in the log. They tell future-you (and anyone reviewing) what *didn't* work and why. A log with only KEEPs is dishonest — it pretends the optimization was linear.

A good REVERT entry has:
- The exact speedup delta (e.g., "-3.2%")
- The hypothesis that was disproved
- One sentence on what the profile showed that explained the regression
- A note on whether the idea is dead or just mistimed ("retry after Split-K is in")

## Speedup trend chart

Maintain an ASCII chart at the bottom of `perf_log.md`. Update it on every KEEP.

```
## Speedup Trend (KEEPs only)

Speedup
  ▲
170┤                                          ● v36 164.5x
160┤                                        ● v32 157.8x
150┤
140┤
130┤                                    ● v31 127.9x
120┤                                ● v28 120.3x
110┤                            ● v26 109.7x
100┤                        ● v25 104.3x
 90┤                  ● v22 88.6x
   │               ● v21 87.6x
 80┤        ● v9 78.1x  ● v16 83.3x
 70┤   ● v7 64.9x
 60┤ ● v5 52.7x
 50┤
 40┤ ● v1 43.7x
   └──┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬── KEEPs
      v1  v5  v7  v9  v16 v21 v22 v25 v26 v28 v31 v32 v36
```

The chart is powerful for two reasons:
1. **Monotonicity is visible.** A zigzag means your keep/revert discipline is broken — you kept a regression somewhere.
2. **Plateaus are obvious.** When the curve flattens, you know to consult the plateau row in `cuda-roofline-strategy`.

## Workload-level table (optional, recommended for final submission)

When you're near the end of a run, add a workload-level breakdown of the best version so you can spot outliers:

```markdown
## v<BEST> — per-workload speedup

| Workload | tokens | topk | avg speedup | status |
|----------|--------|------|-------------|--------|
| ...      | 1      | 2048 | 314.4x      | CORRECT |
| ...      | 8      | 2048 | 163.7x      | CORRECT |
| ...      | 2      | 2048 | 22.4x       | CORRECT (bottleneck) |
```

The workload with the lowest speedup is usually your next optimization target — see `cuda-roofline-strategy`, "per-workload NCU" section.

## Don'ts

- Don't edit old entries. If you discover an error later, add a note in the *next* entry: "v17 speedup was mis-parsed; actual was 86.1x, not 91.2x."
- Don't skip REVERTs to keep the log "clean." They're not noise.
- Don't put code in the log. Code lives in git; the log links to commits via hash.
- Don't put more than one change per entry. If your experiment bundled two changes, split them retroactively in the entry: "v<N>a: changed block size to 128. v<N>b: added `__launch_bounds__`." Then commit separately next time.
