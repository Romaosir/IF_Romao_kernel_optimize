# RMS Norm Kernel — Performance Log

Append-only. One entry per experiment (KEEP, REVERT, CRASH, TIMEOUT).

## v1 — Naive baseline

- **Status**: KEEP (initial)
- **Date**: <fill in on first run>
- **Commit**: <short hash>
- **Speedup (vs torch.compile, geomean)**: <to be measured by agent on first eval>
- **Change**: Starter kit — tree-reduced sum-of-squares in smem, scalar loads.
- **Hypothesis**: n/a (baseline)
- **Outcome**: Establishes v1 as the baseline to beat.
- **Profile summary**: <run NCU after this; expect bandwidth-bound, low occupancy for small-batch>
- **Next**: See plan/plan_v2.md. Likely starting point — vectorized loads + warp-shuffle reduction.

---

## Speedup Trend (KEEPs only)

```
(speedup vs torch.compile geomean — filled in by agent during loop)
```
