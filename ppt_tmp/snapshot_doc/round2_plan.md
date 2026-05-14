# Round 2 Optimization Plan: MOE FP8 Kernel on B200

**Baseline:** 64.19x avg speedup (Run1: 63.08x, Run2: 65.30x)
**Target:** 70x+ speedup
**Date:** 2026-04-01

---

## Baseline Per-Workload Speedups

| SeqLen | Avg Speedup | Notes |
|--------|-------------|-------|
| 1      | 108.77x     | Best case (tiny M) |
| 7      | 83.69x      | |
| 14-16  | ~76x        | |
| 32-62  | ~62x        | Main workload range |
| 80     | 58.83x      | |
| 901    | 58.34x      | |
| 11948  | 40.14x      | Large seq |
| 14107  | 37.21x      | Worst case (large M) |

---

## Priority 1: 2-SM Cooperative GEMM (Expected: 8-15% total speedup)

**Rationale**: Current kernel uses `KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100` (1-SM).
Switch to `KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100` where two SMs cooperate on one tile,
doubling compute throughput per tile. B200 has 148 SMs = 74 effective pairs.

**Implementation**:
1. In embedded CUTLASS .so source (kernel.cu lines 136-283), add 2-SM instantiation:
   - Cluster `Shape<_2,_1,_1>` (2-SM cluster)
   - Epilogue: `PtrArrayTmaWarpSpecialized2Sm`
   - Mainloop: `KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100`
   - Tile: `Shape<_128,_128,_128>`
2. Add `extern "C" cutlass_blockwise_fp8_gemm_2sm()` export
3. Load via dlsym (kernel.cu ~line 339)
4. Use 2-SM for GEMM1 and GEMM2 when workload is large enough

## Priority 2: Tile Shape Tuning 128x256x128 (Expected: 3-8%)

For GEMM2 (N=7168): 7168/256=28 tiles vs 7168/128=56 tiles
For GEMM1 (N=4096): 4096/256=16 tiles vs 4096/128=32 tiles

## Priority 3: Pull-Scatter Vectorization

Investigate uint4 reads for pull_scatter, though alignment issues (7168/8=896, 896/256=3.5) make this tricky.

## Priority 4: Profile and iterate

After P1+P2, profile with NCU to identify remaining bottlenecks.

---

## Implementation Order

1. **Step 1**: Add 2-SM CUTLASS variant → benchmark
2. **Step 2**: Add 128x256x128 tile variant → benchmark
3. **Step 3**: Profile with NCU → identify next targets
