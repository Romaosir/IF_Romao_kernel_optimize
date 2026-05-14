# Round 2 Summary

**Baseline:** 64.19x avg speedup
**Final:** 64.9x avg speedup (after 2-SM addition, marginal gain)
**Date:** 2026-04-01

## What was tried

### Step 1: 2-SM Cooperative GEMM (KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100)
- Added 2-SM variant with cluster=2x1x1, tile=128x128x128
- With threshold max_M > 512: +1.1% improvement
- NCU profiling showed 2-SM is NOT being activated even for large M (~440)
- Likely `can_implement()` fails silently, falling back to 1-SM

### Step 2: Threshold tuning + wider tile exploration
- Lowered 2-SM threshold from 512 to 128: no improvement (within noise)
- Attempted 128x128 tile threshold change: slight regression
- 64x256x128 tile: not attempted (uncertain benefit vs. compilation cost)

## Key Findings from NCU Profiling
1. **CUTLASS FP8 grouped GEMM IS active** - confirmed working
2. **GEMM occupancy: 14%** - limited by 168 regs + 218KB smem = 1 block/SM
3. **73% idle scheduler cycles** during GEMM
4. **Non-GEMM overhead: 18-25%** for large workloads
   - routing: 9.1% at seq_len=14107
   - pull_scatter: 6.6% at seq_len=14107
5. **FP8 TFLOPS utilization: 15-66%** depending on M size
6. **Run-to-run variance: ~4x** makes small gains hard to measure

## Next Steps
- Debug why 2-SM `can_implement()` fails
- Optimize non-GEMM kernels (routing, pull_scatter, gather)
- Pipeline improvements (overlap SwiGLU with GEMM2)
- Consider alternative GEMM approaches for better occupancy
