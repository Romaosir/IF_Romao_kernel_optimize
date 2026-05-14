# NCU Profile Report: MOE FP8 Kernel (Round 2)

**Date**: 2026-04-01
**GPU**: NVIDIA B200 (SM100, Blackwell, 148 SMs)
**Kernel**: `solution/cuda/kernel.cu`
**Current baseline**: ~64.9x avg speedup (up from ~60x in round 1)
**NCU Reports**: `ncu_reports/round2_seqlen{80,901,14107}_full.ncu-rep`

---

## 1. Executive Summary

**CUTLASS FP8 grouped GEMM is now ACTIVE** and successfully replaced the per-expert cuBLAS FP16 GEMM path from round 1. This is the biggest change from round 1:

| Metric | Round 1 (cuBLAS FP16) | Round 2 (CUTLASS FP8) |
|--------|----------------------|----------------------|
| GEMM launches per MOE call | 32-64 (one per expert) | 2 (one GEMM1 + one GEMM2) |
| Data type | FP16 | FP8 (e4m3fn) |
| Weight dequant | Required (dequant_w13 + dequant_w2) | Not needed (FP8 native) |
| GEMM engine | cuBLAS `nvjet_sm100_*` | CUTLASS `device_kernel` |
| Kernel pipeline length | 6-8 steps + 32-64 GEMMs | 9 steps total |

**Current pipeline** (per MOE invocation):
1. `routing_kernel` - TopK routing with softmax
2. `scatter_local_assignments_kernel` - Build expert assignments (fused scan + scatter)
3. `gather_fp8_and_scales_tight_k` - Gather hidden states as FP8 + scales
4. `prep` - CUTLASS GEMM1 argument setup (1 block, 32 threads)
5. `device_kernel` - **CUTLASS Grouped FP8 GEMM1** (A@W13, all experts in one launch)
6. `swiglu_to_fp8_tight_kernel` - SwiGLU + quantize to FP8
7. `prep` - CUTLASS GEMM2 argument setup
8. `device_kernel` - **CUTLASS Grouped FP8 GEMM2** (C@W2, all experts in one launch)
9. `pull_scatter_bf16_from_bf16_tight_kernel` - Scatter results back to output (BF16)

---

## 2. CUTLASS GEMM Configuration

Both GEMM1 and GEMM2 use CUTLASS SM100 PtrArray TMA WarpSpecialized Blockwise FP8 GEMM:
- **MMA instruction**: `SM100_MMA_F8F6F4_SS` (FP8 x FP8 -> FP32 accumulate)
- **Output**: BF16 via TMA store epilogue
- **Scale handling**: Blockwise scale (128x128 blocks) for both A and B matrices

### Tile selection by workload:

| Workload | max_M (per expert) | Tile Shape | Pipeline Stages | Grid Size | Block Size | Shared Mem |
|----------|-------------------|------------|-----------------|-----------|------------|------------|
| seq_len=80 | ~2-3 | 64x128x128 (1-SM) | 8 stages | 148 | 384 | 217.6 KB |
| seq_len=901 | ~28 | 64x128x128 (1-SM) | 8 stages | 148 | 384 | 217.6 KB |
| seq_len=14107 | ~440 | 128x128x128 (1-SM) | 5 stages | 148 | 384 | 201.2 KB |

**Note**: The 2-SM cooperative variant (Cl2 = 2x1x1) exists in code but is NOT being selected, even for seq_len=14107 where max_M > 512. It likely fails `can_implement()` and falls back to the 128x128 1-SM variant.

---

## 3. Time Breakdown by Workload

### seq_len=80 (total kernel time: ~266 us)

| Step | Kernel | Duration (us) | % Time | SM Busy% | DRAM TP% | Achieved Occ% | Regs |
|------|--------|--------------|--------|----------|----------|---------------|------|
| 1 | routing_kernel | 9.15 | 3.4% | 4.5% | 0.1% | 12.3% | 30 |
| 2 | scatter_local_assignments | 6.18 | 2.3% | 0.1% | 0.0% | 12.8% | 18 |
| 3 | gather_fp8_tight | 7.14 | 2.7% | 1.0% | 0.5% | 10.1% | 32 |
| 4 | prep (GEMM1) | 4.51 | 1.7% | 0.0% | 0.0% | 1.6% | 32 |
| 5 | **CUTLASS GEMM1** | **135.30** | **50.8%** | 43.9% | 77.0% | 14.1% | 168 |
| 6 | swiglu_to_fp8_tight | 6.43 | 2.4% | 2.7% | 1.2% | 11.1% | 27 |
| 7 | prep (GEMM2) | 4.38 | 1.6% | 0.0% | 0.0% | 1.5% | 32 |
| 8 | **CUTLASS GEMM2** | **83.81** | **31.5%** | 37.4% | 62.5% | 14.1% | 168 |
| 9 | pull_scatter_bf16_tight | 9.47 | 3.6% | 1.1% | 1.5% | 12.1% | 56 |

**GEMM total: 219.11 us (82.3%)** | Non-GEMM overhead: 47.26 us (17.7%)

### seq_len=901 (total kernel time: ~314 us)

| Step | Kernel | Duration (us) | % Time | SM Busy% | DRAM TP% | Achieved Occ% | Regs |
|------|--------|--------------|--------|----------|----------|---------------|------|
| 1 | routing_kernel | 14.05 | 4.5% | 32.9% | 0.9% | 63.0% | 30 |
| 2 | scatter_local_assignments | 6.02 | 1.9% | 0.1% | 0.1% | 11.8% | 18 |
| 3 | gather_fp8_tight | 8.32 | 2.7% | 12.2% | 5.4% | 62.4% | 32 |
| 4 | prep (GEMM1) | 4.51 | 1.4% | 0.0% | 0.0% | 1.3% | 32 |
| 5 | **CUTLASS GEMM1** | **158.88** | **50.7%** | 42.9% | 78.8% | 14.0% | 168 |
| 6 | swiglu_to_fp8_tight | 8.64 | 2.8% | 26.2% | 11.4% | 63.5% | 27 |
| 7 | prep (GEMM2) | 4.48 | 1.4% | 0.0% | 0.0% | 1.8% | 32 |
| 8 | **CUTLASS GEMM2** | **96.83** | **30.9%** | 33.3% | 65.4% | 14.1% | 168 |
| 9 | pull_scatter_bf16_tight | 11.94 | 3.8% | 10.6% | 16.1% | 36.3% | 56 |

**GEMM total: 255.71 us (81.5%)** | Non-GEMM overhead: 57.96 us (18.5%)

### seq_len=14107 (total kernel time: ~1200 us)

| Step | Kernel | Duration (us) | % Time | SM Busy% | DRAM TP% | Achieved Occ% | Regs |
|------|--------|--------------|--------|----------|----------|---------------|------|
| 1 | routing_kernel | 109.79 | 9.1% | 68.8% | 1.7% | 90.5% | 30 |
| 2 | scatter_local_assignments | 11.07 | 0.9% | 1.2% | 0.5% | 25.1% | 18 |
| 3 | gather_fp8_tight | 37.73 | 3.1% | 42.0% | 42.3% | 77.5% | 32 |
| 4 | prep (GEMM1) | 4.42 | 0.4% | 0.0% | 0.0% | 1.3% | 32 |
| 5 | **CUTLASS GEMM1** | **555.55** | **46.3%** | 55.8% | 27.1% | 14.0% | 168 |
| 6 | swiglu_to_fp8_tight | 49.18 | 4.1% | 77.3% | 34.1% | 89.4% | 27 |
| 7 | prep (GEMM2) | 4.90 | 0.4% | 0.0% | 0.0% | 1.3% | 32 |
| 8 | **CUTLASS GEMM2** | **348.70** | **29.1%** | 42.0% | 25.3% | 14.1% | 168 |
| 9 | pull_scatter_bf16_tight | 78.69 | 6.6% | 24.9% | 61.9% | 43.9% | 56 |

**GEMM total: 904.25 us (75.4%)** | Non-GEMM overhead: 295.78 us (24.6%)

---

## 4. Bottleneck Analysis

### 4.1 CUTLASS GEMM: Occupancy-Limited

The CUTLASS GEMM kernels are the dominant bottleneck at 75-82% of total time:

| Metric | GEMM1 (seq901) | GEMM2 (seq901) | Analysis |
|--------|---------------|---------------|----------|
| Registers/thread | 168 | 168 | Block limit = 1 (max regs) |
| Shared memory/block | 217.6 KB | 217.6 KB | Block limit = 1 (max smem) |
| Theoretical occupancy | 18.75% | 18.75% | Limited by BOTH regs and smem |
| Achieved occupancy | 14.0% | 14.1% | Only 8.97 active warps/SM (of 64 max) |
| Waves per SM | 1 | 1 | Exactly 1 block per SM, no wave overlap |
| No Eligible scheduler cycles | 73.1% | 73.9% | **73% of cycles, no warp can issue** |
| Warp cycles per instruction | 8.36 | 8.63 | Low ILP, mostly waiting for memory |

**Key insight**: The CUTLASS GEMM is fundamentally limited by the 1-block-per-SM constraint (168 regs + 218KB smem). With only 12 warps per SM (384 threads / 32), and 73% of cycles with no eligible warps, the tensor cores are ~55% idle even during GEMM execution.

### 4.2 DRAM Bandwidth Utilization

| Workload | GEMM1 DRAM% | GEMM2 DRAM% | GEMM1 L2 Hit% | GEMM2 L2 Hit% |
|----------|-------------|-------------|---------------|---------------|
| seq_len=80 | 77.0% | 62.5% | 8.2% | 8.4% |
| seq_len=901 | 78.8% | 65.4% | 12.0% | 12.3% |
| seq_len=14107 | 27.1% | 25.3% | - | - |

- For small/medium M: GEMMs are **DRAM bandwidth bound** (77-79% utilization)
- For large M: GEMMs become **compute bound** (DRAM only 27%, compute reaches 56%)
- L2 hit rate is very low (8-12%) -- each expert's weights are only used once for its tokens, no inter-expert weight reuse in L2

### 4.3 Non-GEMM Overhead Analysis

For seq_len=14107, non-GEMM kernels account for 24.6% (296 us):

| Kernel | Duration | % of Total | Bottleneck |
|--------|----------|-----------|------------|
| routing_kernel | 109.8 us | 9.1% | Compute-bound (serial TopK selection) |
| pull_scatter_bf16_tight | 78.7 us | 6.6% | DRAM-bound (scatter pattern, low L2 hit) |
| swiglu_to_fp8_tight | 49.2 us | 4.1% | Well-parallelized (89% occupancy) |
| gather_fp8_tight | 37.7 us | 3.1% | Memory-bound (gather pattern) |
| prep (x2) | 9.3 us | 0.8% | Negligible (metadata setup) |
| scatter_local_assignments | 11.1 us | 0.9% | Negligible |

For seq_len=901, the overhead is only 58 us (18.5%) -- routing (14us), pull_scatter (12us), gather (8us), swiglu (9us).

### 4.4 The `prep` Kernel Serialization Issue

The `prep` kernel runs with grid=1, block=32 to set up CUTLASS per-group arguments. It takes ~4.5 us each (9 us total for both GEMMs). While small in absolute terms, it creates a serial dependency point:
- Uses `cudaLaunchAttributeProgrammaticStreamSerialization` to synchronize with prior work
- Only 1 thread block on 1 SM -- the other 147 SMs are idle during this time
- This serialization gap could be eliminated by computing arguments on the host side

---

## 5. Comparison: Round 1 vs Round 2

| Metric | Round 1 (cuBLAS FP16) | Round 2 (CUTLASS FP8) | Improvement |
|--------|----------------------|----------------------|-------------|
| Avg speedup | ~60x | ~64.9x | +8% |
| GEMM launches (seq901) | 29 (23 GEMM1 + 6 GEMM2) | 2 (1 GEMM1 + 1 GEMM2) | 14.5x fewer |
| GEMM data type | FP16 | FP8 | 2x less bandwidth |
| Weight dequant time | ~includes in pipeline | 0 (eliminated) | Removed |
| GEMM occupancy | 9.6% | 14.0% | +46% |
| GEMM SM busy (seq901) | 37% (GEMM1) | 43% (GEMM1) | +16% |
| Total kernel time (seq901) | ~467 us | ~314 us | 33% faster |

---

## 6. Optimization Recommendations (Prioritized)

### P0: Improve CUTLASS GEMM Efficiency

**6.1 Try 2-SM Cooperative Variant**
The code has a 2-SM cooperative GEMM (cluster=2x1x1) that should give better occupancy by allowing 2 SMs to cooperate on a tile. Currently it's NOT being selected even for large M. Debug why `can_implement()` fails and fix it. Expected improvement: 10-20% on large workloads.

**6.2 Investigate Persistent Kernel / Stream-K CUTLASS**
The current grouped GEMM launches exactly 148 CTAs (one per SM, one wave). For workloads where the work is uneven across experts, some SMs finish early and go idle. Stream-K scheduling could help balance load across SMs.

### P1: Reduce Non-GEMM Overhead

**6.3 Optimize routing_kernel for large T**
At seq_len=14107, routing takes 109.8 us (9.1%). The kernel processes one token per block (grid=T) with serial TopK selection in thread 0. For large T, this is significant. Options:
- Use parallel TopK with shared memory reduction
- Fuse the routing + scatter into one kernel

**6.4 Optimize pull_scatter_bf16_tight**
Takes 78.7 us (6.6%) at seq_len=14107. Low L2 hit rate (0.5%) suggests random scatter pattern. Options:
- Sort output indices for better memory coalescing
- Use vectorized stores (currently scalar BF16 stores)

**6.5 Fuse gather_fp8 + swiglu_to_fp8 with GEMM**
If CUTLASS epilogue fusion is possible, fuse the gather/scatter quantization steps into the GEMM kernel to eliminate separate kernel launches and intermediate memory traffic.

### P2: Eliminate Serial Bottlenecks

**6.6 Move `prep` computation to host**
The prep kernel (grid=1, block=32) computes per-group strides and pointers. This could be done on the CPU and memcpy'd to device, overlapped with prior kernels. Saves ~9 us and eliminates serialization gap.

**6.7 Reduce kernel launch overhead**
9 kernel launches per MOE call, with ~2-3 us host overhead each. Consider fusing adjacent kernels (e.g., scatter + gather, swiglu + prep).

### P3: Longer-term

**6.8 Custom GEMM for small M**
For seq_len=80, each expert has only 2-3 tokens. CUTLASS still launches 148 CTAs for this. A custom kernel that processes all experts' small GEMMs with better work distribution could help.

**6.9 Investigate CUTLASS tile tuning**
The 64x128x128 tile with 168 registers and 218KB smem limits occupancy to 18.75%. If a tile configuration exists with fewer registers (e.g., smaller tile or fewer pipeline stages), it could allow 2 blocks per SM and double occupancy.

---

## 7. FLOPS Efficiency Analysis

For seq_len=901 GEMM1 (M~28 per expert, N=4096, K=7168, ~32 groups):
- Theoretical FLOPs: ~32 * 28 * 4096 * 7168 * 2 = ~52.7 GFLOPS
- Duration: 158.88 us
- Achieved: 52.7 / 0.000159 = 331 TFLOPS
- B200 FP8 peak: ~4500 TFLOPS (with sparsity) or ~2250 TFLOPS (dense)
- Utilization: ~14.7% of dense peak

For seq_len=14107 GEMM1 (M~440 per expert, N=4096, K=7168, ~32 groups):
- Theoretical FLOPs: ~32 * 440 * 4096 * 7168 * 2 = ~826 GFLOPS
- Duration: 555.55 us
- Achieved: 826 / 0.000556 = 1486 TFLOPS
- Utilization: ~66% of dense peak -- much better with larger M

**Conclusion**: CUTLASS FP8 GEMM achieves 15-66% of peak FP8 TFLOPS depending on M size. The primary limiter for small M is low occupancy (14%) and DRAM bandwidth for weight loading. For large M, utilization is reasonable at 66%.

---

## 8. Summary of Key Findings

1. **CUTLASS FP8 path IS active** -- this is the main change from round 1
2. **GEMM still dominates** at 75-82% of total time, but now with only 2 launches instead of 32-64
3. **Occupancy is 14%** (limited by 168 regs + 218KB smem per block, exactly 1 block/SM)
4. **73% of scheduler cycles have no eligible warps** -- this is the primary efficiency limiter
5. **2-SM cooperative variant is NOT being used** despite being available in the code
6. **Non-GEMM overhead is 18-25%**, with routing (9%) and pull_scatter (7%) being largest for seq_len=14107
7. **FP8 TFLOPS utilization**: 15% (small M) to 66% (large M) of dense peak
