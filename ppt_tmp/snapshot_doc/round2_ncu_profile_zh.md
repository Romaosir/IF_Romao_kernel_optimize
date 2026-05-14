# NCU Profile 报告: MOE FP8 Kernel (Round 2)

**日期**: 2026-04-01
**GPU**: NVIDIA B200 (SM100, Blackwell, 148 SMs)
**Kernel**: `solution/cuda/kernel.cu`
**当前基线**: ~64.9× 平均加速比(由 round 1 的 ~60× 提升而来)
**NCU 报告**: `ncu_reports/round2_seqlen{80,901,14107}_full.ncu-rep`

---

## 1. 摘要

**CUTLASS FP8 grouped GEMM 现在已激活**,成功取代了 round 1 的 per-expert cuBLAS FP16 GEMM。这是相对 round 1 最大的变化:

| 指标 | Round 1 (cuBLAS FP16) | Round 2 (CUTLASS FP8) |
|--------|----------------------|----------------------|
| 每次 MOE 调用的 GEMM launch 数 | 32-64(每 expert 一次)| 2(一次 GEMM1 + 一次 GEMM2)|
| 数据类型 | FP16 | FP8 (e4m3fn) |
| Weight dequant | 需要(dequant_w13 + dequant_w2)| 不需要(FP8 原生)|
| GEMM 引擎 | cuBLAS `nvjet_sm100_*` | CUTLASS `device_kernel` |
| Kernel pipeline 长度 | 6-8 步 + 32-64 个 GEMM | 9 步总计 |

**当前 pipeline**(每次 MOE 调用):

1. `routing_kernel` —— 带 softmax 的 TopK routing
2. `scatter_local_assignments_kernel` —— 构造 expert assignment(fused scan + scatter)
3. `gather_fp8_and_scales_tight_k` —— 收集 hidden state 为 FP8 + scale
4. `prep` —— CUTLASS GEMM1 参数准备(1 block,32 thread)
5. `device_kernel` —— **CUTLASS Grouped FP8 GEMM1**(A@W13,所有 expert 一次 launch)
6. `swiglu_to_fp8_tight_kernel` —— SwiGLU + 量化为 FP8
7. `prep` —— CUTLASS GEMM2 参数准备
8. `device_kernel` —— **CUTLASS Grouped FP8 GEMM2**(C@W2,所有 expert 一次 launch)
9. `pull_scatter_bf16_from_bf16_tight_kernel` —— 把结果 scatter 回输出(BF16)

---

## 2. CUTLASS GEMM 配置

GEMM1 和 GEMM2 都使用 CUTLASS SM100 PtrArray TMA WarpSpecialized Blockwise FP8 GEMM:

- **MMA 指令**: `SM100_MMA_F8F6F4_SS`(FP8 × FP8 → FP32 累加)
- **输出**: BF16,通过 TMA store epilogue 写出
- **Scale 处理**: A 和 B 矩阵都用 blockwise scale(128×128 block)

### 按 workload 的 tile 选择:

| Workload | max_M (per expert) | Tile Shape | Pipeline Stages | Grid Size | Block Size | Shared Mem |
|----------|-------------------|------------|-----------------|-----------|------------|------------|
| seq_len=80 | ~2-3 | 64×128×128 (1-SM) | 8 stages | 148 | 384 | 217.6 KB |
| seq_len=901 | ~28 | 64×128×128 (1-SM) | 8 stages | 148 | 384 | 217.6 KB |
| seq_len=14107 | ~440 | 128×128×128 (1-SM) | 5 stages | 148 | 384 | 201.2 KB |

**注**: 2-SM 协同变体(Cl2 = 2×1×1)代码里存在,但实际**没有被选中**,即使在 seq_len=14107(max_M > 512)时也是如此。推测它 `can_implement()` 失败后回退到了 128×128 1-SM 变体。

---

## 3. 各 workload 的时间分布

### seq_len=80(kernel 总时间: ~266 us)

| 步骤 | Kernel | 耗时 (us) | 时间占比 | SM Busy% | DRAM TP% | Achieved Occ% | Regs |
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

**GEMM 合计: 219.11 us (82.3%)** | 非 GEMM overhead: 47.26 us (17.7%)

### seq_len=901(kernel 总时间: ~314 us)

| 步骤 | Kernel | 耗时 (us) | 时间占比 | SM Busy% | DRAM TP% | Achieved Occ% | Regs |
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

**GEMM 合计: 255.71 us (81.5%)** | 非 GEMM overhead: 57.96 us (18.5%)

### seq_len=14107(kernel 总时间: ~1200 us)

| 步骤 | Kernel | 耗时 (us) | 时间占比 | SM Busy% | DRAM TP% | Achieved Occ% | Regs |
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

**GEMM 合计: 904.25 us (75.4%)** | 非 GEMM overhead: 295.78 us (24.6%)

---

## 4. 瓶颈分析

### 4.1 CUTLASS GEMM: occupancy 受限

CUTLASS GEMM kernel 是主导瓶颈,占总时间的 75-82%:

| 指标 | GEMM1 (seq901) | GEMM2 (seq901) | 解读 |
|--------|---------------|---------------|----------|
| 寄存器/thread | 168 | 168 | Block 上限 = 1(寄存器顶死)|
| Shared memory/block | 217.6 KB | 217.6 KB | Block 上限 = 1(shmem 顶死)|
| 理论 occupancy | 18.75% | 18.75% | 寄存器和 shmem **同时**限制 |
| Achieved occupancy | 14.0% | 14.1% | 每 SM 只有 8.97 active warp(满载是 64)|
| Waves per SM | 1 | 1 | 每 SM 正好 1 个 block,无 wave 重叠 |
| No Eligible scheduler cycles | 73.1% | 73.9% | **73% 的周期里没有 warp 能 issue** |
| Warp cycles per instruction | 8.36 | 8.63 | ILP 低,大部分在等 memory |

**关键洞察**: CUTLASS GEMM 受限于 1-block-per-SM(168 寄存器 + 218KB shmem)的硬性约束。每 SM 只有 12 warp(384 thread / 32),73% 的周期没有 eligible warp —— 即使在 GEMM 执行期间,tensor core 也大约 55% 时间处于空闲。

### 4.2 DRAM 带宽利用率

| Workload | GEMM1 DRAM% | GEMM2 DRAM% | GEMM1 L2 Hit% | GEMM2 L2 Hit% |
|----------|-------------|-------------|---------------|---------------|
| seq_len=80 | 77.0% | 62.5% | 8.2% | 8.4% |
| seq_len=901 | 78.8% | 65.4% | 12.0% | 12.3% |
| seq_len=14107 | 27.1% | 25.3% | - | - |

- 小/中 M: GEMM **DRAM 带宽受限**(77-79% 利用率)
- 大 M: GEMM 变为 **compute 受限**(DRAM 仅 27%,compute 达到 56%)
- L2 命中率极低(8-12%)—— 每个 expert 的权重只被它自己的 token 用一次,expert 之间在 L2 上没有权重复用

### 4.3 非 GEMM overhead 分析

seq_len=14107 时,非 GEMM kernel 占 24.6%(296 us):

| Kernel | 耗时 | 占总时间 | 瓶颈 |
|--------|----------|-----------|------------|
| routing_kernel | 109.8 us | 9.1% | Compute-bound(串行 TopK 选择)|
| pull_scatter_bf16_tight | 78.7 us | 6.6% | DRAM-bound(scatter 模式,低 L2 命中)|
| swiglu_to_fp8_tight | 49.2 us | 4.1% | 并行良好(89% occupancy)|
| gather_fp8_tight | 37.7 us | 3.1% | Memory-bound(gather 模式)|
| prep (×2) | 9.3 us | 0.8% | 可忽略(metadata 准备)|
| scatter_local_assignments | 11.1 us | 0.9% | 可忽略 |

seq_len=901 时,overhead 只有 58 us(18.5%)—— routing(14us)、pull_scatter(12us)、gather(8us)、swiglu(9us)。

### 4.4 `prep` Kernel 的串行化问题

`prep` kernel 用 grid=1, block=32 来给 CUTLASS 准备 per-group 参数。每次约 4.5 us(两个 GEMM 加起来 9 us)。绝对时间小,但创造了一个串行依赖点:

- 用 `cudaLaunchAttributeProgrammaticStreamSerialization` 与上游同步
- 只有 1 个 thread block 在 1 个 SM 上 —— 这段时间另外 147 个 SM 是空闲的
- 把参数计算挪到 host 侧可以消除这个串行 gap

---

## 5. 对比: Round 1 vs Round 2

| 指标 | Round 1 (cuBLAS FP16) | Round 2 (CUTLASS FP8) | 提升 |
|--------|----------------------|----------------------|-------------|
| 平均加速比 | ~60× | ~64.9× | +8% |
| GEMM launches (seq901) | 29 (23 GEMM1 + 6 GEMM2) | 2 (1 GEMM1 + 1 GEMM2) | 减少 14.5× |
| GEMM 数据类型 | FP16 | FP8 | 带宽减半 |
| Weight dequant 时间 | ~pipeline 中包含 | 0(已移除)| 移除 |
| GEMM occupancy | 9.6% | 14.0% | +46% |
| GEMM SM busy (seq901) | 37% (GEMM1) | 43% (GEMM1) | +16% |
| Kernel 总时间 (seq901) | ~467 us | ~314 us | 快 33% |

---

## 6. 优化建议(按优先级)

### P0: 提升 CUTLASS GEMM 效率

**6.1 启用 2-SM 协同变体**
代码中已经有 2-SM 协同 GEMM(cluster=2×1×1),应该能通过让 2 个 SM 协作处理 tile 提升 occupancy。当前即使在 large M 时也**没有被选中**。需要 debug `can_implement()` 为什么失败并修复。预期提升: large workload 上 10-20%。

**6.2 探索 Persistent Kernel / Stream-K CUTLASS**
当前 grouped GEMM 恰好 launch 148 个 CTA(每 SM 一个,一个 wave)。但 expert 之间工作量不均时,一些 SM 提前完成进入空闲。Stream-K 调度可以在 SM 间均衡负载。

### P1: 削减非 GEMM overhead

**6.3 优化 large T 下的 routing_kernel**
seq_len=14107 时,routing 占 109.8 us(9.1%)。该 kernel 每 token 一个 block(grid=T),thread 0 做串行 TopK 选择。大 T 时这一段就很显著。选项:
- 用 shared memory reduction 做并行 TopK
- 把 routing + scatter 融合成一个 kernel

**6.4 优化 pull_scatter_bf16_tight**
seq_len=14107 时占 78.7 us(6.6%)。L2 命中率极低(0.5%)说明 scatter 模式较随机。选项:
- 对输出 index 排序以改善 memory coalescing
- 用向量化 store(当前是 scalar BF16 store)

**6.5 把 gather_fp8 + swiglu_to_fp8 融合进 GEMM**
如果 CUTLASS epilogue fusion 可行,把 gather/scatter 的量化步骤融进 GEMM kernel,消除单独的 kernel launch 和中间显存流量。

### P2: 消除串行瓶颈

**6.6 把 `prep` 计算挪到 host 端**
prep kernel(grid=1, block=32)在算每个 group 的 stride 和指针。这部分可以在 CPU 上算完后 memcpy 到 device,与上游 kernel 重叠。省 ~9 us,同时消除串行 gap。

**6.7 减少 kernel launch 开销**
每次 MOE 调用 9 次 kernel launch,每次 host overhead ~2-3 us。可以考虑融合相邻 kernel(例如 scatter + gather,swiglu + prep)。

### P3: 中长期

**6.8 为 small M 写专用 GEMM**
seq_len=80 时,每个 expert 只有 2-3 个 token。CUTLASS 仍然要 launch 148 个 CTA。写一个能更好分配工作的自定义 kernel 处理多个 expert 的 small GEMM 可能有帮助。

**6.9 探索 CUTLASS tile 调优**
64×128×128 tile + 168 寄存器 + 218KB shmem 把 occupancy 限制在 18.75%。如果能找到寄存器更少的 tile 配置(例如更小 tile 或更少 pipeline stage),就可以让每 SM 跑 2 个 block,occupancy 翻倍。

---

## 7. FLOPS 效率分析

seq_len=901 的 GEMM1(M~28 per expert,N=4096,K=7168,~32 group):

- 理论 FLOPs: ~32 × 28 × 4096 × 7168 × 2 = ~52.7 GFLOPS
- 耗时: 158.88 us
- 实测: 52.7 / 0.000159 = 331 TFLOPS
- B200 FP8 peak: ~4500 TFLOPS(带稀疏)或 ~2250 TFLOPS(dense)
- 利用率: ~14.7% of dense peak

seq_len=14107 的 GEMM1(M~440 per expert,N=4096,K=7168,~32 group):

- 理论 FLOPs: ~32 × 440 × 4096 × 7168 × 2 = ~826 GFLOPS
- 耗时: 555.55 us
- 实测: 826 / 0.000556 = 1486 TFLOPS
- 利用率: ~66% of dense peak —— M 大时显著更好

**结论**: CUTLASS FP8 GEMM 在 dense peak FP8 TFLOPS 上的利用率为 15-66%,随 M 大小变化。小 M 的主要限制是低 occupancy(14%)以及权重加载的 DRAM 带宽。大 M 时利用率合理,达到 66%。

---

## 8. 关键发现汇总

1. **CUTLASS FP8 路径已激活** —— 这是相对 round 1 的核心变化
2. **GEMM 仍占主导**,75-82% 的时间,但 launch 数从 32-64 降到 2
3. **Occupancy 是 14%**(每 block 168 寄存器 + 218KB shmem,正好每 SM 1 个 block)
4. **73% 的 scheduler 周期没有 eligible warp** —— 这是当前效率的核心限制
5. **2-SM 协同变体没有被使用**,尽管代码中已经存在
6. **非 GEMM overhead 18-25%**,seq_len=14107 时 routing(9%)和 pull_scatter(7%)占大头
7. **FP8 TFLOPS 利用率**: 小 M 15%,大 M 66%(占 dense peak)
