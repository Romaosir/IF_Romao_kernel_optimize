# Round 2 总结

**基线:** 64.19× 平均加速比
**最终:** 64.9× 平均加速比(加入 2-SM 变体后,提升微弱)
**日期:** 2026-04-01

## 这一轮尝试了什么

### Step 1: 2-SM 协同 GEMM (KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100)

- 加入 2-SM 变体,cluster=2×1×1,tile=128×128×128
- 阈值设为 `max_M > 512` 时: +1.1% 提升
- NCU profile 显示 **2-SM 实际并没有被激活**,即使在 large M(~440)时也没有
- 推测 `can_implement()` 静默失败,回退到了 1-SM

### Step 2: 阈值调优 + 更宽 tile 探索

- 把 2-SM 阈值从 512 降到 128: 没有提升(噪声内)
- 尝试 128×128 tile 阈值变更: 轻微 regression
- 64×256×128 tile: 未尝试(收益不确定 vs 编译成本)

## NCU profile 关键发现

1. **CUTLASS FP8 grouped GEMM 已激活** —— 确认正常工作
2. **GEMM occupancy: 14%** —— 受限于 168 寄存器 + 218KB shmem = 每 SM 1 个 block
3. GEMM 期间 **73% 的 scheduler 周期处于 idle**
4. **非 GEMM overhead: 18-25%**(大 workload 上)
   - routing: seq_len=14107 时 9.1%
   - pull_scatter: seq_len=14107 时 6.6%
5. **FP8 TFLOPS 利用率: 15-66%**,随 M 大小变化
6. **Run-to-run 方差 ~4×** 让小幅提升难以测量

## 下一步

- 排查 2-SM `can_implement()` 为什么失败
- 优化非 GEMM kernel(routing、pull_scatter、gather)
- Pipeline 改进(SwiGLU 与 GEMM2 重叠)
- 考虑替代 GEMM 方案以获得更好的 occupancy
