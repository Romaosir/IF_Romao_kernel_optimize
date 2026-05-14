# Round 2 优化计划: MOE FP8 Kernel on B200

**基线:** 64.19× 平均加速比 (Run1: 63.08×, Run2: 65.30×)
**目标:** 70×+ 加速比
**日期:** 2026-04-01

---

## 基线 per-workload 加速比

| SeqLen | 平均加速比 | 备注 |
|--------|-------------|-------|
| 1      | 108.77×     | 最佳(M 极小) |
| 7      | 83.69×      | |
| 14-16  | ~76×        | |
| 32-62  | ~62×        | 主要 workload 区间 |
| 80     | 58.83×      | |
| 901    | 58.34×      | |
| 11948  | 40.14×      | 长 seq |
| 14107  | 37.21×      | 最差(M 大) |

---

## Priority 1: 2-SM 协同 GEMM (预期: 8-15% 整体提升)

**理由**: 当前 kernel 用的是 `KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100`(1-SM)。
切换到 `KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100`,让两个 SM 协作处理一个 tile,
单 tile 计算吞吐翻倍。B200 有 148 SM = 74 对有效组合。

**实现**:
1. 在内嵌的 CUTLASS `.so` 源码中(kernel.cu 第 136-283 行)加入 2-SM 实例化:
   - Cluster `Shape<_2,_1,_1>`(2-SM cluster)
   - Epilogue: `PtrArrayTmaWarpSpecialized2Sm`
   - Mainloop: `KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100`
   - Tile: `Shape<_128,_128,_128>`
2. 加 `extern "C" cutlass_blockwise_fp8_gemm_2sm()` 导出
3. 通过 dlsym 加载(kernel.cu 约第 339 行)
4. workload 足够大时,GEMM1 和 GEMM2 都用 2-SM

## Priority 2: Tile Shape 调优 128×256×128 (预期: 3-8%)

GEMM2 (N=7168): 7168/256=28 个 tile vs 7168/128=56 个 tile
GEMM1 (N=4096): 4096/256=16 个 tile vs 4096/128=32 个 tile

## Priority 3: Pull-Scatter 向量化

研究 pull_scatter 用 uint4 读取的可行性,但 alignment 问题(7168/8=896,896/256=3.5)让这条比较 tricky。

## Priority 4: Profile 后迭代

完成 P1+P2 后,用 NCU profile 找下一个目标。

---

## 实现顺序

1. **Step 1**: 加 2-SM CUTLASS 变体 → benchmark
2. **Step 2**: 加 128×256×128 tile 变体 → benchmark
3. **Step 3**: NCU profile → 找下一批目标
