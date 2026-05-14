# From NCU report to classification — walkthrough

A concrete procedure to turn an NCU report into the one-line classification that Step 1 of `SKILL.md` asks for. Useful when the numbers don't tell a clean story.

## Step A — Read the Speed-of-Light section first

NCU's "GPU Speed Of Light Throughput" (`--set full`) section gives you three numbers at the top:

- SM %  — compute throughput
- Memory %  — DRAM (HBM) throughput
- Max (SM or Memory)  — the higher of the two, which is the binding constraint

If either is > 70%, the answer is "you're bound by that." Proceed to Step D.

## Step B — If both are middling (< 70%), check occupancy

Look at "Achieved Occupancy" (`sm__warps_active.avg.pct_of_peak_sustained_active`):

- **< 30%** → **occupancy-limited**, nearly certainly. Go to Step E.
- **30–60%** → possibly occupancy, possibly latency. Check warp stalls (Step C).
- **> 60%** → not occupancy-limited. Go to Step C.

## Step C — Warp stall breakdown (the "why are we idle" section)

NCU's "Warp State Statistics" shows where warp cycles go. The top 3 categories usually tell you:

| Top stall category | Likely bottleneck class | Next move |
|---|---|---|
| Long Scoreboard | Global-memory latency | Pipelining, `cp.async`, prefetch |
| Short Scoreboard | Shared-memory / register data dependency | ILP, hoist invariants, unroll |
| Wait | `__syncthreads()` / barriers | Rebalance loads across warps, fewer sync points |
| MIO Throttle | Memory I/O pipes saturated | Vectorize loads, reduce traffic |
| LG Throttle | Local / Global pipe throttled | Same — widen loads, remove local mem spills |
| Tex Throttle | Texture / LDG pipe saturated | `ldg.128`, vectorize |
| IMC Miss | I-cache miss | Reduce code size in inner loop, watch unroll factor |
| Math Pipe Throttle | FP / integer ALU saturated (you're close to compute-bound) | Re-check compute %, may be higher than SOL says |
| Not Selected | Warp was ready; scheduler picked a different one | Usually means enough occupancy; check other stalls |
| Selected (never stalled) | Fine, there's just something faster | Not actionable |

If the top stall is one of the "throttle" categories, treat the kernel as pipe-bound even if SOL numbers are middling — you're saturating a narrower pipe than the top-level SOL.

## Step D — When bandwidth or compute is the binding side

Drill into the next level:

**Bandwidth-bound side checks:**
- L1/TEX hit rate — if low, many requests are going to L2 or DRAM. Could re-use in shared memory cut it?
- L2 hit rate — if low, tiles aren't being reused across blocks. Could block schedule / swizzle help?
- DRAM throughput % alone vs DRAM bytes/cycle — are you actually using the bus width?
- Sector coalesce — are 32B sectors being read as 4B scattered? Classic uncoalesced access.

**Compute-bound side checks:**
- Tensor core utilization (`sm__inst_executed_pipe_tensor.*`) — if 0 and the op is matmul-shaped, you're leaving a 10–30× multiplier on the table.
- FMA / ALU pipe utilization — which specific pipe is saturated?
- Instructions per cycle — below 1 IPC while nominally compute-bound usually means the compiler is stalling on dependency chains.

## Step E — Occupancy-limited? Find the reason

Occupancy is limited by *exactly one* of:

- **Registers per thread.** Check the "Launch Statistics" → "Registers Per Thread." If this * block size > registers per SM * max blocks, you lose occupancy. Fix: `__launch_bounds__`, smaller tiles, factor out helpers.
- **Shared memory per block.** Check "Static Shared Memory" + "Dynamic Shared Memory." If per-block smem * max blocks > smem per SM, you lose occupancy. Fix: smaller tiles, compressed layouts.
- **Block size.** If threads/block is too large (e.g., 1024), only 2 blocks fit per SM even with modest resources. Try 128 or 256.
- **Warps per block.** Similar.

NCU's "Occupancy" section will explicitly tell you "occupancy is limited by registers" or similar. Trust that.

## Step F — Latency-bound means "enough warps, just not ready"

If occupancy > 60% and throughputs are middling, warps are in flight but frequently stalled. Look for:
- Long dependency chains in the inner loop. Can you interleave independent work?
- Missing software pipelining. Are you overlapping loads with compute?
- Excessive `__syncthreads`. Can you split work so warps can proceed independently?
- Producer/consumer imbalance. Warp specialization (some warps load, others compute) can help.

---

## Multi-workload case

If the kernel is called with different input shapes and you're looking at an aggregate profile, the numbers are averages. Individual workloads may be in completely different regimes. Two options:

1. Profile each outlier workload separately (`ncu --launch-skip N --launch-count 1`). Classify each.
2. If you can't isolate, use the dominant workload. If the dominant one is compute-bound but the 2-token case is clearly occupancy-limited, your kernel needs adaptive dispatch (see SKILL.md Step 4).

## Worked examples

### Example 1: Classic memory-bound gemm

```
SM %: 38%     Memory %: 82%     Occupancy: 52%
Top stall: Long Scoreboard (41%)
Tensor core util: 0%
```

Classification: **bandwidth-bound**, with an obvious tensor-core lift available once memory is sorted. Matrix shape is matmul-like, so the long-term winning path is probably tensor cores + async copy + double buffer. Short-term: vectorize the loads, swizzle smem to kill bank conflicts.

### Example 2: Flash attention kernel, post-split-K

```
SM %: 68%     Memory %: 64%     Occupancy: 58%
Top stall: Short Scoreboard (22%), Wait (19%)
```

Classification: **balanced**, leaning compute. Close to simultaneously hitting both ceilings. Next moves are small — remove a `__syncthreads()` if possible, hoist invariants out of the inner loop, try warp-level reduction for the softmax max/sum step.

### Example 3: The 2-token outlier

```
SM %: 8%     Memory %: 12%     Occupancy: 6.21%
Top stall: Not Selected (huge)
```

Classification: **occupancy-limited, catastrophically.** Only 16 CTAs launching for 2 tokens × 16 heads / 2 (one head group) — not enough to fill 148 SMs. Fix: for this workload, increase Split-K so more CTAs launch. This is the "adaptive dispatch" move — the kernel needs a different code path for small batch sizes.

### Example 4: FuseMoE long-seq GEMM, classifying at the wall

```
SM %: 50.7%      Memory %: 24.9% (DRAM) / 44.1% (SOL composite)
Occupancy: 14.1%
Top stall: smem/regs blocking next-CTA (not Warp State Stats)
```

This is **compute-bound at the library's occupancy ceiling**, *not* memory-bound. Easy to misread because the SOL "Memory Throughput" composite (44.1%) is higher than DRAM-specific (24.9%) and looks alarming. The classification step that matters: at 14% achieved occupancy and 168 reg/thread + 218 KB shmem per block, the SM is full — it just can only fit one CTA. SM throughput (50.7%) is what that 1 CTA can deliver; DRAM has slack (24.9%) because there isn't a second consumer to pull bytes.

Common mis-classification: reading "SM Active Cycles 97%" (cycles where *any* warp is issuing) and concluding "compute-bound at 97%". That metric is not the SM throughput — it's the duty cycle, and it conflates productive issue with stalls. Always read `sm__throughput.avg.pct_of_peak_sustained_elapsed` for the SOL number that drives strategy.

Next move: the **Plateau → Hand-write tcgen05** option in `strategy-matrix.md`. TMEM accumulator relaxes the register pressure that pins this kernel to 1 CTA/SM. See Example E in `strategy-matrix.md` for the full case study.
