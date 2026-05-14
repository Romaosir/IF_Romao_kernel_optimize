# Dual-Tile CUTLASS Dispatch (+13%)

Dispatch between 64×128×128 and 128×128×128 tiles based on estimated per-expert M. Handles the small-M / large-M split without tuning for one case at the cost of the other.

## Rule

```
max_M_estimate = total_rows / num_active_experts + 1

if max_M_estimate > 256:
    → use 128-tile variant
else:
    → use 64-tile variant
```

## The prep kernel — specification (write this yourself)

A small device kernel that builds the per-group argument arrays CUTLASS grouped GEMM expects. Runs once per invocation before the GEMM, one thread per group.

### Types to pull from the CUTLASS kernel definition

- `ProblemShape::UnderlyingProblemShape`
- `StrideA`, `StrideB`, `StrideD`
- `InternalLayoutSFA`, `InternalLayoutSFB`

### Inputs

- `A` (FP8), `B` (FP8, per-expert), `SFA` (FP32 per-row block scales), `SFB` (FP32 per-expert per-block scales), `D` (BF16 output)
- `m_indptr[num_groups + 1]` — prefix sum of per-group row counts (so group `i` spans rows `[m_indptr[i], m_indptr[i+1])`)
- `expert_ids[num_groups]` — which weight slice each group consumes
- Dims: `N`, `K`, `num_groups`

### Outputs (one entry per group)

- Problem shape `(m, N, K)` where `m = m_indptr[i+1] - m_indptr[i]`
- Stride for A, B, D — build with `cutlass::make_cute_packed_stride(StA{}, {m, K, 1})` and the analogous calls for B / D
- Pointers:
  - `pA[i] = A + row_offset * K` (per-expert rows are contiguous)
  - `pB[i] = B + expert_id * N * K` (expert-indexed weights)
  - `pD[i] = D + row_offset * N`
  - `pSFA[i] = SFA + row_offset * (K / 128)`
  - `pSFB[i] = SFB + expert_id * (N / 128) * (K / 128)`
- Scale layouts: `tile_atom_to_shape_SFA(make_shape(m, N, K, 1))` and the SFB equivalent

### Launch

- Grid: `ceil(num_groups / 256)` blocks × 256 threads
- Launch attribute: `cudaLaunchAttributeProgrammaticStreamSerialization = true`
- PTX at entry: `griddepcontrol.wait;` followed by `griddepcontrol.launch_dependents;` — this lets the CUTLASS GEMM launch immediately after prep completes, without a host round-trip

### Pitfalls

- Order of `griddepcontrol.wait` vs any memory op — wait must come first
- Use `int64_t` for offsets — `row_offset * K` can overflow 32-bit for large workloads
- `SFA` is per-row per-K-block; `SFB` is per-expert per-(N-block × K-block). Getting the strides swapped is a classic silent-correctness bug

## The two tile variants

Two CUTLASS collectives, same prep kernel, same caller signature. Different tile shapes compiled into different collective types.

```cpp
using Gm64 = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassBlockScaledTensorOp,
    EA, LayoutA,    16,
    EB, LayoutB,    16,
    EAcc,
    Shape<_64,  _128, _128>,          // M=64 tile
    Shape<_1, _1, _1>,                // 1-SM cluster
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(EC)>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100
>::CollectiveOp;

using Gm128 = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassBlockScaledTensorOp,
    EA, LayoutA,    16,
    EB, LayoutB,    16,
    EAcc,
    Shape<_128, _128, _128>,          // M=128 tile
    Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(EC)>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100
>::CollectiveOp;
```

Each collective is wrapped in a `GemmUniversal` kernel and exposed as an `extern "C"` function so the main kernel can `dlsym` it:

```cpp
## Wrapper — structure

Each tile variant is exposed as an `extern "C"` function that callers dlsym from the JIT-compiled `.so` (see `cutlass-jit-compile.md`). The wrapper does the following, roughly in order:

1. **One-time init** — query `cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0)` once per process
2. **Grow device argument arrays** if the group count has increased since last call (allocate-and-remember pattern, never shrink)
3. **Launch the prep kernel** with `cudaLaunchAttributeProgrammaticStreamSerialization = true`
4. **Build `Gm::Arguments`** in `GemmUniversalMode::kGrouped`, pointing at the arrays prep just populated
5. **Set `args.scheduler.max_swizzle_size = 4`** — measured sweet spot (see `tuning-knobs.md`)
6. **Cache `can_implement()` and `get_workspace_size()` results** behind a `static bool s_validated` — both are O(ms) first call, so re-running each invocation is expensive
7. **`g_gemm.initialize(args, workspace, stream)` → `g_gemm.run(stream)`** — return non-zero error codes on failure so the caller can fall back to a different path

Compile two of these wrappers, one per tile variant (`Gm64` and `Gm128`), exported under distinct symbol names.

## Dispatch from the main kernel — rule

```
max_M_estimate = total_rows / num_active_experts + 1

pick 64-tile wrapper if max_M_estimate ≤ 256, else 128-tile wrapper
if the chosen wrapper returns non-zero, fall back to the 64-tile wrapper
```

The fallback matters: the 128-tile variant can fail `can_implement()` for certain small-M shapes, in which case 64-tile still works.

## Observed result

On typical MoE workloads: **+13%** gain. Single largest CUTLASS-internal optimization observed in practice.

## Pitfalls

| Pitfall | Symptom |
|---|---|
| Threshold tuned to wrong workload | One tile variant wins on your dataset; verify with measurement |
| Tried 256-tile variant | `−18%` from wave quantization on 2-SM cluster |
| Tried 128×256 or 64×256 N-tile | `−60%` from wide-N wave quantization |
| Only compiled one variant | Dispatch falls through with no alternative |
| `can_implement` called every invocation | Significant overhead; cache the result |

## Why the threshold is 256

- Below max_M=256: the 64-tile produces enough tiles to fill the GPU's 148 SMs; each tile does real work
- Above 256: the 64-tile produces *too many* tiles, and per-tile overhead (TMA descriptor setup, scheduler bookkeeping) dominates
- The 128-tile amortizes that overhead by doing more work per tile

Measured on MoE workloads with variable expert token counts; threshold may shift by ~50 depending on distribution. Verify with measurement before changing.
