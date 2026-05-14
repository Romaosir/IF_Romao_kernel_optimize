# Zero-Sync Fast Path (+16.3%)

Eliminate `cudaStreamSynchronize` by moving all metadata construction to the GPU. Single largest observed gain in practice after the cuBLAS→CUTLASS swap.

## The problem

Original pipeline (before zero-sync):

```
[GPU] routing kernel        ─┐
[GPU] expert counts to GPU   │
[GPU] D2H memcpy of counts   │
                              ▼
[CPU] cudaStreamSynchronize          ← stall, ~50µs
[CPU] prefix scan on counts
[CPU] sort active experts by count
[CPU] build CUTLASS argument arrays
[CPU] H2D memcpy of argument arrays  ← ~10µs
                              ▲
[GPU] gather kernel         ─┘
[GPU] quantize kernel
[GPU] CUTLASS FP8 grouped GEMM
```

The `cudaStreamSynchronize` alone is ~50µs. For small-T workloads (seq_len ≤ 16), that's ~37% of total runtime.

## The fix

Move **every piece of metadata** to GPU kernels. Chain them with `programmaticStreamSerialization` to avoid host round-trips.

```
[GPU] routing kernel
[GPU] prefix scan kernel          (new — was CPU)
[GPU] expert sort kernel          (new — was CPU)
[GPU] CUTLASS argument setup      (new — was CPU, now in prep kernel)
[GPU] gather kernel
[GPU] quantize kernel
[GPU] CUTLASS FP8 grouped GEMM

All chained with PSS — no cudaStreamSynchronize anywhere.
```

## PSS launch pattern

Every kernel in the chain uses `cudaLaunchAttributeProgrammaticStreamSerialization`:

```cpp
cudaLaunchConfig_t cfg = {};
cfg.gridDim = compute_grid(n);
cfg.blockDim = 256;
cfg.dynamicSmemBytes = 0;
cfg.stream = stream;

cudaLaunchAttribute attr[1] = {};
attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attr[0].val.programmaticStreamSerializationAllowed = true;
cfg.numAttrs = 1;
cfg.attrs = attr;

cudaLaunchKernelEx(&cfg, my_kernel, arg1, arg2, ...);
```

Every consumer kernel in the chain pairs this with `griddepcontrol` PTX at the start:

```cpp
__global__ void my_kernel(...) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;");           // wait for predecessor grid
    asm volatile("griddepcontrol.launch_dependents;"); // allow next grid to start
#endif

    // ... kernel body ...
}
```

The combination gives **grid-level dependency**: the consumer's grid doesn't start until the producer's grid completes, but there's no host round-trip.

## Example: GPU-side active expert sort

Before (CPU):

```cpp
// Host code
cudaMemcpyAsync(h_counts, d_counts, sizeof(int) * num_experts,
                cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);                      // <<< STALL

std::vector<int> sorted_eids(num_experts);
std::iota(sorted_eids.begin(), sorted_eids.end(), 0);
std::sort(sorted_eids.begin(), sorted_eids.end(),
          [&](int a, int b) { return h_counts[a] > h_counts[b]; });

int active_count = std::count_if(h_counts, h_counts + num_experts,
                                 [](int c) { return c > 0; });

cudaMemcpyAsync(d_active_eids, sorted_eids.data(),
                sizeof(int) * active_count,
                cudaMemcpyHostToDevice, stream);
```

After — specification (write a GPU kernel):

A single-block kernel (at most one CTA) that:

1. Enters with the `griddepcontrol.wait; griddepcontrol.launch_dependents;` PTX pair so it chains after the routing kernel without host sync
2. Streams through `counts[]`, filtering `counts[expert] > 0` via `atomicAdd` to a shared counter — produces a compact array of active expert IDs
3. Sorts that array by `counts[expert]` descending in shared memory — for typical `num_experts ≤ 256` a single-thread insertion sort is enough; for larger, fall back to block-level bitonic sort
4. Writes the sorted IDs to `active_eids[]` and the active count to `active_count_out[]` in global memory

The key invariants:

- Use `__syncthreads()` after the atomic-append phase and before the sort
- The sort must be deterministic — don't break ties by thread ID or arrival order, tie on expert-ID to ensure reruns produce identical output (correctness tests compare bit-exact)
- Don't read `active_count_out` from the host — do all downstream work on the GPU (see "`active_count` special case" below)

The original implementation fit in ~40 lines of CUDA; the logic is straightforward once the invariants are clear.

Launch with the same PSS attribute:

```cpp
cudaLaunchConfig_t cfg = {};
cfg.gridDim  = 1;
cfg.blockDim = 256;
cfg.stream   = stream;
// + PSS attribute as shown above
cudaLaunchKernelEx(&cfg, sort_active_experts_kernel, d_counts,
                   d_active_eids, d_active_count_out, num_experts);

```

The following gather kernel also uses `griddepcontrol.wait` and automatically blocks until this sort is done.

## `active_count` — the special case

Note that `active_count_out` is an int produced on the GPU. If any downstream kernel needs to dimension its grid based on this value *and* be launched from the host, you've reintroduced a sync. The solution is one of:

1. **Upper bound the grid** — launch for `num_experts` (the max), have each block check if it's in the active range
2. **Device-side kernel launch** — launch downstream kernels from the GPU via dynamic parallelism (high overhead, not usually worth it)
3. **Skip-on-zero** — have each block check if its assigned expert has zero tokens and exit early

In practice, option 3 works for the gather kernel and option 1 for the CUTLASS prep kernel. Neither requires a host-side read of `active_count`.

## Metadata fusion

Once zero-sync is done, adjacent small metadata kernels become candidates for fusion:

```cpp
// Before: two kernels
prefix_scan_kernel<<<...>>>(d_counts, d_offsets, num_experts);
build_row_mapping_kernel<<<...>>>(d_offsets, d_row_to_expert, total_rows);

// After: fused
__global__ void scan_and_map_kernel(
    const int* counts, int num_experts,
    int* offsets, int* row_to_expert
) {
    // Step 1: prefix scan in shared memory
    __shared__ int s_off[256];
    int tid = threadIdx.x;
    if (tid < num_experts) s_off[tid] = counts[tid];
    __syncthreads();

    // ... scan logic ...

    // Step 2: immediate row-to-expert mapping from the just-computed offsets
    for (int r = tid; r < total_rows; r += blockDim.x) {
        // binary search r in s_off to find which expert it belongs to
        // ... mapping logic ...
        row_to_expert[r] = which_expert;
    }
}
```

This eliminates one kernel launch per invocation. Multiple such fusions combined gave the **+5.8%** from "threadfence removal + metadata fusion" (ladder item 6).

## Measurement

Where this change matters most is short-sequence workloads (small T), where the CPU stall is a larger fraction of end-to-end runtime. In practice:

- Very short sequences (seq_len = 1): +40% range
- Short sequences (seq_len ≈ 16): +25% range
- Geometric mean across a mixed workload: **+16.3%**

Large-T workloads see smaller gains because the CPU stall is a smaller fraction of their runtime.

## Pitfalls

| Pitfall | Symptom |
|---|---|
| Forgot `griddepcontrol.wait` in a consumer | Consumer runs before producer finishes, produces wrong results |
| Forgot `griddepcontrol.launch_dependents` in producer | Next kernel stalls waiting for grid exit |
| Missed one `cudaStreamSynchronize` | Entire chain breaks — find and remove all syncs |
| Relied on `active_count` to size a host-launched grid | Re-introduces sync; use upper bound or skip-on-zero instead |
| Wrong PTX guard | Build fails on older CUDA or SM; guard with `__CUDA_ARCH__ >= 900` |
| Order-sensitive atomicAdd in sort | Different runs produce different order; sort must be deterministic for correctness tests |

## NCU verification

After the change:
- No `Host Wait` periods visible in the timeline
- Kernels chain back-to-back (no gaps > 1µs between them)
- `cudaStreamSynchronize` doesn't appear in CUDA API profile
- Total kernel launch count may decrease (if you fused metadata kernels too)

If you still see a host stall somewhere, search for:
- Implicit `cudaFree` / `cudaMalloc` (synchronous by default)
- `cudaMemcpy` without the `*Async` variant
- `cublasHandle` creation (synchronous)
- Any `.cpu()` or `.item()` on PyTorch tensors in the wrapper
