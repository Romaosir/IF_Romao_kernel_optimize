# Blackwell tuning notes for B200

Use this file for architecture-specific decisions that affect CUDA kernel structure on B200.

## Facts to rely on

- B200 / Blackwell datacenter GPU is compute capability 10.0.
- Unified L1 / texture / shared-memory capacity is 256 KB per SM.
- Shared-memory carveout options per SM include: 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB.
- CUDA reserves 1 KB shared memory per thread block, so the maximum addressable shared memory for a block is about 227 KB.
- Portable thread-block cluster size is 8.
- B200 supports nonportable cluster size 16 with opt-in.

## What to do with these facts

### Shared memory
Ask:
- does this kernel benefit from larger tiles enough to justify lower occupancy?
- does dynamic shared memory force too few active blocks per SM?
- should carveout be tuned via `cudaFuncAttributePreferredSharedMemoryCarveout`?
- if using very large dynamic shared memory, was opt-in handled correctly?

### Clusters
Consider cluster launch only when:
- the kernel has meaningful inter-block cooperation
- data reuse or synchronization benefit is measurable
- occupancy loss is acceptable
- portability cost is acceptable

### Occupancy tradeoff
Do not assume maximum occupancy is optimal.
On B200, large tiles and larger shared-memory allocations may still win if they materially improve reuse, TensorCore feed rate, or memory efficiency.

### Blackwell-specific fast paths
When considering Blackwell-specific instructions or Tensor Memory paths:
- use `sm_100a` only if the code path truly depends on architecture-specific features
- otherwise prefer `sm_100` for broader compatibility

## Review questions

- Is this kernel memory-bound, latency-bound, or occupancy-limited?
- Is shared memory actually reducing global traffic enough?
- Did cluster launch help wall time, not just local counters?
- Did the chosen tile shape improve effective throughput on real inputs?
