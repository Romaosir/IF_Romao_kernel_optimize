# Using the local CUDA Runtime API docs

Prefer Runtime API references for normal CUDA application and kernel-launch behavior.

## Use them for

- `cudaError_t`
- `cudaDeviceProp`
- launch APIs
- streams/events/graphs
- memory allocation and async allocation
- occupancy APIs
- kernel attributes
- shared-memory carveout and dynamic shared-memory opt-in

## Search patterns

```bash
grep -R "cudaFuncSetAttribute" references/cuda-runtime-docs/
grep -R "cudaFuncAttributePreferredSharedMemoryCarveout" references/cuda-runtime-docs/
grep -R "cudaFuncAttributeNonPortableClusterSizeAllowed" references/cuda-runtime-docs/
grep -R "cudaDeviceProp" references/cuda-runtime-docs/
grep -R "cudaError" references/cuda-runtime-docs/
```

## Typical questions

- How do I opt in to larger dynamic shared memory?
- What does this runtime error code mean?
- Which kernel attributes can be set?
- What occupancy helper API should I use?
