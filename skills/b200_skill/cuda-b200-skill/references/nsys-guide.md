# nsys workflow notes

Use `nsys` first to decide whether a kernel matters enough to optimize.

## Basic flow

```bash
nsys profile -o report ./kernel
nsys stats report.nsys-rep --report cuda_gpu_kern_sum
```

## Questions `nsys` should answer

- Which kernels dominate runtime?
- Is the GPU being starved by the CPU?
- Are launches too frequent or too small?
- Are there memcpy gaps or synchronization stalls?
- Is there overlap between compute and transfers?

## Rule

Do not do deep kernel micro-optimization until `nsys` shows the kernel is materially important.
