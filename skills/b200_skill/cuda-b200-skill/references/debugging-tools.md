# Debugging tools workflow

Use these tools before performance tuning if correctness is uncertain.

## Sanitizers

```bash
compute-sanitizer --tool memcheck ./kernel_debug
compute-sanitizer --tool racecheck ./kernel_debug
compute-sanitizer --tool synccheck ./kernel_debug
compute-sanitizer --tool initcheck ./kernel_debug
```

## Device-side printf

Prefer small and guarded prints:

```cuda
if (blockIdx.x == 0 && threadIdx.x < 4) {
    printf("tid=%d idx=%d value=%f\n", threadIdx.x, idx, value);
}
```

## PTX/SASS inspection

```bash
cuobjdump -ptx ./kernel > kernel.ptx
cuobjdump -sass ./kernel > kernel.sass
cuobjdump -res-usage ./kernel
```

Inspect:
- register count
- shared memory usage
- evidence of spills
- expected instruction selection
- unintended scalarization or predication
