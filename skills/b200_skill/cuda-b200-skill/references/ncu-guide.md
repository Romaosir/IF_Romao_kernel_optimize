# ncu workflow notes

Use `ncu` only after `nsys` identifies the kernel of interest.

## Basic flow

```bash
ncu --set basic ./kernel
ncu --kernel-name "<kernel_name>" --set full ./kernel
```

## Questions `ncu` should answer

- Is occupancy limiting performance?
- Is memory throughput near hardware limits?
- Are accesses coalesced?
- What are the dominant stall reasons?
- Is register pressure too high?
- Is shared memory helping or hurting?
- Are TensorCore instructions actually being used?

## Rule

Make one optimization hypothesis from `ncu`, change one thing, and re-measure.
