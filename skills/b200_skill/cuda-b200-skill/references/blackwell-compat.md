# Blackwell compatibility and build-target notes

Use this file when deciding which nvcc target to emit and what compatibility guarantees apply.

## Main rule

Separate these cases:

- `sm_100` / `compute_100`: generic Blackwell target
- `sm_100a` / `compute_100a`: architecture-specific Blackwell target

## Recommended default

For ordinary B200 kernels, prefer:

```bash
nvcc -O3 -lineinfo -arch=sm_100 kernel.cu -o kernel
```

## Architecture-specific target

Use only when the kernel depends on architecture-specific Blackwell accelerated features:

```bash
nvcc -O3 -lineinfo --generate-code arch=compute_100a,code=sm_100a kernel.cu -o kernel
```

## PTX inclusion

When compatibility and deployability matter, include PTX as well:

```bash
nvcc -O3 -lineinfo \
  --generate-code arch=compute_100,code=sm_100 \
  --generate-code arch=compute_100,code=compute_100 \
  kernel.cu -o kernel
```

## Important cautions

- Do not confuse Hopper-specific `sm_90a` guidance with Blackwell-specific `sm_100a` guidance.
- Code that depends on architecture-specific accelerated features does not enjoy the same compatibility expectations as generic PTX.
- Use `sm_100a` only when you need it; otherwise prefer `sm_100`.
- If the toolchain is too old to emit native Blackwell cubins, PTX fallback matters more.

## Decision rule

Choose `sm_100` unless all of the following are true:
- you are intentionally using architecture-specific Blackwell features
- you know the deployment fleet supports the target
- the speedup is measured and meaningful
