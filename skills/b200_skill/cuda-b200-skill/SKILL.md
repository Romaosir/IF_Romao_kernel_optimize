---
name: cuda-b200
description: CUDA kernel development and optimization for NVIDIA B200 (Blackwell, compute capability 10.0) in Claude Code. Use when writing, reviewing, debugging, profiling, or optimizing CUDA kernels for B200/Blackwell. Covers kernel structure, memory hierarchy, shared memory and cluster constraints, PTX/TensorCore inspection, nsys/ncu profiling, compute-sanitizer debugging, and use of local PTX/Runtime/Driver reference docs. Prefer this skill for CUDA, PTX, Tensor Core, WGMMA/TMA/tcgen05, shared memory, occupancy, coalescing, register pressure, launch configuration, and Blackwell-specific tuning.
---

# CUDA B200 / Blackwell Skill

## Mission

Produce CUDA kernels that are:

1. **Correct first**
2. **Measured before optimized**
3. **Explicitly targeted for NVIDIA B200 / Blackwell**
4. **Grounded in references, not guessed from memory**

This skill is for **kernel authoring and optimization on NVIDIA B200 (compute capability 10.0)**.
When architecture-specific features are involved, distinguish carefully between:

- **generic Blackwell target**: `sm_100`
- **architecture-specific Blackwell target**: `sm_100a` / `compute_100a`

Do **not** assume Hopper (`sm_90` / `sm_90a`) guidance transfers unchanged to B200.

---

## Core Operating Rules

### 1. Correctness beats premature optimization
Before tuning:
- verify indexing
- verify bounds
- verify synchronization
- verify launch configuration
- verify shared memory sizing
- verify numerical expectations

Use:
- `compute-sanitizer --tool memcheck`
- `compute-sanitizer --tool racecheck`
- `compute-sanitizer --tool synccheck`
- `compute-sanitizer --tool initcheck`

For kernel bugs, prefer:
1. smallest reproducible input
2. targeted device-side `printf`
3. sanitizer
4. source + PTX/SASS inspection

### 2. Profile before changing code
Optimization loop:
1. establish baseline
2. profile with **nsys** to find important kernels
3. profile target kernel with **ncu**
4. form one concrete hypothesis
5. change one thing
6. re-measure
7. keep or revert

Never optimize based only on intuition.

### 3. Trust references over memory
This skill expects searchable local documentation under:
- `references/ptx-docs/`
- `references/cuda-runtime-docs/`
- `references/cuda-driver-docs/`

Treat these as **fast local references**.
For Blackwell-specific architecture behavior, compilation targets, compatibility, and newest toolchain details, cross-check official online docs listed in the Reference section.

### 4. Prefer Runtime API unless lower-level control is needed
For most kernel development:
- use **CUDA Runtime API** first

Use **CUDA Driver API** only when you specifically need:
- explicit context management
- dynamic PTX/CUBIN/module loading
- virtual memory management
- lower-level tool/framework control

---

## B200 / Blackwell Facts That Must Shape Decisions

### Architecture target
NVIDIA **B200** is **compute capability 10.0**.

### Shared memory / L1 / texture
B200 has a unified **L1 / texture / shared memory** structure with total capacity **256 KB per SM**.

Supported shared-memory carveout capacities per SM include:
`0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB`.

CUDA reserves **1 KB** of shared memory per thread block, so a single block can address up to about **227 KB** of shared memory on B200 when configured correctly.

Implications:
- dynamic shared memory sizing matters
- carveout tuning matters
- large-tile kernels must consider occupancy tradeoffs
- always inspect occupancy and active blocks after shared-memory changes

### Cluster size
Portable cluster size is **8**.
B200 supports nonportable cluster size **16** if you opt in with:
- `cudaFuncAttributeNonPortableClusterSizeAllowed`

Use this only when measured benefit outweighs occupancy and portability costs.

### Architecture-specific features
Some Hopper/Blackwell architecture-accelerated features are **not** covered by normal PTX forward-compatibility guarantees.

If your kernel uses Blackwell-specific accelerated features, use the appropriate target:
- `sm_100a` / `compute_100a`

Do not assume code built for `compute_90a` works on Blackwell.

---

## When to Use Which Reference

### `references/ptx-docs/`
Use for:
- PTX instruction syntax
- inline PTX
- WGMMA
- TMA / tensor map details
- `mbarrier`
- `tcgen05.*`
- swizzling
- memory consistency model
- register fragment layouts
- understanding compiler-generated PTX

Typical questions:
- “What is the operand layout for this PTX instruction?”
- “What fragment layout does this WGMMA shape use?”
- “How does `tcgen05.mma` work?”
- “What are the legal swizzle / layout combinations?”

### `references/cuda-runtime-docs/`
Use for:
- `cudaError_t`
- `cudaFuncSetAttribute`
- `cudaDeviceProp`
- streams/events/graphs
- memory allocation APIs
- async memory ops
- occupancy APIs
- launch configuration behavior

Typical questions:
- “How do I enable larger dynamic shared memory?”
- “What does this runtime error mean?”
- “What fields exist in `cudaDeviceProp`?”
- “What are the synchronization semantics of this API?”

### `references/cuda-driver-docs/`
Use only when needed for:
- `CUresult`
- `cuCtxCreate`
- `cuModuleLoad*`
- `cuModuleGetFunction`
- `cuMemMap` / `cuMemCreate`
- virtual memory
- advanced control paths

Typical questions:
- “How do I load PTX/cubin manually?”
- “How does Driver API virtual memory work?”
- “What is this `CUDA_ERROR_*` value?”

---

## B200 Kernel Review Checklist

When reviewing or generating a kernel, check these in order:

### Correctness
- Are global memory accesses in-bounds?
- Is shared memory sized correctly?
- Is dynamic shared memory passed correctly at launch?
- Are all required barriers present?
- Is warp-level or cluster-level synchronization correct?
- Are edge tiles / tails handled?
- Are alignment assumptions valid?
- Are pointer aliasing assumptions explicit?

### Mapping and occupancy
- Does thread/block mapping match data layout?
- Is block size justified?
- Is register usage too high?
- Is shared memory limiting occupancy?
- Would a different tile shape improve active warps/SM?

### Memory behavior
- Are global accesses coalesced?
- Is shared memory access bank-conflict aware?
- Is data reused enough to justify shared memory?
- Are vectorized loads/stores safe and aligned?
- Are async copy / TMA opportunities real and measurable?

### Blackwell-specific opportunities
- Would `sm_100a` unlock an architecture-specific fast path?
- Is this actually a candidate for `tcgen05.*` / Tensor Memory?
- Would larger shared memory carveout help more than it hurts occupancy?
- Would cluster launch help?
- If using cluster launch, is the portability tradeoff acceptable?

### Tool validation
- `compute-sanitizer` clean?
- `nsys` confirms kernel matters?
- `ncu` confirms actual bottleneck?
- PTX/SASS matches intended codegen?

---

## Default Workflow for Claude

When asked to write or optimize a B200 kernel, follow this sequence:

1. **State the target**
   - Assume B200 / Blackwell / cc 10.0 unless user specifies otherwise.

2. **Clarify compilation target internally**
   - Use `sm_100` by default.
   - Upgrade to `sm_100a` only when architecture-specific Blackwell features are explicitly needed.

3. **Draft a simple correct kernel first**
   - prioritize correctness, clear indexing, and predictable memory behavior

4. **Audit the draft**
   - bounds
   - synchronization
   - shared memory sizing
   - launch shape
   - edge handling

5. **Explain likely bottleneck category**
   - memory-bound
   - latency-bound
   - instruction/pipe-bound
   - occupancy-limited
   - sync-limited

6. **Suggest measurement plan**
   - `nsys` for hotspot discovery
   - `ncu` for detailed kernel diagnosis
   - `compute-sanitizer` for correctness if needed

7. **Only then propose advanced changes**
   - tile reshaping
   - vectorization
   - shared-memory staging
   - carveout tuning
   - warp specialization
   - cluster launch
   - TensorCore / PTX path

---

## Compilation Guidance

### Baseline B200 build
```bash
nvcc -O3 -lineinfo -arch=sm_100 kernel.cu -o kernel
```

### Blackwell architecture-specific build
Use only when you intentionally rely on architecture-specific Blackwell features:
```bash
nvcc -O3 -lineinfo --generate-code arch=compute_100a,code=sm_100a kernel.cu -o kernel
```

### Debug build
```bash
nvcc -O0 -g -G -lineinfo -arch=sm_100 kernel.cu -o kernel_debug
```

### Include PTX for compatibility / inspection
```bash
nvcc -O3 -lineinfo \
  --generate-code arch=compute_100,code=sm_100 \
  --generate-code arch=compute_100,code=compute_100 \
  kernel.cu -o kernel
```

### Notes
- Always include `-lineinfo` for profiling builds.
- For B200 deployment, prefer toolchains new enough to understand native Blackwell code generation.
- If using CUDA 12.7 or earlier, native Blackwell cubin generation is not available; PTX inclusion becomes important for compatibility.
- If using architecture-specific Blackwell features, target `compute_100a` / `sm_100a`.

---

## Debugging Workflow

### First pass
```bash
compute-sanitizer --tool memcheck ./kernel_debug
compute-sanitizer --tool racecheck ./kernel_debug
compute-sanitizer --tool synccheck ./kernel_debug
compute-sanitizer --tool initcheck ./kernel_debug
```

### Device-side tracing
Use small, guarded prints:
```cuda
if (blockIdx.x == 0 && threadIdx.x < 4) {
    printf("tid=%d idx=%d value=%f\n", threadIdx.x, idx, value);
}
```

### Binary inspection
```bash
cuobjdump -ptx ./kernel > kernel.ptx
cuobjdump -sass ./kernel > kernel.sass
cuobjdump -res-usage ./kernel
```

Inspect:
- register count
- shared memory usage
- whether intended instructions actually appear
- whether codegen is unexpectedly scalarized / spilled / predicated

---

## Profiling Workflow

### System-level hotspot discovery
```bash
nsys profile -o report ./kernel
nsys stats report.nsys-rep --report cuda_gpu_kern_sum
```

Use `nsys` to answer:
- which kernels dominate time?
- is launch overhead significant?
- is CPU starving the GPU?
- are there memcpy or synchronization gaps?

### Kernel-level diagnosis
```bash
ncu --set basic ./kernel
ncu --kernel-name "<kernel_name>" --set full ./kernel
```

Use `ncu` to answer:
- occupancy-limited or not?
- memory throughput vs theoretical?
- coalescing quality?
- stall reasons?
- register pressure?
- shared-memory bottlenecks?
- TensorCore utilization, if applicable?

---

## PTX / TensorCore / Blackwell-Specific Guidance

### When PTX inspection is worth it
Inspect PTX/SASS when:
- compiler output seems suspicious
- inline PTX is involved
- TensorCore instructions should appear but do not
- async/TMA codegen is unclear
- architecture-specific features are being used

### WGMMA vs tcgen05
- **WGMMA** is central for Hopper-era warpgroup MMA work
- **tcgen05** is the Blackwell TensorCore 5th-generation family path

For B200-specific fast paths, look first in:
- `references/ptx-docs/9-instruction-set/9.7.16-*`

### Do not guess instruction details
For any of the following, consult PTX docs directly:
- operand order
- fragment layout
- descriptor layout
- memory ordering requirements
- swizzle legality
- barrier / wait semantics
- packing / unpacking rules

---

## Search Shortcuts

### PTX
```bash
grep -R "wgmma" references/ptx-docs/9-instruction-set/
grep -R "tcgen05" references/ptx-docs/9-instruction-set/
grep -R "mbarrier" references/ptx-docs/
grep -R "swizzle" references/ptx-docs/
grep -R "Tensor Memory" references/ptx-docs/9-instruction-set/
```

### Runtime API
```bash
grep -R "cudaFuncSetAttribute" references/cuda-runtime-docs/
grep -R "cudaFuncAttributePreferredSharedMemoryCarveout" references/cuda-runtime-docs/
grep -R "cudaFuncAttributeNonPortableClusterSizeAllowed" references/cuda-runtime-docs/
grep -R "cudaDeviceProp" references/cuda-runtime-docs/
grep -R "cudaError" references/cuda-runtime-docs/
```

### Driver API
```bash
grep -R "cuModuleLoad" references/cuda-driver-docs/
grep -R "cuCtxCreate" references/cuda-driver-docs/
grep -R "cuMemMap" references/cuda-driver-docs/
grep -R "CUDA_ERROR_" references/cuda-driver-docs/
```

---

## Output Style for Claude

When answering, prefer this structure:

1. **Goal**
2. **Likely bottleneck or risk**
3. **Recommended kernel or code change**
4. **Why it should help on B200**
5. **What to measure next**
6. **Which reference supports the claim**

Be explicit about uncertainty.
If a claim depends on PTX or API behavior, cite the local reference path.
If a claim depends on Blackwell architecture behavior or target compatibility, cite the official Blackwell docs.

---

## Reference

### Local references in this skill
- `references/ptx-docs/`
- `references/cuda-runtime-docs/`
- `references/cuda-driver-docs/`
- `references/ptx-isa.md`
- `references/cuda-runtime.md`
- `references/cuda-driver.md`
- `references/blackwell-tuning.md`
- `references/blackwell-compat.md`
- `references/nsys-guide.md`
- `references/ncu-guide.md`
- `references/debugging-tools.md`

### Official online references to trust for Blackwell-specific details
- NVIDIA Blackwell Tuning Guide
- NVIDIA Blackwell Compatibility Guide
- CUDA Compiler Driver (nvcc) documentation
- CUDA Runtime API documentation
- CUDA Driver API documentation
- PTX ISA documentation

### High-value facts to remember
- B200 = compute capability 10.0
- B200 unified L1/Texture/Shared = 256 KB per SM
- B200 shared-memory carveout options go up to 228 KB per SM
- a single block can address about 227 KB shared memory with proper opt-in
- portable cluster size = 8
- B200 nonportable cluster size = 16 with opt-in
- `sm_100a` / `compute_100a` are required for some architecture-specific Blackwell features

---

## Anti-Patterns

Do not:
- recommend optimization before measuring
- assume Hopper-specific fast paths are valid on B200
- confuse `sm_100` with `sm_100a`
- default to Driver API for ordinary kernel work
- claim TensorCore/PTX behavior without checking docs
- use shared memory aggressively without checking occupancy
- suggest cluster launch without discussing portability and active-block tradeoffs
- trust stale architecture labels or copied arch tables blindly
