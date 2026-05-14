# Hardware discovery

Every optimization decision downstream depends on knowing the target GPU precisely. Don't assume, don't guess, and don't use defaults from "modern NVIDIA GPU" — the gap between, say, an L4 and a B200 is two orders of magnitude on some metrics.

## The minimum spec sheet

Capture these facts at the start of Phase A and keep them visible:

| Field | Why it matters |
|---|---|
| GPU name + SM arch + compute capability | Determines which PTX intrinsics and tensor-core instructions are available |
| Number of SMs | Sets the grid size sweet spot; affects persistent-kernel viability |
| Shared memory per SM (default + opt-in max) | Caps tile size; over 48 KB requires `cudaFuncSetAttribute` |
| Registers per SM and per thread max | Sets the occupancy vs ILP tradeoff |
| HBM bandwidth | The bandwidth-bound ceiling for the roofline |
| HBM capacity | Caps problem size |
| L2 cache size | Determines how much reuse is "free" — large L2 changes tile scheduling strategy |
| Tensor core generation | WMMA / WGMMA / MMA sync; fp8/bf16/fp16/tf32 matrix |
| Async copy support | `cp.async` (SM80+), TMA (SM90+), `cp.async.bulk` |
| L1 / unified cache behavior | Affects coalescing penalties |

## Discovery order

### 1. Already-provided inputs

Check these first — the user has usually given you the answer already:
- A `program.md`, `README.md`, or `<arch>-optimization-guide.md` file in the repo
- A `config.toml` with an arch or SM field
- An explicit statement in the current conversation or in CLAUDE.md
- A hardware guide in `references/` or `docs/`

When you find it, echo the specific facts back to the user: "I see this is targeting B200 (SM100, 148 SMs, 228 KB opt-in shared mem/SM, 8 TB/s HBM3e, 126 MB L2). Correct?" Getting confirmation up front is cheap; discovering you had the wrong SM 20 experiments in is expensive.

### 2. Live detection

If no spec is provided, try:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free --format=csv
nvcc --version
# If you can compile and run: device query
cat > /tmp/devquery.cu <<'EOF'
#include <cuda_runtime.h>
#include <cstdio>
int main() {
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("name=%s cc=%d.%d sms=%d smem/sm=%zu regs/sm=%d warpsz=%d\n",
           p.name, p.major, p.minor, p.multiProcessorCount,
           p.sharedMemPerMultiprocessor, p.regsPerMultiprocessor, p.warpSize);
    return 0;
}
EOF
nvcc -o /tmp/devquery /tmp/devquery.cu && /tmp/devquery
```

This gives you SM count, shared mem, register count, compute capability — the bare essentials. Everything else comes from the spec sheet or the web.

### 3. Ask the user

If live detection fails (no CUDA available in the env, remote dev where the GPU is elsewhere), ask directly:

> Which GPU should I target? I need: the product name (e.g. "H100 SXM"), the compute capability (e.g. "9.0"), and the CUDA toolkit version. If you have an architecture-specific optimization guide, that helps too.

Don't proceed without an answer. A CUDA kernel targeting the wrong SM may compile and even run, but the optimizations you pick will be mis-tuned.

### 4. Fill gaps with web search

Once you know the GPU family, look up the missing fields authoritatively. Good sources, in order of preference:
- NVIDIA's architecture whitepaper for the GPU (e.g. "B200 architecture whitepaper")
- NVIDIA's CUDA C Programming Guide Appendix for compute capability X.Y
- The `nvidia-smi` / `cuobjdump` / `cuda-samples/deviceQuery` output from a public source

Cite the source in your hardware-spec block so the user can check it.

## Arch-specific reminders

These change across GPU generations and are the usual gotchas:

| Feature | SM80 (A100) | SM89 (Ada) | SM90 (H100/H200) | SM100 (B100/B200) |
|---|---|---|---|---|
| `cp.async` | yes | yes | yes (+ bulk) | yes |
| TMA | no | no | **yes** | yes |
| WGMMA | no | no | **yes** | yes |
| Cluster launch | no | no | **yes** | yes |
| Thread block cluster smem | no | no | **yes** | yes |
| WMMA (legacy) | yes | yes | yes (prefer WGMMA) | yes (prefer WGMMA) |
| FP8 | no | yes | yes | yes |
| MXFP | no | no | no | **yes** |

If you're writing code for SM90+ and not using TMA / WGMMA where it fits, you're leaving performance on the table. Conversely, if you're targeting SM80 and trying to use WGMMA, the code won't even compile.

## Recording the result

Put the confirmed spec at the top of whatever planning doc you use (or in a dedicated `notes/hardware.md`). It should look like:

```markdown
# Target hardware

- **GPU**: NVIDIA B200
- **Arch**: SM100 (compute capability 10.0)
- **SMs**: 148
- **Shared mem/SM**: 228 KB (opt-in max, 256 KB unified)
- **Registers/SM**: 65536 / 255 per thread
- **HBM**: 8 TB/s HBM3e, 192 GB
- **L2**: 126 MB
- **Tensor cores**: WGMMA (fp8/bf16/fp16/tf32), MXFP
- **Async copy**: cp.async, TMA, cp.async.bulk
- **CUDA toolkit**: 12.4
- **Source**: NVIDIA B200 architecture whitepaper + `deviceQuery` on host

## Implications for this run
- Enable opt-in shared mem via `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, 228*1024)`
- Prefer WGMMA over WMMA
- Use TMA for bulk loads from global to shared when applicable
```

The "implications" section is the load-bearing part — it turns the spec sheet into decisions the loop can act on.
