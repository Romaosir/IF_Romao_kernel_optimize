/*
 * RMS Norm — already-decent baseline.
 *
 * Features:
 *   - vec4 (uint4) loads for bf16/fp16 — 8 elements per thread per load
 *   - warp-shuffle reduction (no shared-memory tree)
 *   - row loaded twice (once for sum, once for normalize); agent may cache
 *     in registers to avoid this if register budget permits
 *   - block_threads = 256 for D >= 256
 *
 * Speedup vs naive on the same shapes typically ~2–3x. This should already
 * be in the neighborhood of torch.compile for many workloads. The agent's
 * job in scenario 10: push *beyond* this near-ceiling baseline — plateau
 * territory. Likely next moves: register caching of the row, multi-row per
 * block, fused residual/affine without recomputation.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" {

// ---- warp shuffle reductions ----
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    int num_warps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < num_warps) ? smem[lane] : 0.0f;
    if (warp == 0) v = warp_reduce_sum(v);
    return v;
}

// -------- bf16 plain: vec4 loads (8 bf16 per load), warp-shuffle reduction
__global__ void rmsnorm_plain_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ Y,
    int M, int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float smem[32];

    // Vectorized load: each uint4 is 8 bf16 values
    const int D8 = D >> 3;  // assume D % 8 == 0
    const uint4* X8 = reinterpret_cast<const uint4*>(X + row * D);

    float local_sum = 0.0f;
    for (int i = tid; i < D8; i += blockDim.x) {
        uint4 packed = X8[i];
        __nv_bfloat16* parts = reinterpret_cast<__nv_bfloat16*>(&packed);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float v = __bfloat162float(parts[k]);
            local_sum += v * v;
        }
    }
    float total = block_reduce_sum(local_sum, smem);
    __shared__ float rsqrt_v;
    if (tid == 0) rsqrt_v = rsqrtf(total / (float)D + eps);
    __syncthreads();

    uint4* Y8 = reinterpret_cast<uint4*>(Y + row * D);
    const uint4* W8 = reinterpret_cast<const uint4*>(W);
    for (int i = tid; i < D8; i += blockDim.x) {
        uint4 packed_x = X8[i];
        uint4 packed_w = W8[i];
        uint4 out_packed;
        __nv_bfloat16* px = reinterpret_cast<__nv_bfloat16*>(&packed_x);
        __nv_bfloat16* pw = reinterpret_cast<__nv_bfloat16*>(&packed_w);
        __nv_bfloat16* po = reinterpret_cast<__nv_bfloat16*>(&out_packed);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            po[k] = __float2bfloat16(__bfloat162float(px[k]) * rsqrt_v * __bfloat162float(pw[k]));
        }
        Y8[i] = out_packed;
    }
}

// -------- fp16 plain (same pattern)
__global__ void rmsnorm_plain_fp16_kernel(
    const __half* __restrict__ X,
    const __half* __restrict__ W,
    __half* __restrict__ Y,
    int M, int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float smem[32];
    const int D8 = D >> 3;
    const uint4* X8 = reinterpret_cast<const uint4*>(X + row * D);

    float local_sum = 0.0f;
    for (int i = tid; i < D8; i += blockDim.x) {
        uint4 packed = X8[i];
        __half* parts = reinterpret_cast<__half*>(&packed);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float v = __half2float(parts[k]);
            local_sum += v * v;
        }
    }
    float total = block_reduce_sum(local_sum, smem);
    __shared__ float rsqrt_v;
    if (tid == 0) rsqrt_v = rsqrtf(total / (float)D + eps);
    __syncthreads();

    uint4* Y8 = reinterpret_cast<uint4*>(Y + row * D);
    const uint4* W8 = reinterpret_cast<const uint4*>(W);
    for (int i = tid; i < D8; i += blockDim.x) {
        uint4 packed_x = X8[i];
        uint4 packed_w = W8[i];
        uint4 out_packed;
        __half* px = reinterpret_cast<__half*>(&packed_x);
        __half* pw = reinterpret_cast<__half*>(&packed_w);
        __half* po = reinterpret_cast<__half*>(&out_packed);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            po[k] = __float2half(__half2float(px[k]) * rsqrt_v * __half2float(pw[k]));
        }
        Y8[i] = out_packed;
    }
}

// -------- fp32 plain (float4 loads = 4 floats)
__global__ void rmsnorm_plain_fp32_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W,
    float* __restrict__ Y,
    int M, int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float smem[32];
    const int D4 = D >> 2;
    const float4* X4 = reinterpret_cast<const float4*>(X + row * D);

    float local_sum = 0.0f;
    for (int i = tid; i < D4; i += blockDim.x) {
        float4 v = X4[i];
        local_sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    float total = block_reduce_sum(local_sum, smem);
    __shared__ float rsqrt_v;
    if (tid == 0) rsqrt_v = rsqrtf(total / (float)D + eps);
    __syncthreads();

    float4* Y4 = reinterpret_cast<float4*>(Y + row * D);
    const float4* W4 = reinterpret_cast<const float4*>(W);
    for (int i = tid; i < D4; i += blockDim.x) {
        float4 vx = X4[i];
        float4 vw = W4[i];
        float4 vo;
        vo.x = vx.x * rsqrt_v * vw.x;
        vo.y = vx.y * rsqrt_v * vw.y;
        vo.z = vx.z * rsqrt_v * vw.z;
        vo.w = vx.w * rsqrt_v * vw.w;
        Y4[i] = vo;
    }
}

// -------- bf16 residual (same pattern; recomputes X+R twice)
__global__ void rmsnorm_residual_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ R,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ Y,
    int M, int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float smem[32];
    const int D8 = D >> 3;
    const uint4* X8 = reinterpret_cast<const uint4*>(X + row * D);
    const uint4* R8 = reinterpret_cast<const uint4*>(R + row * D);

    float local_sum = 0.0f;
    for (int i = tid; i < D8; i += blockDim.x) {
        uint4 px = X8[i]; uint4 pr = R8[i];
        __nv_bfloat16* ax = reinterpret_cast<__nv_bfloat16*>(&px);
        __nv_bfloat16* ar = reinterpret_cast<__nv_bfloat16*>(&pr);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float v = __bfloat162float(ax[k]) + __bfloat162float(ar[k]);
            local_sum += v * v;
        }
    }
    float total = block_reduce_sum(local_sum, smem);
    __shared__ float rsqrt_v;
    if (tid == 0) rsqrt_v = rsqrtf(total / (float)D + eps);
    __syncthreads();

    uint4* Y8 = reinterpret_cast<uint4*>(Y + row * D);
    const uint4* W8 = reinterpret_cast<const uint4*>(W);
    for (int i = tid; i < D8; i += blockDim.x) {
        uint4 px = X8[i]; uint4 pr = R8[i]; uint4 pw = W8[i]; uint4 po;
        __nv_bfloat16* ax = reinterpret_cast<__nv_bfloat16*>(&px);
        __nv_bfloat16* ar = reinterpret_cast<__nv_bfloat16*>(&pr);
        __nv_bfloat16* aw = reinterpret_cast<__nv_bfloat16*>(&pw);
        __nv_bfloat16* ao = reinterpret_cast<__nv_bfloat16*>(&po);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float v = __bfloat162float(ax[k]) + __bfloat162float(ar[k]);
            ao[k] = __float2bfloat16(v * rsqrt_v * __bfloat162float(aw[k]));
        }
        Y8[i] = po;
    }
}

// -------- bf16 affine (same pattern)
__global__ void rmsnorm_affine_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ Y,
    int M, int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float smem[32];
    const int D8 = D >> 3;
    const uint4* X8 = reinterpret_cast<const uint4*>(X + row * D);

    float local_sum = 0.0f;
    for (int i = tid; i < D8; i += blockDim.x) {
        uint4 packed = X8[i];
        __nv_bfloat16* parts = reinterpret_cast<__nv_bfloat16*>(&packed);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float v = __bfloat162float(parts[k]);
            local_sum += v * v;
        }
    }
    float total = block_reduce_sum(local_sum, smem);
    __shared__ float rsqrt_v;
    if (tid == 0) rsqrt_v = rsqrtf(total / (float)D + eps);
    __syncthreads();

    uint4* Y8 = reinterpret_cast<uint4*>(Y + row * D);
    const uint4* W8 = reinterpret_cast<const uint4*>(W);
    const uint4* B8 = reinterpret_cast<const uint4*>(B);
    for (int i = tid; i < D8; i += blockDim.x) {
        uint4 px = X8[i]; uint4 pw = W8[i]; uint4 pb = B8[i]; uint4 po;
        __nv_bfloat16* ax = reinterpret_cast<__nv_bfloat16*>(&px);
        __nv_bfloat16* aw = reinterpret_cast<__nv_bfloat16*>(&pw);
        __nv_bfloat16* ab = reinterpret_cast<__nv_bfloat16*>(&pb);
        __nv_bfloat16* ao = reinterpret_cast<__nv_bfloat16*>(&po);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float v = __bfloat162float(ax[k]) * rsqrt_v * __bfloat162float(aw[k]) + __bfloat162float(ab[k]);
            ao[k] = __float2bfloat16(v);
        }
        Y8[i] = po;
    }
}

// ---- C-style dispatch entry points ---------------------------------------

void rmsnorm_plain_bf16_launch(const void* X, const void* W, void* Y,
                               int M, int D, float eps,
                               int /*block_threads*/, cudaStream_t stream) {
    rmsnorm_plain_bf16_kernel<<<dim3(M), dim3(256), 0, stream>>>(
        (const __nv_bfloat16*)X, (const __nv_bfloat16*)W, (__nv_bfloat16*)Y, M, D, eps);
}
void rmsnorm_plain_fp16_launch(const void* X, const void* W, void* Y,
                               int M, int D, float eps,
                               int /*block_threads*/, cudaStream_t stream) {
    rmsnorm_plain_fp16_kernel<<<dim3(M), dim3(256), 0, stream>>>(
        (const __half*)X, (const __half*)W, (__half*)Y, M, D, eps);
}
void rmsnorm_plain_fp32_launch(const void* X, const void* W, void* Y,
                               int M, int D, float eps,
                               int /*block_threads*/, cudaStream_t stream) {
    rmsnorm_plain_fp32_kernel<<<dim3(M), dim3(256), 0, stream>>>(
        (const float*)X, (const float*)W, (float*)Y, M, D, eps);
}
void rmsnorm_residual_bf16_launch(const void* X, const void* R, const void* W, void* Y,
                                  int M, int D, float eps,
                                  int /*block_threads*/, cudaStream_t stream) {
    rmsnorm_residual_bf16_kernel<<<dim3(M), dim3(256), 0, stream>>>(
        (const __nv_bfloat16*)X, (const __nv_bfloat16*)R, (const __nv_bfloat16*)W,
        (__nv_bfloat16*)Y, M, D, eps);
}
void rmsnorm_affine_bf16_launch(const void* X, const void* W, const void* B, void* Y,
                                int M, int D, float eps,
                                int /*block_threads*/, cudaStream_t stream) {
    rmsnorm_affine_bf16_kernel<<<dim3(M), dim3(256), 0, stream>>>(
        (const __nv_bfloat16*)X, (const __nv_bfloat16*)W, (const __nv_bfloat16*)B,
        (__nv_bfloat16*)Y, M, D, eps);
}

}  // extern "C"
