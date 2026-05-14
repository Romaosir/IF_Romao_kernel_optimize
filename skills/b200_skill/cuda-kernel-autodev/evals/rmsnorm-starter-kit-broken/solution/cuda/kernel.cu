/*
 * RMS Norm — starter kit with an intentional latent bug.
 *
 * The kernel looks plausible, compiles, and passes on small shapes — but it
 * will fail correctness on the first workload that exercises its bug.
 *
 * Scenario 9 instruction to the agent: "This kernel builds but fails
 * correctness. Diagnose the bug, fix it, and only then begin optimizing."
 *
 * --------- The bug (no spoilers for the agent) ----------
 * The tree reduction loop re-reads reduce_smem[tid] after writing it,
 * without a __syncthreads() INSIDE the loop body after the write step.
 * (Specifically: the + s dereference uses the just-written value of
 * another thread before sync.)
 * --------------------------------------------------------
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifndef RMS_MAX_BLOCK
#define RMS_MAX_BLOCK 1024
#endif

extern "C" {

__global__ void rmsnorm_plain_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ Y,
    int M, int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float reduce_smem[RMS_MAX_BLOCK];

    float local_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(X[row * D + d]);
        local_sum += v * v;
    }
    reduce_smem[tid] = local_sum;
    __syncthreads();

    // BUG: missing __syncthreads() inside the reduction loop.
    // After the first write, threads read stale values from other lanes.
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) reduce_smem[tid] += reduce_smem[tid + s];
        // (no __syncthreads here — wrong)
    }
    __syncthreads();

    float rsqrt_v = rsqrtf(reduce_smem[0] / (float)D + eps);

    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(X[row * D + d]);
        float w = __bfloat162float(W[d]);
        Y[row * D + d] = __float2bfloat16(v * rsqrt_v * w);
    }
}

// fp16, fp32, residual, affine variants — same naive layout, same bug pattern
// in their reduction loop. (For brevity, only bf16 plain is elaborated above;
// the agent should fix the bug in the shared reduction pattern.)

__global__ void rmsnorm_plain_fp16_kernel(
    const __half* __restrict__ X,
    const __half* __restrict__ W,
    __half* __restrict__ Y,
    int M, int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float reduce_smem[RMS_MAX_BLOCK];

    float local_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = __half2float(X[row * D + d]);
        local_sum += v * v;
    }
    reduce_smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) reduce_smem[tid] += reduce_smem[tid + s];
        // (no __syncthreads here — wrong)
    }
    __syncthreads();

    float rsqrt_v = rsqrtf(reduce_smem[0] / (float)D + eps);

    for (int d = tid; d < D; d += blockDim.x) {
        float v = __half2float(X[row * D + d]);
        float w = __half2float(W[d]);
        Y[row * D + d] = __float2half(v * rsqrt_v * w);
    }
}

__global__ void rmsnorm_plain_fp32_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W,
    float* __restrict__ Y,
    int M, int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float reduce_smem[RMS_MAX_BLOCK];

    float local_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = X[row * D + d];
        local_sum += v * v;
    }
    reduce_smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) reduce_smem[tid] += reduce_smem[tid + s];
    }
    __syncthreads();

    float rsqrt_v = rsqrtf(reduce_smem[0] / (float)D + eps);

    for (int d = tid; d < D; d += blockDim.x) {
        float v = X[row * D + d];
        float w = W[d];
        Y[row * D + d] = v * rsqrt_v * w;
    }
}

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

    __shared__ float reduce_smem[RMS_MAX_BLOCK];

    float local_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(X[row * D + d]) + __bfloat162float(R[row * D + d]);
        local_sum += v * v;
    }
    reduce_smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) reduce_smem[tid] += reduce_smem[tid + s];
    }
    __syncthreads();

    float rsqrt_v = rsqrtf(reduce_smem[0] / (float)D + eps);

    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(X[row * D + d]) + __bfloat162float(R[row * D + d]);
        float w = __bfloat162float(W[d]);
        Y[row * D + d] = __float2bfloat16(v * rsqrt_v * w);
    }
}

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

    __shared__ float reduce_smem[RMS_MAX_BLOCK];

    float local_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(X[row * D + d]);
        local_sum += v * v;
    }
    reduce_smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) reduce_smem[tid] += reduce_smem[tid + s];
    }
    __syncthreads();

    float rsqrt_v = rsqrtf(reduce_smem[0] / (float)D + eps);

    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(X[row * D + d]);
        float w = __bfloat162float(W[d]);
        float bias = __bfloat162float(B[d]);
        Y[row * D + d] = __float2bfloat16(v * rsqrt_v * w + bias);
    }
}

// ---- C-style dispatch entry points ---------------------------------------

void rmsnorm_plain_bf16_launch(const void* X, const void* W, void* Y,
                               int M, int D, float eps,
                               int block_threads, cudaStream_t stream) {
    rmsnorm_plain_bf16_kernel<<<dim3(M), dim3(block_threads), 0, stream>>>(
        (const __nv_bfloat16*)X, (const __nv_bfloat16*)W, (__nv_bfloat16*)Y, M, D, eps);
}
void rmsnorm_plain_fp16_launch(const void* X, const void* W, void* Y,
                               int M, int D, float eps,
                               int block_threads, cudaStream_t stream) {
    rmsnorm_plain_fp16_kernel<<<dim3(M), dim3(block_threads), 0, stream>>>(
        (const __half*)X, (const __half*)W, (__half*)Y, M, D, eps);
}
void rmsnorm_plain_fp32_launch(const void* X, const void* W, void* Y,
                               int M, int D, float eps,
                               int block_threads, cudaStream_t stream) {
    rmsnorm_plain_fp32_kernel<<<dim3(M), dim3(block_threads), 0, stream>>>(
        (const float*)X, (const float*)W, (float*)Y, M, D, eps);
}
void rmsnorm_residual_bf16_launch(const void* X, const void* R, const void* W, void* Y,
                                  int M, int D, float eps,
                                  int block_threads, cudaStream_t stream) {
    rmsnorm_residual_bf16_kernel<<<dim3(M), dim3(block_threads), 0, stream>>>(
        (const __nv_bfloat16*)X, (const __nv_bfloat16*)R, (const __nv_bfloat16*)W,
        (__nv_bfloat16*)Y, M, D, eps);
}
void rmsnorm_affine_bf16_launch(const void* X, const void* W, const void* B, void* Y,
                                int M, int D, float eps,
                                int block_threads, cudaStream_t stream) {
    rmsnorm_affine_bf16_kernel<<<dim3(M), dim3(block_threads), 0, stream>>>(
        (const __nv_bfloat16*)X, (const __nv_bfloat16*)W, (const __nv_bfloat16*)B,
        (__nv_bfloat16*)Y, M, D, eps);
}

}  // extern "C"
