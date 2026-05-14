/*
 * RMS Norm — naive but correct baseline.
 *
 * One block per row. Parallel tree reduction in shared memory.
 * No vectorization. No warp shuffles. No fused variants yet — this kernel
 * supports `variant="plain"`; `residual` and `affine` fall back to a trivial
 * extension here and should be optimized by the agent.
 *
 * The agent's job: make this beat torch.compile on the assigned workload set.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifndef RMS_MAX_BLOCK
#define RMS_MAX_BLOCK 1024
#endif

extern "C" {

// --------------------------- bf16 plain -----------------------------------
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

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) reduce_smem[tid] += reduce_smem[tid + s];
        __syncthreads();
    }

    float rsqrt_v = rsqrtf(reduce_smem[0] / (float)D + eps);

    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(X[row * D + d]);
        float w = __bfloat162float(W[d]);
        Y[row * D + d] = __float2bfloat16(v * rsqrt_v * w);
    }
}

// --------------------------- fp16 plain -----------------------------------
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
        __syncthreads();
    }

    float rsqrt_v = rsqrtf(reduce_smem[0] / (float)D + eps);

    for (int d = tid; d < D; d += blockDim.x) {
        float v = __half2float(X[row * D + d]);
        float w = __half2float(W[d]);
        Y[row * D + d] = __float2half(v * rsqrt_v * w);
    }
}

// --------------------------- fp32 plain -----------------------------------
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
        __syncthreads();
    }

    float rsqrt_v = rsqrtf(reduce_smem[0] / (float)D + eps);

    for (int d = tid; d < D; d += blockDim.x) {
        float v = X[row * D + d];
        float w = W[d];
        Y[row * D + d] = v * rsqrt_v * w;
    }
}

// --------------------------- bf16 residual --------------------------------
// Y = (X + R) * rsqrt(mean((X+R)^2) + eps) * W
// Naive implementation — agent may fuse or improve.
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
        __syncthreads();
    }

    float rsqrt_v = rsqrtf(reduce_smem[0] / (float)D + eps);

    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(X[row * D + d]) + __bfloat162float(R[row * D + d]);
        float w = __bfloat162float(W[d]);
        Y[row * D + d] = __float2bfloat16(v * rsqrt_v * w);
    }
}

// --------------------------- bf16 affine ----------------------------------
// Y = X * rsqrt(mean(X^2) + eps) * W + B
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
        __syncthreads();
    }

    float rsqrt_v = rsqrtf(reduce_smem[0] / (float)D + eps);

    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(X[row * D + d]);
        float w = __bfloat162float(W[d]);
        float bias = __bfloat162float(B[d]);
        Y[row * D + d] = __float2bfloat16(v * rsqrt_v * w + bias);
    }
}

// ---- C-style dispatch entry points (called from binding.py) --------------

void rmsnorm_plain_bf16_launch(const void* X, const void* W, void* Y,
                               int M, int D, float eps,
                               int block_threads, cudaStream_t stream) {
    dim3 grid(M);
    dim3 block(block_threads);
    rmsnorm_plain_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)X, (const __nv_bfloat16*)W, (__nv_bfloat16*)Y,
        M, D, eps);
}

void rmsnorm_plain_fp16_launch(const void* X, const void* W, void* Y,
                               int M, int D, float eps,
                               int block_threads, cudaStream_t stream) {
    dim3 grid(M);
    dim3 block(block_threads);
    rmsnorm_plain_fp16_kernel<<<grid, block, 0, stream>>>(
        (const __half*)X, (const __half*)W, (__half*)Y, M, D, eps);
}

void rmsnorm_plain_fp32_launch(const void* X, const void* W, void* Y,
                               int M, int D, float eps,
                               int block_threads, cudaStream_t stream) {
    dim3 grid(M);
    dim3 block(block_threads);
    rmsnorm_plain_fp32_kernel<<<grid, block, 0, stream>>>(
        (const float*)X, (const float*)W, (float*)Y, M, D, eps);
}

void rmsnorm_residual_bf16_launch(const void* X, const void* R, const void* W, void* Y,
                                  int M, int D, float eps,
                                  int block_threads, cudaStream_t stream) {
    dim3 grid(M);
    dim3 block(block_threads);
    rmsnorm_residual_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)X, (const __nv_bfloat16*)R, (const __nv_bfloat16*)W,
        (__nv_bfloat16*)Y, M, D, eps);
}

void rmsnorm_affine_bf16_launch(const void* X, const void* W, const void* B, void* Y,
                                int M, int D, float eps,
                                int block_threads, cudaStream_t stream) {
    dim3 grid(M);
    dim3 block(block_threads);
    rmsnorm_affine_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)X, (const __nv_bfloat16*)W, (const __nv_bfloat16*)B,
        (__nv_bfloat16*)Y, M, D, eps);
}

}  // extern "C"
