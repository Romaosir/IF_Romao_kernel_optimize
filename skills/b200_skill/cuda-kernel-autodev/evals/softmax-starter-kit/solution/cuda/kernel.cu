/*
 * Softmax — naive three-pass baseline.
 *
 *   Y[i, :] = exp(X[i, :] - max) / sum(exp(X[i, :] - max))
 *
 * Pass 1: tree reduction to find max of the row
 * Pass 2: tree reduction to find sum of exp(x - max)
 * Pass 3: write y = exp(x - max) / sum
 *
 * One block per row. Scalar loads. Re-reads X three times (implicitly — once per
 * pass) and recomputes exp twice. The agent's job: optimize this — online
 * softmax (fused max+sum in one pass), vectorized loads, warp-shuffle
 * reductions, register caching, base-2 exp.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifndef SM_MAX_BLOCK
#define SM_MAX_BLOCK 1024
#endif

extern "C" {

__global__ void softmax_plain_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,
    __nv_bfloat16* __restrict__ Y,
    int M, int D)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float reduce_smem[SM_MAX_BLOCK];

    // Pass 1: max
    float local_max = -INFINITY;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(X[row * D + d]);
        if (v > local_max) local_max = v;
    }
    reduce_smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) reduce_smem[tid] = fmaxf(reduce_smem[tid], reduce_smem[tid + s]);
        __syncthreads();
    }
    float row_max = reduce_smem[0];

    // Pass 2: sum of exp
    float local_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(X[row * D + d]);
        local_sum += __expf(v - row_max);
    }
    reduce_smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) reduce_smem[tid] += reduce_smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce_smem[0];

    // Pass 3: write
    for (int d = tid; d < D; d += blockDim.x) {
        float v = __bfloat162float(X[row * D + d]);
        Y[row * D + d] = __float2bfloat16(__expf(v - row_max) * inv_sum);
    }
}

__global__ void softmax_plain_fp16_kernel(
    const __half* __restrict__ X,
    __half* __restrict__ Y,
    int M, int D)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float reduce_smem[SM_MAX_BLOCK];

    float local_max = -INFINITY;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = __half2float(X[row * D + d]);
        if (v > local_max) local_max = v;
    }
    reduce_smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) reduce_smem[tid] = fmaxf(reduce_smem[tid], reduce_smem[tid + s]);
        __syncthreads();
    }
    float row_max = reduce_smem[0];

    float local_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = __half2float(X[row * D + d]);
        local_sum += __expf(v - row_max);
    }
    reduce_smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) reduce_smem[tid] += reduce_smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce_smem[0];

    for (int d = tid; d < D; d += blockDim.x) {
        float v = __half2float(X[row * D + d]);
        Y[row * D + d] = __float2half(__expf(v - row_max) * inv_sum);
    }
}

__global__ void softmax_plain_fp32_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int D)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float reduce_smem[SM_MAX_BLOCK];

    float local_max = -INFINITY;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = X[row * D + d];
        if (v > local_max) local_max = v;
    }
    reduce_smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) reduce_smem[tid] = fmaxf(reduce_smem[tid], reduce_smem[tid + s]);
        __syncthreads();
    }
    float row_max = reduce_smem[0];

    float local_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = X[row * D + d];
        local_sum += __expf(v - row_max);
    }
    reduce_smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) reduce_smem[tid] += reduce_smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce_smem[0];

    for (int d = tid; d < D; d += blockDim.x) {
        float v = X[row * D + d];
        Y[row * D + d] = __expf(v - row_max) * inv_sum;
    }
}

// ---- C-style dispatch entry points --------------------------------------

void softmax_plain_bf16_launch(const void* X, void* Y, int M, int D,
                               int block_threads, cudaStream_t stream) {
    softmax_plain_bf16_kernel<<<dim3(M), dim3(block_threads), 0, stream>>>(
        (const __nv_bfloat16*)X, (__nv_bfloat16*)Y, M, D);
}
void softmax_plain_fp16_launch(const void* X, void* Y, int M, int D,
                               int block_threads, cudaStream_t stream) {
    softmax_plain_fp16_kernel<<<dim3(M), dim3(block_threads), 0, stream>>>(
        (const __half*)X, (__half*)Y, M, D);
}
void softmax_plain_fp32_launch(const void* X, void* Y, int M, int D,
                               int block_threads, cudaStream_t stream) {
    softmax_plain_fp32_kernel<<<dim3(M), dim3(block_threads), 0, stream>>>(
        (const float*)X, (float*)Y, M, D);
}

}  // extern "C"
