/*
 * GEMM with fused bias + GELU(tanh) epilogue — naive tiled baseline.
 *
 *   Y[m, n] = GELU( sum_k X[m, k] * W[n, k] + bias[n] )
 *
 * Shapes:
 *   X    [M, K]   bf16  (row-major)
 *   W    [N, K]   bf16  (row-major — W[n, k] = the weight for output n, input k)
 *   bias [N]      bf16
 *   Y    [M, N]   bf16
 *
 * Baseline tiling: 16x16 output tile per block, each thread computes 1 output
 * element, scalar K-loop with shared-memory staging of a BM×BK = 16×16 A-tile
 * and a BK×BN = 16×16 B-tile. No tensor cores. No vectorization.
 *
 * The agent's job: make this beat torch.compile(F.gelu(x @ w.T + b)). Natural
 * moves on B200: tensor cores (WMMA/WGMMA), vectorized global loads,
 * cp.async/TMA pipelining, per-thread register tiles (4x4 or 8x8),
 * swizzled shared-memory layouts, larger block tiles.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifndef GEMM_BM
#define GEMM_BM 16
#endif
#ifndef GEMM_BN
#define GEMM_BN 16
#endif
#ifndef GEMM_BK
#define GEMM_BK 16
#endif

extern "C" {

// GELU(tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
__device__ __forceinline__ float gelu_tanh(float x) {
    const float C1 = 0.7978845608028654f; // sqrt(2/pi)
    const float C2 = 0.044715f;
    float x3 = x * x * x;
    float t = tanhf(C1 * (x + C2 * x3));
    return 0.5f * x * (1.0f + t);
}

// ---------- bf16 plain: Y = X @ W.T ----------
__global__ void gemm_plain_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,  // [M, K]
    const __nv_bfloat16* __restrict__ W,  // [N, K]
    __nv_bfloat16* __restrict__ Y,        // [M, N]
    int M, int N, int K)
{
    int bm = blockIdx.y;
    int bn = blockIdx.x;
    int tm = threadIdx.y;
    int tn = threadIdx.x;
    int m = bm * GEMM_BM + tm;
    int n = bn * GEMM_BN + tn;

    __shared__ float A_tile[GEMM_BM][GEMM_BK];
    __shared__ float B_tile[GEMM_BK][GEMM_BN];

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
        // Cooperative load of A_tile (BM x BK) and B_tile (BK x BN)
        if (m < M && (k0 + tn) < K)
            A_tile[tm][tn] = __bfloat162float(X[m * K + (k0 + tn)]);
        else
            A_tile[tm][tn] = 0.0f;
        // W is [N, K] so to get B_tile[k, n] we read W[n, k]
        if (n < N && (k0 + tm) < K)
            B_tile[tm][tn] = __bfloat162float(W[n * K + (k0 + tm)]);
        else
            B_tile[tm][tn] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GEMM_BK; k++) {
            acc += A_tile[tm][k] * B_tile[k][tn];
        }
        __syncthreads();
    }

    if (m < M && n < N) {
        Y[m * N + n] = __float2bfloat16(acc);
    }
}

// ---------- bf16 fused: Y = GELU(X @ W.T + bias) ----------
__global__ void gemm_fused_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,  // [M, K]
    const __nv_bfloat16* __restrict__ W,  // [N, K]
    const __nv_bfloat16* __restrict__ B_,  // [N]
    __nv_bfloat16* __restrict__ Y,        // [M, N]
    int M, int N, int K)
{
    int bm = blockIdx.y;
    int bn = blockIdx.x;
    int tm = threadIdx.y;
    int tn = threadIdx.x;
    int m = bm * GEMM_BM + tm;
    int n = bn * GEMM_BN + tn;

    __shared__ float A_tile[GEMM_BM][GEMM_BK];
    __shared__ float B_tile[GEMM_BK][GEMM_BN];

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
        if (m < M && (k0 + tn) < K)
            A_tile[tm][tn] = __bfloat162float(X[m * K + (k0 + tn)]);
        else
            A_tile[tm][tn] = 0.0f;
        if (n < N && (k0 + tm) < K)
            B_tile[tm][tn] = __bfloat162float(W[n * K + (k0 + tm)]);
        else
            B_tile[tm][tn] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GEMM_BK; k++) {
            acc += A_tile[tm][k] * B_tile[k][tn];
        }
        __syncthreads();
    }

    if (m < M && n < N) {
        float bias_v = __bfloat162float(B_[n]);
        float out = gelu_tanh(acc + bias_v);
        Y[m * N + n] = __float2bfloat16(out);
    }
}

// ---- C-style launch entry points ----

void gemm_plain_bf16_launch(const void* X, const void* W, void* Y,
                            int M, int N, int K, cudaStream_t stream) {
    dim3 block(GEMM_BN, GEMM_BM);  // (tn, tm)
    dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (M + GEMM_BM - 1) / GEMM_BM);
    gemm_plain_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)X, (const __nv_bfloat16*)W, (__nv_bfloat16*)Y, M, N, K);
}

void gemm_fused_bf16_launch(const void* X, const void* W, const void* B_, void* Y,
                            int M, int N, int K, cudaStream_t stream) {
    dim3 block(GEMM_BN, GEMM_BM);
    dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (M + GEMM_BM - 1) / GEMM_BM);
    gemm_fused_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)X, (const __nv_bfloat16*)W,
        (const __nv_bfloat16*)B_, (__nv_bfloat16*)Y, M, N, K);
}

}  // extern "C"
