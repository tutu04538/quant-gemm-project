// V2b：一个 block/warp 负责一个 16×16 输出 tile，多次遍历 K。
#include "v2_wmma_common.cuh"
namespace wmma = nvcuda::wmma;

__global__ void v2_wmma(const __nv_bfloat16* a, const __nv_bfloat16* b,
                        float* c, int m, int n, int k) {
    // 只支持 M/N/K 均为正的 16 倍数；每 block 恰好 32 线程。
    // 全 warp 共用同一个 blockIdx，WMMA 调用必须由完整 warp 一致执行。
    const int row = blockIdx.y * 16;
    const int col = blockIdx.x * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    // TODO 1：清零 acc。
    wmma::fill_fragment(acc, 0.0f);
    for (int base = 0; base < k; base += 16) {
        // TODO 2：定位并加载 A[row:row+16,base:base+16]。
        wmma::load_matrix_sync(af, &a[row*k+base], k);
        // TODO 3：定位并加载 B[base:base+16,col:col+16]。
        wmma::load_matrix_sync(bf, &b[base*n+col], n);
        // TODO 4：执行 MMA，累加到 acc；acc 不能在 K 循环内重置。
        wmma::mma_sync(acc, af, bf, acc);
    }
    // TODO 5：把 acc 写回 C[row:row+16,col:col+16]。
    wmma::store_matrix_sync(&c[row*n+col], acc, n, wmma::mem_row_major);
}
