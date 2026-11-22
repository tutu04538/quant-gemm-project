// V2a：一个 warp 完成固定 16×16×16 BF16 WMMA。
#define V2_SINGLE_TILE
#include "v2_wmma_common.cuh"
namespace wmma = nvcuda::wmma;

__global__ void v2_wmma(const __nv_bfloat16* a, const __nv_bfloat16* b,
                        float* c, int m, int n, int k) {
    // 本练习固定 M=N=K=16，调用方保证 block=(32,1,1)，grid=(1,1,1)。
    // 这三个 fragment 是 warp 协作的抽象，不要猜测每个 lane 的内部布局。
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    // TODO 1：将 acc 初始化为 0。
    wmma::fill_fragment(acc, 0.0f);
    // TODO 2：按行优先加载 A 与 B；本例行步长都是 16 个 BF16 元素。
    wmma::load_matrix_sync(af, a, 16);
    wmma::load_matrix_sync(bf, b, 16);
    // TODO 3：让完整 warp 协作执行一次 MMA，将结果累加到 acc。
    wmma::mma_sync(acc, af, bf, acc);
    // TODO 4：按行优先把 acc 写入 C；输出行步长是 16 个 float 元素。
    wmma::store_matrix_sync(c, acc, 16, wmma::mem_row_major);
}
