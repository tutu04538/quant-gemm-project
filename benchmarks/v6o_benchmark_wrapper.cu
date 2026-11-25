// Benchmark bridge only; v6o kernel remains the user's fixed-shape implementation.
#define main v6o_training_main
#include "../kernels/marlin_from_scratch/v6o_four_block_template.cu"
#undef main
extern "C" int launch_v6o(const void* a,const void* q,const void* z,const void* s,const void* bias,void* out,void* diag,void* tmp,void* locks,void* stream) {
    v6o_compute<<<4,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(a),static_cast<const uint32_t*>(q),static_cast<const uint32_t*>(z),
        static_cast<const __nv_bfloat16*>(s),static_cast<const __nv_bfloat16*>(bias),static_cast<__nv_bfloat16*>(out),
        static_cast<float*>(diag),static_cast<float*>(tmp),static_cast<int*>(locks));
    return int(cudaGetLastError());
}
