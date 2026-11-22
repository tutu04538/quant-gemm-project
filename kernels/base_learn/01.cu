__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    c[x] = a[x] + b[x];
}