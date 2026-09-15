// Vector addition in CUDA C++, the counterpart of the Triton kernel in v0_basic.py.
//
// Build and run: make && ./vector_add
//
// Checks the result against a CPU loop, then times the kernel with CUDA events
// and reports GB/s over three float arrays. A 256 MB buffer is written before each
// timed launch to evict the inputs from L2, as triton.testing.do_bench does.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

__global__ void add_kernel(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global element index
    if (i < n) {                                     // last block may overrun
        out[i] = x[i] + y[i];
    }
}

// Median kernel time in milliseconds over `reps` launches; *ok reports correctness.
static float run(int n, int block, int reps, bool* ok) {
    const size_t bytes = (size_t)n * sizeof(float);
    std::vector<float> h_x(n), h_y(n), h_out(n);
    for (int i = 0; i < n; ++i) { h_x[i] = (float)i * 0.5f; h_y[i] = 1.0f - (float)i * 0.25f; }

    float *d_x, *d_y, *d_out;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice);

    int grid = (n + block - 1) / block;              // ceiling division
    add_kernel<<<grid, block>>>(d_x, d_y, d_out, n);
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    *ok = true;
    for (int i = 0; i < n; ++i) {
        if (fabsf(h_out[i] - (h_x[i] + h_y[i])) > 1e-5f) { *ok = false; break; }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    const size_t flush_bytes = 256u << 20;
    void* flush = nullptr;
    cudaMalloc(&flush, flush_bytes);
    std::vector<float> ms(reps);
    for (int r = 0; r < reps; ++r) {
        cudaMemsetAsync(flush, 0, flush_bytes);      // evict x, y, out from L2
        cudaEventRecord(start);
        add_kernel<<<grid, block>>>(d_x, d_y, d_out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms[r], start, stop);
    }
    std::sort(ms.begin(), ms.end());

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(flush); cudaFree(d_x); cudaFree(d_y); cudaFree(d_out);
    return ms[reps / 2];
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Running on %s\n\n", p.name);

    const int block = 256;
    const int sizes[] = {1000, 1 << 16, 1 << 20, 1 << 24};
    printf("%12s %8s %10s %10s %8s\n", "elements", "blocks", "median ms", "GB/s", "check");
    for (int n : sizes) {
        bool ok;
        float ms = run(n, block, 50, &ok);
        double gbps = 3.0 * n * sizeof(float) / (ms * 1e-3) / 1e9;
        printf("%12d %8d %10.4f %10.1f %8s\n", n, (n + block - 1) / block, ms, gbps, ok ? "OK" : "FAIL");
    }
    return 0;
}
