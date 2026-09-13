// Vector addition in CUDA C++, for comparison with the Triton kernel in v0_basic.py.
//
// Build and run (from this directory):
//     make && ./vector_add
//
// The kernel is the standard many-block form: every thread computes one output
// element, its position comes from blockIdx, blockDim and threadIdx, and a bounds
// check protects the final block, which usually extends past the end of the array.
// The program checks the result against a CPU loop, then times the kernel with
// CUDA events and reports GB/s, counting three arrays of four-byte floats.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void add_kernel(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global element index
    if (i < n) {                                     // the last block runs past the end
        out[i] = x[i] + y[i];
    }
}

// Runs the kernel once for correctness, then `reps` more times for timing.
// Returns the median kernel time in milliseconds.
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

    // Time the kernel alone. Events are recorded on the GPU's own timeline, so this
    // measures kernel execution rather than the CPU issuing the launch.
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    std::vector<float> ms(reps);
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(start);
        add_kernel<<<grid, block>>>(d_x, d_y, d_out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms[r], start, stop);
    }
    std::sort(ms.begin(), ms.end());

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_out);
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
