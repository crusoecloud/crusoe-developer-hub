// Square every element of a small array on the GPU, one thread per element.
//
// Build and run (from this directory):
//     make square && ./02_square_array
//
// This is the smallest complete CUDA program: allocate memory on the device,
// copy the input across, launch one block of threads, copy the result back.

#include <cstdio>

__global__ void square_kernel(const float* in, float* out, int n) {
    int i = threadIdx.x;      // one block, so the thread index is the element index
    if (i < n) {              // never touch memory past the end of the array
        out[i] = in[i] * in[i];
    }
}

int main() {
    const int n = 8;
    const size_t bytes = n * sizeof(float);

    // Host (CPU) arrays.
    float h_in[n], h_out[n];
    for (int i = 0; i < n; ++i) h_in[i] = (float)(i + 1);

    // Step 1: allocate device (GPU) memory.
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // Step 2: copy the input from host memory to device memory.
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Step 3: launch one block of n threads. The kernel runs on the device.
    square_kernel<<<1, n>>>(d_in, d_out, n);

    // Step 4: copy the result back. cudaMemcpy waits for the kernel to finish first.
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Step 5: free device memory.
    cudaFree(d_in);
    cudaFree(d_out);

    for (int i = 0; i < n; ++i) printf("%4.0f squared is %4.0f\n", h_in[i], h_out[i]);
    return 0;
}
