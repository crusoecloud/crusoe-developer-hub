// Square eight numbers on the GPU, one thread per element.
//
// Build and run: make square && ./02_square_array

#include <cstdio>

__global__ void square_kernel(const float* in, float* out, int n) {
    int i = threadIdx.x;      // one block: thread index = element index
    if (i < n) {              // never touch memory past the array's end
        out[i] = in[i] * in[i];
    }
}

int main() {
    const int n = 8;
    const size_t bytes = n * sizeof(float);
    float h_in[n], h_out[n];                            // host arrays
    for (int i = 0; i < n; ++i) h_in[i] = (float)(i + 1);

    float *d_in = nullptr, *d_out = nullptr;            // device pointers
    cudaMalloc(&d_in, bytes);                           // 1. allocate on device
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);   // 2. copy in
    square_kernel<<<1, n>>>(d_in, d_out, n);                 // 3. launch
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost); // 4. copy out
    cudaFree(d_in);                                          // 5. free
    cudaFree(d_out);

    for (int i = 0; i < n; ++i)
        printf("%4.0f squared is %4.0f\n", h_in[i], h_out[i]);
    return 0;
}
