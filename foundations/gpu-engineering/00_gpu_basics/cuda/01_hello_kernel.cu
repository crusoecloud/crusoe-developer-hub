// One block of eight threads; each thread prints its own index.
//
// Build and run: make hello && ./01_hello_kernel

#include <cstdio>

__global__ void hello_kernel() {
    printf("hello from thread %d\n", threadIdx.x);  // position inside the block
}

int main() {
    hello_kernel<<<1, 8>>>();   // <<<blocks, threads per block>>>
    cudaDeviceSynchronize();    // the launch returns immediately; wait for the GPU
    return 0;
}
