// A kernel that does nothing but announce which thread is running it.
//
// Build and run (from this directory):
//     make hello && ./01_hello_kernel
//
// One block of eight threads is launched. Each thread prints its own index.
// The order of the lines is decided by the hardware, not by the program.

#include <cstdio>

__global__ void hello_kernel() {
    // threadIdx.x is this thread's position inside its block, 0 to 7 here.
    printf("hello from thread %d\n", threadIdx.x);
}

int main() {
    // <<<blocks, threads per block>>>: one block, eight threads.
    hello_kernel<<<1, 8>>>();

    // The launch returns immediately. Wait for the GPU to finish before main() exits,
    // otherwise the program may end before any thread has printed.
    cudaDeviceSynchronize();
    return 0;
}
