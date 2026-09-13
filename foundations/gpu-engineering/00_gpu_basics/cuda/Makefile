# Builds the CUDA C++ programs for lesson 00. Requires the CUDA toolkit (nvcc).
#
#     make            # build both
#     make hello      # 01_hello_kernel
#     make square     # 02_square_array
#     make clean

NVCC ?= nvcc
NVCCFLAGS ?= -O2 -arch=native

all: hello square

hello: 01_hello_kernel.cu
	$(NVCC) $(NVCCFLAGS) -o 01_hello_kernel 01_hello_kernel.cu

square: 02_square_array.cu
	$(NVCC) $(NVCCFLAGS) -o 02_square_array 02_square_array.cu

clean:
	rm -f 01_hello_kernel 02_square_array

.PHONY: all hello square clean
