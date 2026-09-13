# Builds the CUDA C++ reference for lesson 01. Requires the CUDA toolkit (nvcc).

NVCC ?= nvcc
NVCCFLAGS ?= -O2 -arch=native

vector_add: vector_add.cu
	$(NVCC) $(NVCCFLAGS) -o vector_add vector_add.cu

clean:
	rm -f vector_add

.PHONY: clean
