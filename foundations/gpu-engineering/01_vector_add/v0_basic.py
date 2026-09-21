"""Vector addition, output[i] = x[i] + y[i]: the first Triton kernel, with a correctness check.

Usage: python v0_basic.py
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr, # constexpr means that the value is known at compile time, unlike a regular argument, which is only known at runtime (e.g. here BLOCK_SIZE is known at compile time, but n_elements is only known at runtime)
):
    
    pid = tl.program_id(axis=0) 
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # last block may run past the end

    x = tl.load(x_ptr + offsets, mask=mask) 
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, 
             x + y, 
             mask=mask)


def add(x: torch.Tensor, 
        y: torch.Tensor, 
        BLOCK_SIZE=32):

    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA"
    assert x.shape == y.shape, "Input tensors must have the same shape"

    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output


def test_add_kernel(size: int):
    torch.manual_seed(0)
    x = torch.randn(size, 
                    device=DEVICE,
                    dtype=torch.float32)
    y = torch.randn(size, 
                    device=DEVICE, 
                    dtype=torch.float32)

    output_torch = x + y
    output_triton = add(x, y)

    triton.testing.assert_close(output_triton, output_torch)
    print(f"size={size:>10}  OK")


if __name__ == "__main__":
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    for n in (32, 64, 128, 256, 512, 1024, 2048, 1024 * 1024 + 7):  # last size is not a multiple of BLOCK_SIZE
        test_add_kernel(size=n)
