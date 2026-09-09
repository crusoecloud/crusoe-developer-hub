"""SwiGLU as two kernels, silu then multiply; the intermediate round-trips through global memory.

Usage: python v0_unfused.py
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def silu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x * tl.sigmoid(x), mask=mask)


@triton.jit
def mul_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, a * b, mask=mask)


def swiglu_unfused(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    half = n // 2
    gate, value = x[:half], x[half:]

    intermediate = torch.empty_like(gate)
    output = torch.empty_like(gate)
    grid = lambda meta: (triton.cdiv(half, meta["BLOCK_SIZE"]),)

    silu_kernel[grid](gate, intermediate, half, BLOCK_SIZE=1024)
    mul_kernel[grid](intermediate, value, output, half, BLOCK_SIZE=1024)
    return output


def test_swiglu_unfused(size: int):
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)
    gate, value = x.chunk(2)
    expected = torch.nn.functional.silu(gate) * value

    triton.testing.assert_close(swiglu_unfused(x), expected, atol=1e-3, rtol=1e-3)
    print(f"size={size:>10}  OK")


if __name__ == "__main__":
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    for n in (128, 1024, 2 * (1024 * 1024 + 8)):  # sizes must be even
        test_swiglu_unfused(n)
