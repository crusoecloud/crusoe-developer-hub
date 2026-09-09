"""
SwiGLU, fused: swiglu(x) = silu(gate) * value, where `gate` and `value` are
the two contiguous halves of x.

One kernel launch. Both halves are loaded, the SiLU activation and the
multiply happen entirely in registers, and only the final result is stored -
no intermediate ever touches global memory.

Run: python v1_fused.py
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def swiglu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    half = n_elements // 2
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < half

    gate = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    value = tl.load(x_ptr + half + offsets, mask=mask, other=0.0)

    silu = gate * tl.sigmoid(gate)
    tl.store(output_ptr + offsets, silu * value, mask=mask)


def swiglu_fused(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    output = torch.empty(n // 2, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(n // 2, meta["BLOCK_SIZE"]),)
    swiglu_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output


def test_swiglu_fused(size: int):
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)
    gate, value = x.chunk(2)
    expected = torch.nn.functional.silu(gate) * value

    triton.testing.assert_close(swiglu_fused(x), expected, atol=1e-3, rtol=1e-3)
    print(f"size={size:>10}  OK")


if __name__ == "__main__":
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    for n in (128, 1024, 2 * (1024 * 1024 + 8)):
        test_swiglu_fused(n)
