"""
SiLU activation: silu(x) = x * sigmoid(x).

Still embarrassingly parallel like vector add, but the compute step now
chains several arithmetic ops (negate, exp, add, div, mul) entirely in
registers before the single store.

Run: python v0_basic.py
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def silu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * tl.sigmoid(x)
    tl.store(output_ptr + offsets, y, mask=mask)


def silu(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    silu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def test_silu_kernel(size: int):
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)

    triton.testing.assert_close(silu(x), torch.nn.functional.silu(x), atol=1e-3, rtol=1e-3)
    print(f"size={size:>10}  OK")


if __name__ == "__main__":
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    for n in (1, 128, 1024, 1024 * 1024 + 7):
        test_silu_kernel(n)
