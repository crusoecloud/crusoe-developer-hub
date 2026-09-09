"""
Value clipping: output[i] = min(max(x[i], v_min), v_max).

A per-element conditional is the classic case for warp divergence: threads
in the same warp can land on either side of the bound, and a Python-style
if/else would force the hardware to run both branches serially. tl.where
keeps every lane on one instruction stream by computing both candidates and
selecting per-lane, at the cost of always evaluating both sides.

Run: python v0_basic.py
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def clip_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    v_min,
    v_max,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    clamped_low = tl.where(x < v_min, v_min, x)
    clamped = tl.where(clamped_low > v_max, v_max, clamped_low)
    tl.store(output_ptr + offsets, clamped, mask=mask)


def clip(x: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    clip_kernel[grid](x, output, n_elements, v_min, v_max, BLOCK_SIZE=1024)
    return output


def test_clip_kernel(size: int, v_min: float = -0.5, v_max: float = 0.5):
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)

    triton.testing.assert_close(clip(x, v_min, v_max), x.clamp(v_min, v_max), atol=1e-3, rtol=1e-3)
    print(f"size={size:>10}  OK")


if __name__ == "__main__":
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    for n in (1, 128, 1024, 1024 * 1024 + 7):
        test_clip_kernel(n)
