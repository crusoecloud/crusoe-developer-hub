"""Row-wise softmax fused into one kernel, one program per row, with a correctness check.

Usage: python v0_basic.py
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)         # BLOCK_SIZE >= n_cols, so one block covers the whole row
    mask = col_offsets < n_cols

    row = tl.load(input_ptr + row_idx * input_row_stride + col_offsets, mask=mask, other=0.0)
    row = tl.where(mask, row, float("-inf"))       # padding must never win the max
    row_minus_max = row - tl.max(row, axis=0)      # subtract the max for numerical stability

    numerator = tl.exp(row_minus_max)              # exp(-inf) = 0, so padding adds nothing to the sum
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(output_ptr + row_idx * output_row_stride + col_offsets, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE <= 2048 else 8 if BLOCK_SIZE <= 8192 else 16
    grid = (n_rows,)
    softmax_kernel[grid](
        x, output, n_cols, x.stride(0), output.stride(0), BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )
    return output


def test_softmax_kernel(n_rows: int, n_cols: int):
    torch.manual_seed(0)
    x = torch.randn(n_rows, n_cols, device=DEVICE)

    triton.testing.assert_close(softmax(x), torch.softmax(x, dim=1), atol=1e-5, rtol=1e-4)
    print(f"n_rows={n_rows:>5}  n_cols={n_cols:>5}  OK")


if __name__ == "__main__":
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    for n_rows, n_cols in ((1, 1), (7, 19), (128, 1000), (64, 4097), (1024, 4096)):  # odd widths exercise the padding
        test_softmax_kernel(n_rows, n_cols)
