"""Fused softmax benchmarked against torch.softmax and an unfused PyTorch version across row widths.

Usage: python v1_benchmark.py
"""

import os

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_ROWS = 4096


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
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(input_ptr + row_idx * input_row_stride + col_offsets, mask=mask, other=0.0)
    row = tl.where(mask, row, float("-inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
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


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    # each line is its own kernel, and each intermediate round-trips through global memory
    x_max = x.max(dim=1, keepdim=True)[0]
    numerator = torch.exp(x - x_max)
    denominator = numerator.sum(dim=1, keepdim=True)
    return numerator / denominator


def test_softmax_kernel(n_rows: int, n_cols: int):
    torch.manual_seed(0)
    x = torch.randn(n_rows, n_cols, device=DEVICE)

    triton.testing.assert_close(softmax(x), torch.softmax(x, dim=1), atol=1e-5, rtol=1e-4)
    print(f"n_rows={n_rows:>5}  n_cols={n_cols:>5}  OK")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_cols"],
        x_vals=[256 * i for i in range(1, 33)],  # 256 to 8192 columns
        line_arg="provider",
        line_vals=["triton", "torch", "naive"],
        line_names=["Triton", "Torch", "Naive torch"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="GB/s",
        plot_name="softmax_performance",
        args={"n_rows": N_ROWS},
    )
)
def benchmark(n_rows, n_cols, provider):
    x = torch.randn(n_rows, n_cols, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    elif provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)

    # minimum traffic: read x once, write output once; the naive version moves more but is credited the same
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)


if __name__ == "__main__":
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    for n_rows, n_cols in ((7, 19), (64, 4097)):
        test_softmax_kernel(n_rows, n_cols)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    benchmark.run(save_path=RESULTS_DIR, print_data=True)
