"""Vector addition benchmarked against torch's add across sizes, reported in GB/s.

Usage: python v1_benchmark.py
"""

import os

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def test_add_kernel(size: int):
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)

    triton.testing.assert_close(add(x, y), x + y, atol=1e-3, rtol=1e-3)
    print(f"size={size:>10}  OK")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(10, 25)],  # 1K to 16M elements
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="vector_add_performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.randn(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)

    # 3 arrays touched (x, y, output) at 4 bytes/element
    gbps = lambda ms: 3 * size * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)


if __name__ == "__main__":
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    for n in (1024, 1024 * 1024 + 7):
        test_add_kernel(n)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    benchmark.run(save_path=RESULTS_DIR, print_data=True)
