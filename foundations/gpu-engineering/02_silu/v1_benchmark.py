"""
Same SiLU kernel as v0, benchmarked against torch.nn.functional.silu.

SiLU does more math per element than vector add (negate, exp, add, div,
mul vs. a single add) but touches the same amount of memory (one load, one
store), so it's a useful data point for how much arithmetic a memory-bound
kernel can absorb "for free".

Run: python v1_benchmark.py
"""

import os

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(10, 25)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="silu_performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.04, 0.95]
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: silu(x), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.silu(x), quantiles=quantiles
        )

    # 2 arrays touched (x, output) at 4 bytes/element
    gbps = lambda ms: 2 * size * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)


if __name__ == "__main__":
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    for n in (1024, 1024 * 1024 + 7):
        test_silu_kernel(n)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    benchmark.run(save_path=RESULTS_DIR, print_data=True)
