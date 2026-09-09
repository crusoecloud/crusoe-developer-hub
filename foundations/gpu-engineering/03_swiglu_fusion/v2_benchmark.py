"""
Unfused vs. fused SwiGLU vs. PyTorch, benchmarked head to head.

Per output element, the unfused path moves 5 values (read gate, write
intermediate, read intermediate, read value, write output) while the fused
path moves only 3 (read gate, read value, write output) - a 5:3 ratio that
should show up directly as a throughput gap between the two.

Run: python v2_benchmark.py
"""

import os

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


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


def swiglu_unfused(x: torch.Tensor) -> torch.Tensor:
    half = x.numel() // 2
    gate, value = x[:half], x[half:]
    intermediate = torch.empty_like(gate)
    output = torch.empty_like(gate)
    grid = lambda meta: (triton.cdiv(half, meta["BLOCK_SIZE"]),)
    silu_kernel[grid](gate, intermediate, half, BLOCK_SIZE=1024)
    mul_kernel[grid](intermediate, value, output, half, BLOCK_SIZE=1024)
    return output


def swiglu_fused(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    output = torch.empty(n // 2, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(n // 2, meta["BLOCK_SIZE"]),)
    swiglu_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output


def swiglu_torch(x: torch.Tensor) -> torch.Tensor:
    gate, value = x.chunk(2)
    return torch.nn.functional.silu(gate) * value


def test_swiglu(size: int):
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)
    expected = swiglu_torch(x)

    triton.testing.assert_close(swiglu_unfused(x), expected, atol=1e-3, rtol=1e-3)
    triton.testing.assert_close(swiglu_fused(x), expected, atol=1e-3, rtol=1e-3)
    print(f"size={size:>10}  OK")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2 ** i for i in range(11, 26)],  # total elements, must be even
        x_log=True,
        line_arg="provider",
        line_vals=["fused", "unfused", "torch"],
        line_names=["Triton (fused)", "Triton (unfused)", "Torch"],
        styles=[("blue", "-"), ("red", "--"), ("green", "-.")],
        ylabel="ms",
        plot_name="swiglu_performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.04, 0.95]
    fn = {"fused": swiglu_fused, "unfused": swiglu_unfused, "torch": swiglu_torch}[provider]
    # raw latency, not derived throughput: fusion's win is fewer bytes moved for the
    # same output, which shows up directly as lower ms, not as a higher GB/s number.
    return triton.testing.do_bench(lambda: fn(x), quantiles=quantiles)


if __name__ == "__main__":
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    for n in (1024, 2 * (1024 * 1024 + 8)):
        test_swiglu(n)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    benchmark.run(save_path=RESULTS_DIR, print_data=True)
