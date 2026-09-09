"""
Same RoPE kernel as v0, benchmarked against the chunk/cat reference
implementation across sequence lengths (feature_dim fixed at a typical
attention head size).

Run: python v1_benchmark.py
"""

import os

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FEATURE_DIM = 128


@triton.jit
def rope_kernel(
    x_ptr,
    output_ptr,
    cos_ptr,
    sin_ptr,
    seq_len,
    feature_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= seq_len:
        return

    half_dim = feature_dim // 2
    local_idx = tl.arange(0, BLOCK_SIZE)
    mask = local_idx < half_dim

    row_start = pid * feature_dim
    first_offset = row_start + local_idx
    second_offset = first_offset + half_dim
    x1 = tl.load(x_ptr + first_offset, mask=mask, other=0.0)
    x2 = tl.load(x_ptr + second_offset, mask=mask, other=0.0)

    angle_offset = pid * half_dim + local_idx
    cos_val = tl.load(cos_ptr + angle_offset, mask=mask, other=0.0)
    sin_val = tl.load(sin_ptr + angle_offset, mask=mask, other=0.0)

    out1 = x1 * cos_val - x2 * sin_val
    out2 = x2 * cos_val + x1 * sin_val
    tl.store(output_ptr + first_offset, out1, mask=mask)
    tl.store(output_ptr + second_offset, out2, mask=mask)


def rope_freqs(seq_len: int, feature_dim: int, base: float = 10000.0, device=DEVICE):
    half_dim = feature_dim // 2
    omega = 1.0 / base ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) * 2 / feature_dim)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = positions[:, None] * omega[None, :]
    return angles.cos(), angles.sin()


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len, feature_dim = x.shape
    output = torch.empty_like(x)
    half_dim = feature_dim // 2
    grid = (seq_len,)
    rope_kernel[grid](
        x, output, cos, sin, seq_len, feature_dim, BLOCK_SIZE=triton.next_power_of_2(half_dim)
    )
    return output


def rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


def test_rope_kernel(seq_len: int, feature_dim: int):
    torch.manual_seed(0)
    x = torch.randn(seq_len, feature_dim, device=DEVICE)
    cos, sin = rope_freqs(seq_len, feature_dim)

    triton.testing.assert_close(rope(x, cos, sin), rope_reference(x, cos, sin), atol=1e-3, rtol=1e-3)
    print(f"seq_len={seq_len:>6}  feature_dim={feature_dim:>5}  OK")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2 ** i for i in range(6, 16)],  # 64 to 32K positions
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="rope_performance",
        args={"feature_dim": FEATURE_DIM},
    )
)
def benchmark(seq_len, feature_dim, provider):
    x = torch.randn(seq_len, feature_dim, device=DEVICE, dtype=torch.float32)
    cos, sin = rope_freqs(seq_len, feature_dim)

    quantiles = [0.5, 0.04, 0.95]
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope(x, cos, sin), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rope_reference(x, cos, sin), quantiles=quantiles
        )

    # per row: read x (feature_dim), read cos+sin (2 * half_dim), write output (feature_dim)
    bytes_per_row = (2 * feature_dim + feature_dim) * 4
    gbps = lambda ms: seq_len * bytes_per_row * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)


if __name__ == "__main__":
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    for seq_len in (16, 2048):
        test_rope_kernel(seq_len, FEATURE_DIM)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    benchmark.run(save_path=RESULTS_DIR, print_data=True)
