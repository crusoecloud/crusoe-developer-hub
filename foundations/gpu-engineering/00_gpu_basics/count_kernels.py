"""Count the GPU kernels behind one line of PyTorch, eager and compiled.

Usage: python count_kernels.py [n_elements]

Prints the names of the kernels the GPU executed for c = (a + b).relu(), first in
eager mode (expect two kernels) and then under torch.compile (expect one).
"""

import sys

import torch
from torch.profiler import ProfilerActivity, profile

DEVICE = torch.device("cuda:0")


def add_relu(a, b):
    return (a + b).relu()


def gpu_kernel_rows(fn, *args):
    """Run fn once under the profiler and return the names of the kernels the GPU executed."""
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn(*args)
        torch.cuda.synchronize()  # wait for the GPU before reading the trace
    return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    a = torch.randn(n, device=DEVICE)
    b = torch.randn(n, device=DEVICE)

    add_relu(a, b)  # warm up once so allocator setup stays out of the trace

    eager = gpu_kernel_rows(add_relu, a, b)
    print(f"\neager: {len(eager)} kernel(s)")
    for name in eager:
        print(f"  {name}")

    compiled = torch.compile(add_relu)
    compiled(a, b)  # the first call compiles; never profile that one
    fused = gpu_kernel_rows(compiled, a, b)
    print(f"\ntorch.compile: {len(fused)} kernel(s)")
    for name in fused:
        print(f"  {name}")
