"""Count the GPU kernels behind one line of PyTorch, eager and compiled.

Usage: python count_kernels.py [n_elements]

Prints the name and GPU time of each kernel the GPU executed for c = (a + b).relu(),
first in eager mode (expect two kernels) and then under torch.compile (expect one).
"""

import sys

import torch
from torch.profiler import ProfilerActivity, profile

DEVICE = torch.device("cuda:0")


def add_relu(a, b):
    return (a + b).relu()


def gpu_kernel_rows(fn, *args):
    """Run fn once under the profiler and return (name, GPU time in us) per kernel."""
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn(*args)
        torch.cuda.synchronize()  # wait for the GPU before reading the trace
    return [
        (e.name, e.device_time)
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
    ]


def report(label, rows, width=60):
    total = sum(t for _, t in rows)
    print(f"\n{label}: {len(rows)} kernel(s), {total:.3f} us on the GPU")
    for name, t in rows:
        if len(name) > width:  # the eager kernel names are template signatures
            name = name[: width - 3] + "..."
        print(f"  {t:8.3f} us  {name}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    a = torch.randn(n, device=DEVICE)
    b = torch.randn(n, device=DEVICE)

    add_relu(a, b)  # warm up once so allocator setup stays out of the trace

    report("eager", gpu_kernel_rows(add_relu, a, b))

    compiled = torch.compile(add_relu)
    compiled(a, b)  # the first call compiles; never profile that one
    report("torch.compile", gpu_kernel_rows(compiled, a, b))
