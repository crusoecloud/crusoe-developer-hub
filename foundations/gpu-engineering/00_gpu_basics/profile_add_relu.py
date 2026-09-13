"""Profile c = (a + b).relu() once and print the profiler's summary table.

Usage: python profile_add_relu.py [n_elements]
"""

import sys

import torch
from torch.profiler import ProfilerActivity, profile

DEVICE = torch.device("cuda:0")


def add_relu(a, b):
    return (a + b).relu()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    a = torch.randn(n, device=DEVICE)
    b = torch.randn(n, device=DEVICE)
    add_relu(a, b)  # warm up once so allocator setup stays out of the trace

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        add_relu(a, b)
        torch.cuda.synchronize()  # wait for the GPU before reading the trace

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
