"""Show why wall-clock timing around a launch measures the CPU, not the GPU.

Usage: python async_timing.py
"""

import time

import torch

DEVICE = torch.device("cuda:0")

if __name__ == "__main__":
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    a = torch.randn(50_000_000, device=DEVICE)
    b = torch.randn(50_000_000, device=DEVICE)
    a + b  # warm up
    torch.cuda.synchronize()

    t0 = time.time()
    c = a + b
    t_launch = time.time() - t0  # how long the CPU spent issuing the work

    torch.cuda.synchronize()     # block until the GPU has finished everything queued
    t_done = time.time() - t0    # how long the GPU actually took, plus the launch

    print(f"launch returned after {t_launch * 1e6:7.0f} us")
    print(f"work finished after   {t_done * 1e6:7.0f} us")
