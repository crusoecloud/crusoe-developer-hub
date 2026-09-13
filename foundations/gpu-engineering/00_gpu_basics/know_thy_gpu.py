"""Print the handful of hardware facts the lessons keep referring to.

Usage: python know_thy_gpu.py
"""

import torch

DEVICE = torch.device("cuda:0")

if __name__ == "__main__":
    p = torch.cuda.get_device_properties(DEVICE)
    print(f"GPU                       {p.name}")
    print(f"Streaming multiprocessors {p.multi_processor_count}")
    print(f"Warp size                 {p.warp_size} threads")
    print(f"Max threads per SM        {p.max_threads_per_multi_processor}")
    print(f"Warp slots per SM         {p.max_threads_per_multi_processor // p.warp_size}")
    print(f"Global memory             {p.total_memory / 2**30:.0f} GiB")
    print(f"L2 cache                  {p.L2_cache_size / 2**20:.0f} MiB")
    print(f"Compute capability        {p.major}.{p.minor}")

    # How many blocks does a one-million-element launch create at 256 threads per block?
    n, block = 1_000_000, 256
    blocks = (n + block - 1) // block
    warps = blocks * block // p.warp_size
    resident = p.multi_processor_count * (p.max_threads_per_multi_processor // p.warp_size)
    print(f"\n{n:,} elements at {block} threads per block")
    print(f"  blocks in the grid      {blocks:,}")
    print(f"  warps in the grid       {warps:,}")
    print(f"  warp slots on the GPU   {resident:,}  ({p.multi_processor_count} SMs x {p.max_threads_per_multi_processor // p.warp_size})")
    print(f"  waves, at best          {warps / resident:.1f}")
