"""Pick the best available device and enable TF32 matmuls on NVIDIA GPUs."""

import torch


def get_device(verbose: bool = True) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        name = "Apple Silicon GPU"
    else:
        device = torch.device("cpu")
        name = "CPU"
    if verbose:
        print(f"device: {device} ({name})")
    return device


def peak_memory_gb(device: torch.device) -> float | None:
    """Peak allocated memory on a CUDA device since the last reset, in GB."""
    if device.type != "cuda":
        return None
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    torch.cuda.reset_peak_memory_stats()
    return peak
