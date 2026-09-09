"""
Same vector-add kernel as v0, this time inspecting what the compiler
produces at each stage of the lowering pipeline (TTIR -> TTGIR -> LLIR -> PTX).

A kernel launch returns a compiled-kernel object whose `.asm` dict holds the
source for every stage. Dumping them to disk lets you grep through them
without flooding the terminal.

Run: python v2_kernel_inspection.py
"""

import os

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STAGES = ("ttir", "ttgir", "llir", "ptx")


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


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    compiled = add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output, compiled


if __name__ == "__main__":
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")

    torch.manual_seed(0)
    x = torch.randn(1024, device=DEVICE)
    y = torch.randn(1024, device=DEVICE)

    output, compiled = add(x, y)
    triton.testing.assert_close(output, x + y, atol=1e-3, rtol=1e-3)
    print("Correctness OK")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for stage in STAGES:
        path = os.path.join(RESULTS_DIR, f"{stage}.md")
        with open(path, "w") as f:
            f.write(compiled.asm[stage])
        lines = compiled.asm[stage].count("\n")
        print(f"{stage:>5}: {lines:>5} lines -> {path}")
