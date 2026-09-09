# GPU Programming

GPU programming, CUDA concepts, and GPU kernels written in
[Triton](https://triton-lang.org/). Topics are grouped into directories
roughly in order of difficulty, and most are split into versions so you can
see a kernel evolve step by step: a bare correctness check first, then a
benchmark, then (where it's useful) a look at what the compiler produced.

Every script is self-contained, runnable on its own, and documented inline,
so someone new to GPU programming can read one top to bottom and follow
along without external context.

## Index

### Part 1: Foundations of GPU Programming

1. [Vector Addition](./01_vector_add/): the first kernel. Program IDs, blocks,
   masking, and launching a grid.
2. [SiLU Activation](./02_silu/): an elementwise activation kernel. Same
   memory pattern as vector add, more arithmetic per element.
3. [Fused SwiGLU](./03_swiglu_fusion/): fusing two elementwise kernels into
   one to cut memory traffic, with a benchmark showing the difference.
4. [Masking with `tl.where`](./04_masking_where/): replacing a per-element
   branch with `tl.where` to avoid warp divergence.
5. [Rotary Positional Embedding](./05_rope/): mapping logical coordinates to
   memory offsets when the layout isn't a straight one-to-one index.

### Part 2: Advanced Kernels

Coming soon.

### Part 3: Triton Integration with PyTorch

Coming soon.

### Part 4: Triton for Production

Coming soon.

## Setup

Requires [`uv`](https://github.com/astral-sh/uv) and an NVIDIA GPU with a working CUDA environment. The scripts explicitly select `cuda:0`. No Crusoe API key is required. If you use a cloud GPU, its compute and storage can incur charges.

```bash
# from this directory
uv venv .venv --python 3.12
uv pip install -r requirements.txt --python .venv/bin/python
```

## Running a script

```bash
.venv/bin/python 01_vector_add/v0_basic.py
```

The first script prints the GPU name and `OK` for each vector size that passes its correctness check. Benchmark scripts add timings and may write results to a local `results/` directory.

Work through the directories in order, and within a directory work through
`v0`, `v1`, `v2`, ... in order.

## Jupyter / VS Code

To use this environment in Jupyter or VS Code, register its kernel and select
**"gpu-programming (.venv)"** from the kernel picker:

```bash
uv pip install ipykernel --python .venv/bin/python
.venv/bin/python -m ipykernel install --user --name=gpu-programming --display-name="gpu-programming (.venv)"
```

After a session, keep any benchmark results you need and remove generated results when they are no longer useful. Delete any cloud resources you created once you have saved your work; exiting Python does not remove them.

[All foundations](../README.md)
