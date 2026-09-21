"""Vector addition benchmarked against torch's add across sizes, reported in GB/s.
 
Usage: python v1_benchmark.py
"""
 
import os
import time
 
import torch
import triton
import triton.language as tl
 
DEVICE = torch.device("cuda:0")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
 
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,  # constexpr means that the value is known at compile time, unlike a regular argument, which is only known at runtime (e.g. here BLOCK_SIZE is known at compile time, but n_elements is only known at runtime)
):
 
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # last block may run past the end
 
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets,
             x + y,
             mask=mask)
 
 
def add(x: torch.Tensor,
        y: torch.Tensor,
        BLOCK_SIZE=1024):
 
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA"
    assert x.shape == y.shape, "Input tensors must have the same shape"
 
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
 
    return output
 
 
def test_add_kernel(size: int):
    torch.manual_seed(0)
    x = torch.randn(size,
                    device=DEVICE,
                    dtype=torch.float32)
    y = torch.randn(size,
                    device=DEVICE,
                    dtype=torch.float32)
 
    output_torch = x + y
    output_triton = add(x, y)
 
    triton.testing.assert_close(output_triton, output_torch)
    print(f"size={size:>10}  OK")
 
 
def bench_cpu(fn, n_warmup=5, n_reps=20):
    """Time a CPU function with perf_counter; return (median, p20, p80) in ms.
 
    do_bench is built for GPU kernels (CUDA-event timing, L2 cache flushes),
    so CPU code gets a plain wall-clock loop instead. Quantiles are ordered
    to match do_bench with quantiles=[0.5, 0.2, 0.8].
    """
    for _ in range(n_warmup):
        fn()
    times_ms = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1e3)
    times_ms.sort()
    return (times_ms[n_reps // 2],
            times_ms[int(n_reps * 0.2)],
            times_ms[int(n_reps * 0.8)])
 
 
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(10, 25)],  
        x_log=True,
        line_arg="provider",
        line_vals=["python", "triton", "torch"],
        line_names=["Python (CPU)", "Triton (GPU)", "Torch (GPU)"],
        styles=[("orange", "-"), ("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector_add_performance",
        args={},
    )
)
def benchmark(size, provider):
    quantiles = [0.5, 0.2, 0.8]
    x_gpu = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y_gpu = torch.randn(size, device=DEVICE, dtype=torch.float32)
 
    if provider == "python":
        x_list = x_gpu.cpu().tolist()
        y_list = y_gpu.cpu().tolist()
        ms, min_ms, max_ms = bench_cpu(
            lambda: [a + b for a, b in zip(x_list, y_list)])
 
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x_gpu, y_gpu),
            warmup=20,  # in milliseconds
            rep=100,    # in milliseconds
            quantiles=quantiles)
 
    elif provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x_gpu + y_gpu,
            warmup=20, 
            rep=100,   
            quantiles=quantiles)
 
    # 3 arrays touched (x, y, output) at 4 bytes/element
    gbps = lambda ms: 3 * size * x_gpu.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)
 
 
if __name__ == "__main__":
    print(f"Running on {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    # for n in (32, 64, 128, 256, 512, 1024, 2048, 1024 * 1024 + 7):  # last size is not a multiple of BLOCK_SIZE
    #     test_add_kernel(size=n)
 
    os.makedirs(RESULTS_DIR, exist_ok=True)
    benchmark.run(save_path=RESULTS_DIR, print_data=True, show_plots=False)