import torch
import triton
import triton.language as tl
import timeit
from einops import einsum


@triton.jit
def weighted_sum_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One Triton program handles one row of X.
    row = tl.program_id(0)

    # Tile of columns handled by this program.
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # X[row, cols]
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)

    # w[cols]
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)

    # Elementwise weighted values
    weighted = x * w

    # Reduce the tile to one value.
    result = tl.sum(weighted, axis=0)

    tl.store(y_ptr + row, result)


def weighted_sum(x, w):
    M, N = x.shape

    BLOCK_SIZE = triton.next_power_of_2(N)

    y = torch.empty(M, device=x.device, dtype=x.dtype)

    weighted_sum_kernel[(M,)](
        x,
        w,
        y,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y


def torch_weighted_sum(x, w):
    return torch.sum(x * w, dim=1)


@torch.compile
def torch_weighted_sum_compile(x, w):
    return torch.sum(x * w, dim=1)


def torch_einsum(x, w):
    return einsum(x, w, "... d, d -> ...")


def benchmark(M, N):
    x = torch.randn((M, N), device="cuda", dtype=torch.float32)
    w = torch.randn((N,), device="cuda", dtype=torch.float32)

    # Correctness
    y_triton = weighted_sum(x, w)
    y_torch = torch_weighted_sum(x, w)
    y_torch_compile = torch_weighted_sum_compile(x, w)
    y_einsum = torch_einsum(x, w)

    torch.testing.assert_close(y_triton, y_torch, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(y_triton, y_torch_compile, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(y_triton, y_einsum, rtol=1e-4, atol=1e-4)

    # Benchmark
    torch_ms = triton.testing.do_bench(
        lambda: torch_weighted_sum(x, w)
    )

    torch_compile_ms = triton.testing.do_bench(
        lambda: torch_weighted_sum_compile(x, w)
    )

    einsum_ms = triton.testing.do_bench(
        lambda: torch_einsum(x, w)
    )

    triton_ms = triton.testing.do_bench(
        lambda: weighted_sum(x, w)
    )

    print(f"M={M}, N={N}")
    print(f"PyTorch: {torch_ms * 1000:.2f} us")
    print(f"PyTorch compile: {torch_compile_ms * 1000:.2f} us")
    print(f"einsum: {einsum_ms * 1000:.2f} us")
    print(f"Triton:  {triton_ms * 1000:.2f} us")
    print(f"Speedup: {torch_ms / triton_ms:.2f}x")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[256, 512, 1024, 2048, 4096, 8192, 16384],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch", "Triton"],
        ylabel="GB/s",
        plot_name="weighted-sum-performance",
        args={"M": 4096},
    )
)
def benchmark1(M, N, provider):
    x = torch.randn((M, N), device="cuda")
    w = torch.randn((N,), device="cuda")

    if provider == "torch":
        ms = triton.testing.do_bench(
            lambda: torch_weighted_sum(x, w)
        )
    else:
        ms = triton.testing.do_bench(
            lambda: weighted_sum(x, w)
        )

    bytes_processed = 2 * M * N * x.element_size()

    gbps = bytes_processed / (ms * 1e-3) / 1e9

    return gbps


if __name__ == "__main__":
    for N in [256, 512, 1024, 4096, 16384]:
        benchmark(4096, N)

    benchmark1.run(show_plots=True, print_data=True)