from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from lpap.ops import lpap_torch
from lpap.triton_ops import lpap_triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LPAP PyTorch and Triton ops."
    )
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--values", type=int, default=1024)
    parser.add_argument("--buckets", type=int, default=64)
    parser.add_argument("--k-max", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def benchmark_cpu(
    name: str,
    op: Callable[[torch.Tensor], object],
    values: torch.Tensor,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        op(values)
    started = time.perf_counter()
    for _ in range(iters):
        op(values)
    elapsed = time.perf_counter() - started
    milliseconds = elapsed * 1000 / iters
    print(f"{name}: {milliseconds:.3f} ms")
    return milliseconds


def benchmark_cuda(
    name: str,
    op: Callable[[torch.Tensor], object],
    values: torch.Tensor,
    warmup: int,
    iters: int,
) -> float:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        op(values)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(iters):
        op(values)
    end_event.record()
    torch.cuda.synchronize()
    milliseconds = start_event.elapsed_time(end_event) / iters
    print(f"{name}: {milliseconds:.3f} ms")
    return milliseconds


def main() -> int:
    args = parse_args()
    if args.values % args.buckets != 0:
        raise ValueError("--values must be divisible by --buckets")

    cpu_values = torch.randn(args.batch, args.values)

    def torch_op(input_values: torch.Tensor) -> object:
        return lpap_torch(input_values, bucket_count=args.buckets, k_max=args.k_max)

    benchmark_cpu("torch/cpu", torch_op, cpu_values, args.warmup, args.iters)

    if torch.cuda.is_available():
        cuda_values = cpu_values.cuda()

        def torch_cuda_op(input_values: torch.Tensor) -> object:
            return lpap_torch(input_values, bucket_count=args.buckets, k_max=args.k_max)

        def triton_op(input_values: torch.Tensor) -> object:
            return lpap_triton(
                input_values, bucket_count=args.buckets, k_max=args.k_max
            )

        benchmark_cuda(
            "torch/cuda", torch_cuda_op, cuda_values, args.warmup, args.iters
        )
        benchmark_cuda("triton/cuda", triton_op, cuda_values, args.warmup, args.iters)
    else:
        print("CUDA is not available; skipped Triton benchmark")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
