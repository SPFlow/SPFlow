#!/usr/bin/env python3
"""
Benchmark differentiable SIMPLE sampling in SumLayer:
1) Baseline: one-hot sample + index_one_hot_fast
2) Fused: simple_straight_through_select (no explicit one-hot sample tensor)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIMPLE_EINET_PARENT = os.path.join(REPO_ROOT, "reference-repos", "autoencoding-pcs")
if SIMPLE_EINET_PARENT not in sys.path:
    sys.path.insert(0, SIMPLE_EINET_PARENT)

from simple_einet.layers.sum import SumLayer
from simple_einet.sampling_utils import (
    SamplingContext,
    index_one_hot_fast,
    sample_categorical_differentiably,
    simple_straight_through_select,
)


@dataclass
class BenchmarkResult:
    name: str
    mean_ms: float
    std_ms: float
    memory_bytes: int


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def build_context(
    num_samples: int,
    num_features_out: int,
    num_sums_out: int,
    num_repetitions: int,
    device: torch.device,
) -> SamplingContext:
    indices_out = torch.randint(0, num_sums_out, (num_samples, num_features_out), device=device)
    indices_out_oh = F.one_hot(indices_out, num_classes=num_sums_out).float()

    indices_repetition = torch.randint(0, num_repetitions, (num_samples,), device=device)
    indices_repetition_oh = F.one_hot(indices_repetition, num_classes=num_repetitions).float()

    return SamplingContext(
        num_samples=num_samples,
        indices_out=indices_out_oh,
        indices_repetition=indices_repetition_oh,
        is_differentiable=True,
        is_mpe=False,
        temperature_sums=1.0,
    )


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def step_baseline(layer: SumLayer, ctx: SamplingContext, readout: torch.Tensor) -> None:
    layer.zero_grad(set_to_none=True)
    log_weights = layer._select_weights(ctx, layer.logits)
    indices = sample_categorical_differentiably(
        dim=-1,
        is_mpe=ctx.is_mpe,
        hard=ctx.hard,
        tau=ctx.tau,
        log_weights=log_weights,
    )
    selected = index_one_hot_fast(readout, index=indices, dim=-1)
    loss = selected.square().mean()
    loss.backward()


def step_fused(layer: SumLayer, ctx: SamplingContext, readout: torch.Tensor) -> None:
    layer.zero_grad(set_to_none=True)
    log_weights = layer._select_weights(ctx, layer.logits)
    selected = simple_straight_through_select(
        tensor=readout,
        log_weights=log_weights,
        dim=-1,
        is_mpe=ctx.is_mpe,
    )
    loss = selected.square().mean()
    loss.backward()


def measure_memory_bytes(fn: Callable[[], None], device: torch.device) -> int:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        fn()
        sync_if_cuda(device)
        return int(torch.cuda.max_memory_allocated(device))

    # CPU fallback: report max per-op memory attribution from torch profiler.
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False) as prof:
        fn()
    return int(max(evt.cpu_memory_usage for evt in prof.key_averages()))


def run_benchmark(
    fn: Callable[[], None],
    device: torch.device,
    warmup: int,
    iters: int,
    name: str,
) -> BenchmarkResult:
    for _ in range(warmup):
        fn()
    sync_if_cuda(device)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync_if_cuda(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    mem_bytes = measure_memory_bytes(fn, device)
    t = torch.tensor(times)
    return BenchmarkResult(
        name=name,
        mean_ms=float(t.mean().item()),
        std_ms=float(t.std(unbiased=False).item()),
        memory_bytes=mem_bytes,
    )


def main() -> None:
    raise RuntimeError(
        "This benchmark targets differentiable-sampling paths that are unavailable after the rollback."
    )
    parser = argparse.ArgumentParser(description="Benchmark SIMPLE differentiable sampling in SumLayer.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--num-features", type=int, default=256)
    parser.add_argument("--num-sums-in", type=int, default=64)
    parser.add_argument("--num-sums-out", type=int, default=64)
    parser.add_argument("--num-repetitions", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    layer = SumLayer(
        num_sums_in=args.num_sums_in,
        num_features=args.num_features,
        num_sums_out=args.num_sums_out,
        num_repetitions=args.num_repetitions,
    ).to(device)

    ctx = build_context(
        num_samples=args.num_samples,
        num_features_out=layer.num_features_out,
        num_sums_out=layer.num_sums_out,
        num_repetitions=layer.num_repetitions,
        device=device,
    )

    # Mimics a downstream tensor indexed by sampled child choices.
    readout = torch.randn(
        args.num_samples,
        layer.num_features_out,
        layer.num_sums_in,
        device=device,
    )

    baseline = run_benchmark(
        fn=lambda: step_baseline(layer, ctx, readout),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        name="baseline_one_hot",
    )
    fused = run_benchmark(
        fn=lambda: step_fused(layer, ctx, readout),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        name="fused_custom_backward",
    )

    print(f"device={device}")
    print(
        "config:",
        {
            "num_samples": args.num_samples,
            "num_features": args.num_features,
            "num_sums_in": args.num_sums_in,
            "num_sums_out": args.num_sums_out,
            "num_repetitions": args.num_repetitions,
            "iters": args.iters,
        },
    )
    print()
    print(f"{'name':24s} {'mean_ms':>12s} {'std_ms':>12s} {'memory_mib':>12s}")
    for r in (baseline, fused):
        print(f"{r.name:24s} {r.mean_ms:12.3f} {r.std_ms:12.3f} {bytes_to_mib(r.memory_bytes):12.3f}")

    speedup = baseline.mean_ms / fused.mean_ms if fused.mean_ms > 0 else float("inf")
    mem_ratio = baseline.memory_bytes / fused.memory_bytes if fused.memory_bytes > 0 else float("inf")
    print()
    print(f"speedup(fused_vs_baseline): {speedup:.3f}x")
    print(f"memory_ratio(baseline/fused): {mem_ratio:.3f}x")

    if device.type == "cpu":
        print("note: CPU memory uses profiler per-op attribution, not CUDA peak allocator memory.")


if __name__ == "__main__":
    main()
