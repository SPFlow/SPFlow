"""Benchmark and profile the current HistogramDist.log_prob implementation."""

from __future__ import annotations

import argparse
import gc
import resource
import sys
from dataclasses import dataclass
from typing import Callable

import torch
import torch.utils.benchmark as torch_benchmark
from einops import rearrange, repeat
from torch.profiler import ProfilerActivity, profile

from spflow.modules.leaves.histogram import HistogramDist

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


@dataclass(frozen=True)
class WorkloadConfig:
    batch_size: int
    num_channels: int
    num_repetitions: int
    num_bins: int
    input_layout: str


@dataclass(frozen=True)
class BenchmarkSummary:
    mean_ms: float
    median_ms: float
    iqr_ms: float
    rss_delta_mb: float | None
    peak_cuda_allocated_mb: float | None
    peak_cuda_reserved_mb: float | None


WORKLOADS: dict[str, WorkloadConfig] = {
    "small": WorkloadConfig(
        batch_size=256, num_channels=8, num_repetitions=4, num_bins=32, input_layout="nf"
    ),
    "target": WorkloadConfig(
        batch_size=1024, num_channels=16, num_repetitions=16, num_bins=64, input_layout="nf"
    ),
    "large": WorkloadConfig(
        batch_size=2048, num_channels=24, num_repetitions=16, num_bins=128, input_layout="nf"
    ),
    "expanded-target": WorkloadConfig(
        batch_size=1024, num_channels=16, num_repetitions=16, num_bins=64, input_layout="nfcr"
    ),
}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rss_mb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / float(1024**2)


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / float(1024**2)
    return usage / float(1024)


def _cuda_peak_memory_mb(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    scale = float(1024**2)
    return (
        torch.cuda.max_memory_allocated(device) / scale,
        torch.cuda.max_memory_reserved(device) / scale,
    )


def _build_workload(
    config: WorkloadConfig, *, device: torch.device, seed: int
) -> tuple[HistogramDist, torch.Tensor]:
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)

    bin_edges = torch.linspace(-4.0, 4.0, config.num_bins + 1, device=device, dtype=torch.float32)
    logits = torch.randn(
        1,
        config.num_channels,
        config.num_repetitions,
        config.num_bins,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    dist = HistogramDist(bin_edges=bin_edges, logits=logits)

    if config.input_layout == "nf":
        x = torch.empty((config.batch_size, 1), device=device, dtype=torch.float32).uniform_(
            -3.5, 3.5, generator=generator
        )
    elif config.input_layout == "nfcr":
        x = torch.empty(
            (config.batch_size, 1, config.num_channels, config.num_repetitions),
            device=device,
            dtype=torch.float32,
        ).uniform_(-3.5, 3.5, generator=generator)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported input_layout={config.input_layout!r}.")

    flat = x.reshape(-1)
    if flat.numel() >= 3:
        flat[::97] = float("nan")
        flat[1::113] = -4.25
        flat[2::127] = 4.25
    return dist, x


def _reference_log_prob(dist: HistogramDist, x: torch.Tensor) -> torch.Tensor:
    x = dist._align_x(x)
    n_samples = x.shape[0]
    target_shape = (n_samples, *dist._logits.shape[:-1])
    x_broadcast = torch.broadcast_to(x, target_shape).contiguous()

    edges = dist.bin_edges.to(device=x_broadcast.device, dtype=x_broadcast.dtype)
    bin_idx = torch.bucketize(x_broadcast, edges, right=True) - 1
    in_support = torch.isfinite(x_broadcast) & (x_broadcast >= edges[0]) & (x_broadcast < edges[-1])

    densities = dist._bin_densities.to(device=x_broadcast.device, dtype=x_broadcast.dtype)
    densities = repeat(rearrange(densities, "f c r b -> 1 f c r b"), "1 f c r b -> n f c r b", n=n_samples)
    gathered = rearrange(
        densities.gather(-1, rearrange(bin_idx.clamp(0, dist.nbins - 1), "n f c r -> n f c r 1")),
        "n f c r 1 -> n f c r",
    )

    min_density = dist._min_prob / dist._bin_widths.max().to(device=gathered.device, dtype=gathered.dtype)
    log_p = torch.log(gathered.clamp_min(min_density))
    return torch.where(in_support, log_p, x_broadcast.new_full((), float("-inf")))


def _prepare_runner(
    dist: HistogramDist, x: torch.Tensor, *, device: torch.device
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        with torch.inference_mode():
            output = dist.log_prob(x)
        _synchronize(device)
        return output

    return run


def _measure_memory(
    run: Callable[[], torch.Tensor], *, device: torch.device
) -> tuple[float | None, float | None, float | None]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    rss_before = _rss_mb()
    peak_rss_before = _peak_rss_mb()
    _synchronize(device)
    run()
    _synchronize(device)
    rss_after = _rss_mb()
    peak_rss_after = _peak_rss_mb()
    peak_allocated_mb, peak_reserved_mb = _cuda_peak_memory_mb(device)
    rss_delta_mb = max(0.0, peak_rss_after - peak_rss_before)
    if rss_before is not None and rss_after is not None:
        rss_delta_mb = max(rss_delta_mb, max(0.0, rss_after - rss_before))
    return rss_delta_mb, peak_allocated_mb, peak_reserved_mb


def _summarize_timer(
    run: Callable[[], torch.Tensor], *, num_threads: int, min_run_time: float
) -> tuple[float, float, float]:
    measurement = torch_benchmark.Timer(
        stmt="run()",
        globals={"run": run},
        num_threads=num_threads,
    ).blocked_autorange(min_run_time=min_run_time)
    return measurement.mean * 1_000.0, measurement.median * 1_000.0, measurement.iqr * 1_000.0


def command_benchmark(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    torch.set_num_threads(args.num_threads)
    workloads = WORKLOADS.values() if args.workload == "all" else [WORKLOADS[args.workload]]

    for index, config in enumerate(workloads):
        if index:
            print()
        dist, x = _build_workload(config, device=device, seed=args.seed)
        run = _prepare_runner(dist, x, device=device)

        with torch.inference_mode():
            actual = dist.log_prob(x)
            expected = _reference_log_prob(dist, x)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        finite = torch.isfinite(actual)
        max_abs_diff = (
            0.0 if not finite.any() else float((actual[finite] - expected[finite]).abs().max().item())
        )

        mean_ms, median_ms, iqr_ms = _summarize_timer(
            run, num_threads=args.num_threads, min_run_time=args.min_run_time
        )
        rss_delta_mb, peak_allocated_mb, peak_reserved_mb = _measure_memory(run, device=device)
        summary = BenchmarkSummary(
            mean_ms=mean_ms,
            median_ms=median_ms,
            iqr_ms=iqr_ms,
            rss_delta_mb=rss_delta_mb,
            peak_cuda_allocated_mb=peak_allocated_mb,
            peak_cuda_reserved_mb=peak_reserved_mb,
        )

        print(
            f"workload: batch={config.batch_size} channels={config.num_channels} "
            f"repetitions={config.num_repetitions} bins={config.num_bins} layout={config.input_layout}"
        )
        print(f"device: {device.type} threads={args.num_threads}")
        print(f"parity: finite_max_abs_diff={max_abs_diff:.6e}")
        print("impl       mean_ms   median_ms   iqr_ms   rss_delta_mb   peak_alloc_mb   peak_resv_mb")
        rss_text = "n/a".rjust(14) if summary.rss_delta_mb is None else f"{summary.rss_delta_mb:14.3f}"
        alloc_text = (
            "n/a".rjust(16)
            if summary.peak_cuda_allocated_mb is None
            else f"{summary.peak_cuda_allocated_mb:16.3f}"
        )
        resv_text = (
            "n/a".rjust(15)
            if summary.peak_cuda_reserved_mb is None
            else f"{summary.peak_cuda_reserved_mb:15.3f}"
        )
        print(
            f"{'current':<10}"
            f"{summary.mean_ms:9.3f}"
            f"{summary.median_ms:12.3f}"
            f"{summary.iqr_ms:10.3f}"
            f"{rss_text}"
            f"{alloc_text}"
            f"{resv_text}"
        )
    return 0


def command_profile(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if args.workload == "all":
        raise ValueError("profile mode requires a single workload, not --workload all.")

    config = WORKLOADS[args.workload]
    dist, x = _build_workload(config, device=device, seed=args.seed)
    run = _prepare_runner(dist, x, device=device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, profile_memory=True, record_shapes=bool(args.record_shapes)) as prof:
        for _ in range(args.iterations):
            run()

    print(
        f"profile: device={device.type} batch={config.batch_size} channels={config.num_channels} "
        f"repetitions={config.num_repetitions} bins={config.num_bins} layout={config.input_layout}"
    )
    print(prof.key_averages().table(sort_by=args.sort_by, row_limit=args.row_limit))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        subparser.add_argument("--workload", choices=tuple(WORKLOADS.keys()) + ("all",), default="target")
        subparser.add_argument("--seed", type=int, default=0)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run parity checks and timing measurements.")
    add_common_arguments(benchmark_parser)
    benchmark_parser.add_argument("--num-threads", type=int, default=1)
    benchmark_parser.add_argument("--min-run-time", type=float, default=0.5)
    benchmark_parser.set_defaults(func=command_benchmark)

    profile_parser = subparsers.add_parser("profile", help="Profile the current implementation.")
    add_common_arguments(profile_parser)
    profile_parser.add_argument("--iterations", type=int, default=5)
    profile_parser.add_argument("--record-shapes", action="store_true")
    profile_parser.add_argument("--row-limit", type=int, default=20)
    profile_parser.add_argument("--sort-by", default="self_cpu_time_total")
    profile_parser.set_defaults(func=command_profile)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
