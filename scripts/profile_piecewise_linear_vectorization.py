"""Benchmark and profile the current PiecewiseLinear vectorized implementation."""

from __future__ import annotations

import argparse
import gc
import resource
import sys
from dataclasses import dataclass

import torch
import torch.utils.benchmark as torch_benchmark
from torch.profiler import ProfilerActivity, profile

from spflow.meta.data import Scope
from spflow.modules.leaves import PiecewiseLinear
from spflow.utils.domain import Domain

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


@dataclass(frozen=True)
class WorkloadConfig:
    batch_size: int
    num_features: int
    num_leaves: int
    num_repetitions: int
    num_points: int


WORKLOADS: dict[str, WorkloadConfig] = {
    "small": WorkloadConfig(batch_size=64, num_features=16, num_leaves=4, num_repetitions=4, num_points=17),
    "target": WorkloadConfig(batch_size=256, num_features=128, num_leaves=8, num_repetitions=8, num_points=33),
    "large": WorkloadConfig(batch_size=512, num_features=256, num_leaves=8, num_repetitions=8, num_points=65),
}


@dataclass(frozen=True)
class BenchmarkSummary:
    mean_ms: float
    median_ms: float
    iqr_ms: float
    rss_delta_mb: float | None
    peak_cuda_allocated_mb: float | None
    peak_cuda_reserved_mb: float | None


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


def _build_leaf(config: WorkloadConfig, *, device: torch.device) -> PiecewiseLinear:
    leaf = PiecewiseLinear(
        scope=Scope(list(range(config.num_features))),
        out_channels=config.num_leaves,
        num_repetitions=config.num_repetitions,
    )
    domains = [Domain.continuous_range(-4.0, 4.0) for _ in range(config.num_features)]
    xs = []
    ys = []
    for i_repetition in range(config.num_repetitions):
        xs_rep = []
        ys_rep = []
        for i_leaf in range(config.num_leaves):
            xs_leaf = []
            ys_leaf = []
            for i_feature in range(config.num_features):
                x = torch.linspace(-4.0, 4.0, config.num_points, device=device, dtype=torch.float32)
                center = -1.0 + 2.0 * ((i_feature + i_leaf + i_repetition) % 7) / 6.0
                scale = 0.45 + 0.05 * ((i_feature + 2 * i_leaf + i_repetition) % 5)
                y = torch.exp(-0.5 * ((x - center) / scale) ** 2)
                y[0] = 0.0
                y[-1] = 0.0
                y = y / torch.trapezoid(y=y, x=x).clamp_min(1e-10)
                xs_leaf.append([x])
                ys_leaf.append([y])
            xs_rep.append(xs_leaf)
            ys_rep.append(ys_leaf)
        xs.append(xs_rep)
        ys.append(ys_rep)

    leaf.xs = xs
    leaf.ys = ys
    leaf.domains = domains
    leaf.is_initialized = True
    return leaf


def _build_log_likelihood_input(config: WorkloadConfig, *, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    data = torch.empty((config.batch_size, config.num_features), device=device, dtype=torch.float32).uniform_(
        -3.5, 3.5, generator=generator
    )
    nan_mask = torch.rand((config.batch_size, config.num_features), device=device, generator=generator) < 0.1
    data[nan_mask] = float("nan")
    return data


def _build_sample_input(config: WorkloadConfig, *, device: torch.device) -> torch.Tensor:
    return torch.full((config.batch_size, config.num_features), float("nan"), device=device, dtype=torch.float32)


def _make_runner(args: argparse.Namespace) -> tuple[callable, WorkloadConfig]:
    config = WORKLOADS[args.workload]
    device = torch.device(args.device)
    leaf = _build_leaf(config, device=device)
    if args.operation == "log_likelihood":
        data = _build_log_likelihood_input(config, device=device, seed=args.seed)

        def run() -> torch.Tensor:
            return leaf.log_likelihood(data)

        return run, config

    sample_data = _build_sample_input(config, device=device)

    def run() -> torch.Tensor:
        torch.manual_seed(args.seed)
        return leaf.sample(num_samples=config.batch_size, data=sample_data.clone())

    return run, config


def _measure_memory(run: callable, *, device: torch.device) -> tuple[float | None, float | None, float | None]:
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


def _summarize_timer(run: callable, *, num_threads: int, min_run_time: float) -> tuple[float, float, float]:
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

    run, config = _make_runner(args)
    mean_ms, median_ms, iqr_ms = _summarize_timer(
        run,
        num_threads=args.num_threads,
        min_run_time=args.min_run_time,
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
        f"workload: {args.workload} operation={args.operation} device={device.type} "
        f"threads={args.num_threads}"
    )
    print(
        f"config: batch={config.batch_size} features={config.num_features} "
        f"leaves={config.num_leaves} repetitions={config.num_repetitions} points={config.num_points}"
    )
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
    run, config = _make_runner(args)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, profile_memory=True, record_shapes=bool(args.record_shapes)) as prof:
        for _ in range(args.iterations):
            run()

    print(
        f"profile: workload={args.workload} operation={args.operation} "
        f"device={device.type} batch={config.batch_size} features={config.num_features} "
        f"leaves={config.num_leaves} repetitions={config.num_repetitions} points={config.num_points}"
    )
    print(prof.key_averages().table(sort_by=args.sort_by, row_limit=args.row_limit))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--operation", choices=("log_likelihood", "sample"), required=True)
        subparser.add_argument("--workload", choices=tuple(WORKLOADS), default="target")
        subparser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        subparser.add_argument("--seed", type=int, default=1337)

    benchmark = subparsers.add_parser("benchmark", help="Benchmark current PiecewiseLinear performance.")
    add_common_arguments(benchmark)
    benchmark.add_argument("--num-threads", type=int, default=1)
    benchmark.add_argument("--min-run-time", type=float, default=0.5)
    benchmark.set_defaults(func=command_benchmark)

    profile_parser = subparsers.add_parser("profile", help="Profile current PiecewiseLinear performance.")
    add_common_arguments(profile_parser)
    profile_parser.add_argument("--iterations", type=int, default=3)
    profile_parser.add_argument("--row-limit", type=int, default=20)
    profile_parser.add_argument("--record-shapes", action="store_true")
    profile_parser.add_argument("--sort-by", default="self_cpu_time_total")
    profile_parser.set_defaults(func=command_profile)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
