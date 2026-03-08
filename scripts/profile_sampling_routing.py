"""Profile and benchmark current sampling-routing workloads."""

from __future__ import annotations

import argparse
import gc
import threading
import time
from dataclasses import dataclass
from typing import Callable

import psutil
import torch
import torch.utils.benchmark as torch_benchmark
from torch.profiler import ProfilerActivity, profile, record_function

from spflow.modules import leaves
from spflow.modules.ops import SplitMode
from spflow.modules.rat import Factorize
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.test_helpers.builders import make_rat_spn
from tests.utils.leaves import make_normal_leaf

PROFILE_MATCHES = ("clone", "repeat", "copy_", "_to_copy", "gather", "mul", "add", "einsum")
CPU_MEMORY_SCALE_MB = float(1024**2)


@dataclass
class Workload:
    """Prepared workload with a deterministic sampling runner."""

    name: str
    run: Callable[[int | None], torch.Tensor]


@dataclass
class BenchmarkResult:
    """Timing and memory summary for the current implementation."""

    mean_ms: float
    median_ms: float
    iqr_ms: float
    cpu_peak_rss_mb: float
    cpu_end_rss_mb: float
    peak_cuda_allocated_mb: float | None
    peak_cuda_reserved_mb: float | None


def _default_device() -> torch.device:
    return torch.empty(0).device


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _synchronize_if_needed() -> None:
    if _cuda_available():
        torch.cuda.synchronize()


def _reset_cuda_peak_memory_stats() -> None:
    if _cuda_available():
        torch.cuda.reset_peak_memory_stats()


def _cuda_peak_memory_mb() -> tuple[float | None, float | None]:
    if not _cuda_available():
        return None, None
    scale = float(1024**2)
    return (
        torch.cuda.max_memory_allocated() / scale,
        torch.cuda.max_memory_reserved() / scale,
    )


class _RssSampler:
    """Sample process RSS from a background thread during memory measurements."""

    def __init__(self, interval_s: float = 0.001) -> None:
        self.interval_s = interval_s
        self.process = psutil.Process()
        self.max_rss = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.max_rss = max(self.max_rss, self.process.memory_info().rss)
            time.sleep(self.interval_s)

    def __enter__(self) -> _RssSampler:
        self.max_rss = self.process.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.max_rss = max(self.max_rss, self.process.memory_info().rss)


def _build_factorize_workload(args: argparse.Namespace, *, differentiable: bool) -> Workload:
    torch.manual_seed(args.seed)
    device = _default_device()
    leaf = make_normal_leaf(
        out_features=args.num_features,
        out_channels=args.num_channels,
        num_repetitions=args.num_repetitions,
    )
    module = Factorize(inputs=[leaf], depth=args.depth, num_repetitions=args.num_repetitions)
    data_template = torch.full((args.batch_size, args.num_features), torch.nan, device=device)
    channel_ids = torch.randint(
        low=0,
        high=module.out_shape.channels,
        size=(args.batch_size, module.out_shape.features),
        device=device,
    )
    repetition_ids = torch.randint(
        low=0,
        high=args.num_repetitions,
        size=(args.batch_size,),
        device=device,
    )
    mask = torch.ones((args.batch_size, module.out_shape.features), dtype=torch.bool, device=device)

    def _make_ctx() -> SamplingContext:
        if differentiable:
            return SamplingContext(
                channel_index=to_one_hot(
                    channel_ids,
                    dim=-1,
                    dim_size=module.out_shape.channels,
                    dtype=torch.get_default_dtype(),
                ),
                mask=mask.clone(),
                repetition_index=to_one_hot(
                    repetition_ids,
                    dim=-1,
                    dim_size=args.num_repetitions,
                    dtype=torch.get_default_dtype(),
                ),
                is_differentiable=True,
            )
        return SamplingContext(
            channel_index=channel_ids.clone(),
            mask=mask.clone(),
            repetition_index=repetition_ids.clone(),
        )

    def _run(seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            return module._sample(data=data_template.clone(), sampling_ctx=_make_ctx(), cache=Cache())

    return Workload(name="factorize-diff" if differentiable else "factorize-int", run=_run)


def _build_rat_workload(args: argparse.Namespace) -> Workload:
    torch.manual_seed(args.seed)
    device = _default_device()
    split_mode = {
        "consecutive": SplitMode.consecutive(),
        "interleaved": SplitMode.interleaved(),
    }[args.split_mode]
    module = make_rat_spn(
        leaf_cls=leaves.Normal,
        depth=args.depth,
        n_region_nodes=args.region_nodes,
        num_leaves=args.num_leaves,
        num_repetitions=args.num_repetitions,
        n_root_nodes=args.root_nodes,
        num_features=args.num_features,
        outer_product=args.outer_product,
        split_mode=split_mode,
    )
    data_template = torch.full((args.batch_size, args.num_features), torch.nan, device=device)

    def _run(seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            return module.sample(data=data_template.clone())

    return Workload(name="rat-int", run=_run)


def build_workload(args: argparse.Namespace) -> Workload:
    if args.workload == "factorize-int":
        return _build_factorize_workload(args, differentiable=False)
    if args.workload == "factorize-diff":
        return _build_factorize_workload(args, differentiable=True)
    if args.workload == "rat-int":
        return _build_rat_workload(args)
    raise ValueError(f"Unsupported workload '{args.workload}'.")


def _warmup(workload: Workload, *, warmups: int) -> None:
    for _ in range(warmups):
        workload.run(None)
    _synchronize_if_needed()


def _measure_memory(
    workload: Workload, *, warmups: int, iterations: int
) -> tuple[float, float, float | None, float | None]:
    _warmup(workload, warmups=warmups)
    gc.collect()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    _reset_cuda_peak_memory_stats()
    with _RssSampler() as sampler:
        for _ in range(iterations):
            workload.run(None)
    _synchronize_if_needed()
    rss_after = process.memory_info().rss
    peak_rss_mb = max(0.0, (sampler.max_rss - rss_before) / CPU_MEMORY_SCALE_MB)
    end_rss_mb = max(0.0, (rss_after - rss_before) / CPU_MEMORY_SCALE_MB)
    peak_allocated_mb, peak_reserved_mb = _cuda_peak_memory_mb()
    return peak_rss_mb, end_rss_mb, peak_allocated_mb, peak_reserved_mb


def _benchmark_workload(
    workload: Workload,
    *,
    warmups: int,
    min_run_time: float,
    num_threads: int,
    memory_iterations: int,
) -> BenchmarkResult:
    _warmup(workload, warmups=warmups)
    timer = torch_benchmark.Timer(
        stmt="run()",
        globals={"run": workload.run},
        num_threads=num_threads,
        label=workload.name,
    )
    measurement = timer.blocked_autorange(min_run_time=min_run_time)

    cpu_peak_rss_mb, cpu_end_rss_mb, peak_allocated_mb, peak_reserved_mb = _measure_memory(
        workload,
        warmups=1,
        iterations=memory_iterations,
    )
    return BenchmarkResult(
        mean_ms=measurement.mean * 1_000.0,
        median_ms=measurement.median * 1_000.0,
        iqr_ms=measurement.iqr * 1_000.0,
        cpu_peak_rss_mb=cpu_peak_rss_mb,
        cpu_end_rss_mb=cpu_end_rss_mb,
        peak_cuda_allocated_mb=peak_allocated_mb,
        peak_cuda_reserved_mb=peak_reserved_mb,
    )


def command_benchmark(args: argparse.Namespace) -> int:
    workload = build_workload(args)
    result = _benchmark_workload(
        workload,
        warmups=args.warmups,
        min_run_time=args.min_run_time,
        num_threads=args.num_threads,
        memory_iterations=args.memory_iterations,
    )

    print(f"workload: {workload.name}")
    print(f"device: {_default_device()}")
    print(f"threads: {args.num_threads}")
    if _cuda_available():
        print(
            "impl       mean_ms   median_ms   iqr_ms   cpu_peak_rss_mb   cpu_end_rss_mb   "
            "peak_alloc_mb   peak_resv_mb"
        )
        print(
            f"{'current':<10}"
            f"{result.mean_ms:8.3f}"
            f"{result.median_ms:12.3f}"
            f"{result.iqr_ms:10.3f}"
            f"{result.cpu_peak_rss_mb:18.3f}"
            f"{result.cpu_end_rss_mb:17.3f}"
            f"{(result.peak_cuda_allocated_mb or 0.0):16.3f}"
            f"{(result.peak_cuda_reserved_mb or 0.0):15.3f}"
        )
    else:
        print("impl       mean_ms   median_ms   iqr_ms   cpu_peak_rss_mb   cpu_end_rss_mb")
        print(
            f"{'current':<10}"
            f"{result.mean_ms:8.3f}"
            f"{result.median_ms:12.3f}"
            f"{result.iqr_ms:10.3f}"
            f"{result.cpu_peak_rss_mb:18.3f}"
            f"{result.cpu_end_rss_mb:17.3f}"
        )
    return 0


def command_profile(args: argparse.Namespace) -> int:
    workload = build_workload(args)
    activities = [ProfilerActivity.CPU]
    if _cuda_available():
        activities.append(ProfilerActivity.CUDA)

    _warmup(workload, warmups=args.warmups)
    with profile(
        activities=activities,
        profile_memory=True,
        record_shapes=bool(args.record_shapes),
    ) as prof:
        for idx in range(args.iterations):
            with record_function(f"sampling_routing:{workload.name}:current:{idx}"):
                workload.run(None)
    _synchronize_if_needed()

    print(f"workload: {workload.name}")
    print(f"device: {_default_device()}")
    print("impl: current")
    print(
        prof.key_averages().table(
            sort_by=args.sort_by,
            row_limit=args.row_limit,
        )
    )
    matched = sorted(
        {event.key for event in prof.key_averages() if any(token in event.key for token in PROFILE_MATCHES)}
    )
    print("matched_ops:", ", ".join(matched) if matched else "<none>")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--workload",
            choices=("factorize-int", "factorize-diff", "rat-int"),
            default="factorize-int",
        )
        subparser.add_argument("--seed", type=int, default=1337)
        subparser.add_argument("--batch-size", type=int, default=256)
        subparser.add_argument("--num-features", type=int, default=256)
        subparser.add_argument("--num-channels", type=int, default=8)
        subparser.add_argument("--num-repetitions", type=int, default=16)
        subparser.add_argument("--depth", type=int, default=6)
        subparser.add_argument("--region-nodes", type=int, default=5)
        subparser.add_argument("--num-leaves", type=int, default=6)
        subparser.add_argument("--root-nodes", type=int, default=4)
        subparser.add_argument(
            "--split-mode",
            choices=("consecutive", "interleaved"),
            default="consecutive",
        )
        subparser.add_argument("--outer-product", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark", help="Benchmark the current sampling-routing implementation."
    )
    add_common_arguments(benchmark)
    benchmark.add_argument("--warmups", type=int, default=2)
    benchmark.add_argument("--min-run-time", type=float, default=0.5)
    benchmark.add_argument("--memory-iterations", type=int, default=3)
    benchmark.add_argument("--num-threads", type=int, default=1)
    benchmark.set_defaults(func=command_benchmark)

    profile_parser = subparsers.add_parser("profile", help="Profile one implementation with torch.profiler.")
    add_common_arguments(profile_parser)
    profile_parser.add_argument("--warmups", type=int, default=2)
    profile_parser.add_argument("--iterations", type=int, default=1)
    profile_parser.add_argument("--row-limit", type=int, default=15)
    profile_parser.add_argument(
        "--sort-by",
        choices=(
            "self_cpu_memory_usage",
            "cpu_memory_usage",
            "self_cpu_time_total",
            "cpu_time_total",
            "cuda_time_total",
        ),
        default="self_cpu_memory_usage",
    )
    profile_parser.add_argument("--record-shapes", action="store_true")
    profile_parser.set_defaults(func=command_profile)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
