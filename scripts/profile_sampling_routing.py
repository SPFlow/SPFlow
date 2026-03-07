"""Profile and benchmark current sampling-routing workloads."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile

from spflow.modules import leaves
from spflow.modules.ops import SplitMode
from spflow.modules.rat import Factorize
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.test_helpers.builders import make_rat_spn
from tests.utils.leaves import make_normal_leaf

PROFILE_MATCHES = ("clone", "repeat", "copy_", "_to_copy", "gather", "mul", "add")


@dataclass
class Workload:
    """Prepared workload with a deterministic sampling runner."""

    name: str
    run: Callable[[int], torch.Tensor]


@dataclass
class BenchmarkResult:
    """Timing and memory summary for the current implementation."""

    mean_ms: float
    median_ms: float
    stdev_ms: float
    peak_cuda_allocated_mb: float | None
    peak_cuda_reserved_mb: float | None


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


def _build_factorize_workload(args: argparse.Namespace, *, differentiable: bool) -> Workload:
    torch.manual_seed(args.seed)
    leaf = make_normal_leaf(
        out_features=args.num_features,
        out_channels=args.num_channels,
        num_repetitions=args.num_repetitions,
    )
    module = Factorize(inputs=[leaf], depth=args.depth, num_repetitions=args.num_repetitions)
    data_template = torch.full((args.batch_size, args.num_features), torch.nan)
    channel_ids = torch.randint(
        low=0,
        high=module.out_shape.channels,
        size=(args.batch_size, module.out_shape.features),
    )
    repetition_ids = torch.randint(
        low=0,
        high=args.num_repetitions,
        size=(args.batch_size,),
    )
    mask = torch.ones((args.batch_size, module.out_shape.features), dtype=torch.bool)

    def _run(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        if differentiable:
            sampling_ctx = SamplingContext(
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
        else:
            sampling_ctx = SamplingContext(
                channel_index=channel_ids.clone(),
                mask=mask.clone(),
                repetition_index=repetition_ids.clone(),
            )
        with torch.no_grad():
            return module._sample(
                data=data_template.clone(),
                sampling_ctx=sampling_ctx,
                cache=Cache(),
            )

    return Workload(name="factorize-diff" if differentiable else "factorize-int", run=_run)


def _build_rat_workload(args: argparse.Namespace) -> Workload:
    torch.manual_seed(args.seed)
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
    data_template = torch.full((args.batch_size, args.num_features), torch.nan)

    def _run(seed: int) -> torch.Tensor:
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


def _time_workload(workload: Workload, *, warmups: int, iterations: int, seed: int) -> list[float]:
    for idx in range(warmups):
        workload.run(seed + idx)
    _synchronize_if_needed()
    _reset_cuda_peak_memory_stats()
    timings_ms: list[float] = []
    for idx in range(iterations):
        _synchronize_if_needed()
        start = time.perf_counter()
        workload.run(seed + 1_000 + idx)
        _synchronize_if_needed()
        timings_ms.append((time.perf_counter() - start) * 1_000.0)
    return timings_ms


def _summarize_timings(timings_ms: list[float]) -> tuple[float, float, float]:
    mean_ms = statistics.mean(timings_ms)
    median_ms = statistics.median(timings_ms)
    stdev_ms = statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0
    return mean_ms, median_ms, stdev_ms


def command_benchmark(args: argparse.Namespace) -> int:
    workload = build_workload(args)

    timings = _time_workload(
        workload,
        warmups=args.warmups,
        iterations=args.iterations,
        seed=args.seed,
    )
    mean_ms, median_ms, stdev_ms = _summarize_timings(timings)
    peak_allocated_mb, peak_reserved_mb = _cuda_peak_memory_mb()
    result = BenchmarkResult(
        mean_ms=mean_ms,
        median_ms=median_ms,
        stdev_ms=stdev_ms,
        peak_cuda_allocated_mb=peak_allocated_mb,
        peak_cuda_reserved_mb=peak_reserved_mb,
    )

    print(f"workload: {workload.name}")
    if _cuda_available():
        print("impl       mean_ms   median_ms   stdev_ms   peak_alloc_mb   peak_resv_mb")
        print(
            f"{'current':<10}"
            f"{result.mean_ms:8.3f}"
            f"{result.median_ms:12.3f}"
            f"{result.stdev_ms:11.3f}"
            f"{(result.peak_cuda_allocated_mb or 0.0):16.3f}"
            f"{(result.peak_cuda_reserved_mb or 0.0):15.3f}"
        )
    else:
        print("impl       mean_ms   median_ms   stdev_ms")
        print(f"{'current':<10}{result.mean_ms:8.3f}{result.median_ms:12.3f}{result.stdev_ms:11.3f}")
    return 0


def command_profile(args: argparse.Namespace) -> int:
    workload = build_workload(args)

    for idx in range(args.warmups):
        workload.run(args.seed + idx)

    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=bool(args.record_shapes),
    ) as prof:
        for idx in range(args.iterations):
            workload.run(args.seed + idx)

    print(f"workload: {workload.name}")
    print("impl: current")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=args.row_limit))
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
    benchmark.add_argument("--warmups", type=int, default=3)
    benchmark.add_argument("--iterations", type=int, default=10)
    benchmark.set_defaults(func=command_benchmark)

    profile_parser = subparsers.add_parser("profile", help="Profile one implementation with torch.profiler.")
    add_common_arguments(profile_parser)
    profile_parser.add_argument("--warmups", type=int, default=2)
    profile_parser.add_argument("--iterations", type=int, default=1)
    profile_parser.add_argument("--row-limit", type=int, default=15)
    profile_parser.add_argument("--record-shapes", action="store_true")
    profile_parser.set_defaults(func=command_profile)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
