"""Benchmark and profile classification gradient-descent workloads."""

from __future__ import annotations

import argparse
import gc
import resource
import sys
from dataclasses import dataclass

import torch
import torch.utils.benchmark as torch_benchmark
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, TensorDataset

from spflow.learn.gradient_descent import (
    TrainingMetrics,
    _process_training_batch,
    _run_validation_epoch,
    classification_loss,
)
from spflow.modules.leaves import Normal
from spflow.modules.ops import SplitMode
from tests.test_helpers.builders import make_einet, make_rat_spn

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


@dataclass(frozen=True)
class WorkloadConfig:
    name: str
    family: str
    batch_size: int
    train_batches: int
    val_batches: int
    num_features: int
    num_classes: int
    depth: int
    num_repetitions: int
    num_leaves: int
    num_sums: int | None = None
    region_nodes: int | None = None
    outer_product: bool = False


WORKLOADS: dict[str, WorkloadConfig] = {
    "small-einet": WorkloadConfig(
        name="small-einet",
        family="einet",
        batch_size=64,
        train_batches=6,
        val_batches=3,
        num_features=16,
        num_classes=4,
        depth=3,
        num_repetitions=2,
        num_leaves=4,
        num_sums=8,
    ),
    "target-einet": WorkloadConfig(
        name="target-einet",
        family="einet",
        batch_size=256,
        train_batches=8,
        val_batches=4,
        num_features=64,
        num_classes=8,
        depth=4,
        num_repetitions=4,
        num_leaves=8,
        num_sums=16,
    ),
    "large-rat": WorkloadConfig(
        name="large-rat",
        family="rat",
        batch_size=256,
        train_batches=8,
        val_batches=4,
        num_features=128,
        num_classes=8,
        depth=4,
        num_repetitions=4,
        num_leaves=8,
        region_nodes=8,
        outer_product=False,
    ),
}


@dataclass(frozen=True)
class BenchmarkSummary:
    mean_ms: float
    median_ms: float
    iqr_ms: float
    rss_delta_mb: float | None
    peak_cuda_allocated_mb: float | None
    peak_cuda_reserved_mb: float | None


class PreparedWorkload:
    """Prepared model, data, and runners for one benchmark workload."""

    def __init__(self, config: WorkloadConfig, *, device: torch.device, seed: int, lr: float) -> None:
        self.config = config
        self.device = device
        self.seed = seed
        self.lr = lr
        self.model = self._build_model().to(device)
        self.initial_state = self._clone_state_dict(self.model.state_dict())
        self.train_loader = self._build_loader(num_batches=config.train_batches, seed=seed + 1)
        self.val_loader = self._build_loader(num_batches=config.val_batches, seed=seed + 2)
        self.train_batch = next(iter(self.train_loader))

    def _build_model(self):
        if self.config.family == "einet":
            return make_einet(
                num_features=self.config.num_features,
                num_classes=self.config.num_classes,
                num_sums=self.config.num_sums or 8,
                num_leaves=self.config.num_leaves,
                depth=self.config.depth,
                num_repetitions=self.config.num_repetitions,
            )
        if self.config.family == "rat":
            return make_rat_spn(
                leaf_cls=Normal,
                depth=self.config.depth,
                n_region_nodes=self.config.region_nodes or 4,
                num_leaves=self.config.num_leaves,
                num_repetitions=self.config.num_repetitions,
                n_root_nodes=self.config.num_classes,
                num_features=self.config.num_features,
                outer_product=self.config.outer_product,
                split_mode=SplitMode.consecutive(),
            )
        raise ValueError(f"Unsupported workload family '{self.config.family}'.")

    def _build_loader(self, *, num_batches: int, seed: int) -> DataLoader:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        num_samples = num_batches * self.config.batch_size
        features = torch.randn((num_samples, self.config.num_features), generator=generator).to(self.device)
        labels = torch.randint(
            low=0,
            high=self.config.num_classes,
            size=(num_samples,),
            generator=generator,
        ).to(self.device)
        return DataLoader(
            TensorDataset(features, labels),
            batch_size=self.config.batch_size,
            shuffle=False,
        )

    @staticmethod
    def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in state_dict.items()}

    def reset_model(self) -> None:
        self.model.load_state_dict(self.initial_state)
        self.model.zero_grad(set_to_none=True)

    def run_train_batch(self) -> torch.Tensor:
        self.reset_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        metrics = TrainingMetrics()
        return _process_training_batch(
            self.model,
            self.train_batch,
            optimizer,
            classification_loss,
            metrics,
            True,
            None,
            1.0,
        )

    def run_validation_epoch(self) -> torch.Tensor:
        self.reset_model()
        metrics = TrainingMetrics()
        return _run_validation_epoch(
            self.model,
            self.val_loader,
            classification_loss,
            metrics,
            True,
            None,
            1.0,
        )


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


def _measure_memory(
    run: callable, *, device: torch.device
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


def _summarize_timer(run: callable, *, num_threads: int, min_run_time: float) -> tuple[float, float, float]:
    measurement = torch_benchmark.Timer(
        stmt="run()",
        globals={"run": run},
        num_threads=num_threads,
    ).blocked_autorange(min_run_time=min_run_time)
    return measurement.mean * 1_000.0, measurement.median * 1_000.0, measurement.iqr * 1_000.0


def _make_phase_runner(prepared: PreparedWorkload, *, phase: str):
    if phase == "train-batch":
        return prepared.run_train_batch
    if phase == "validation-epoch":
        return prepared.run_validation_epoch
    raise ValueError(f"Unsupported phase '{phase}'.")


def command_benchmark(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    config = WORKLOADS[args.workload]
    prepared = PreparedWorkload(config, device=device, seed=args.seed, lr=args.lr)
    phase_runner = _make_phase_runner(prepared, phase=args.phase)

    def timed_run() -> torch.Tensor:
        _synchronize(device)
        result = phase_runner()
        _synchronize(device)
        return result

    mean_ms, median_ms, iqr_ms = _summarize_timer(
        timed_run,
        num_threads=args.num_threads,
        min_run_time=args.min_run_time,
    )
    rss_delta_mb, peak_allocated_mb, peak_reserved_mb = _measure_memory(timed_run, device=device)
    summary = BenchmarkSummary(
        mean_ms=mean_ms,
        median_ms=median_ms,
        iqr_ms=iqr_ms,
        rss_delta_mb=rss_delta_mb,
        peak_cuda_allocated_mb=peak_allocated_mb,
        peak_cuda_reserved_mb=peak_reserved_mb,
    )

    print(
        f"workload: {config.name} family={config.family} phase={args.phase} "
        f"device={device.type} threads={args.num_threads}"
    )
    print(
        f"config: batch={config.batch_size} train_batches={config.train_batches} "
        f"val_batches={config.val_batches} features={config.num_features} "
        f"classes={config.num_classes} depth={config.depth} repetitions={config.num_repetitions}"
    )
    print("impl       mean_ms   median_ms   iqr_ms   rss_delta_mb   peak_alloc_mb   peak_resv_mb")
    rss_text = "n/a".rjust(14) if summary.rss_delta_mb is None else f"{summary.rss_delta_mb:14.3f}"
    alloc_text = (
        "n/a".rjust(16)
        if summary.peak_cuda_allocated_mb is None
        else f"{summary.peak_cuda_allocated_mb:16.3f}"
    )
    resv_text = (
        "n/a".rjust(15) if summary.peak_cuda_reserved_mb is None else f"{summary.peak_cuda_reserved_mb:15.3f}"
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

    config = WORKLOADS[args.workload]
    prepared = PreparedWorkload(config, device=device, seed=args.seed, lr=args.lr)
    phase_runner = _make_phase_runner(prepared, phase=args.phase)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        profile_memory=True,
        record_shapes=bool(args.record_shapes),
    ) as prof:
        for _ in range(args.iterations):
            phase_runner()
            _synchronize(device)

    print(
        f"profile: workload={config.name} family={config.family} phase={args.phase} " f"device={device.type}"
    )
    print(prof.key_averages().table(sort_by=args.sort_by, row_limit=args.row_limit))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--workload", choices=tuple(WORKLOADS), default="target-einet")
        subparser.add_argument("--phase", choices=("train-batch", "validation-epoch"), default="train-batch")
        subparser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        subparser.add_argument("--seed", type=int, default=1337)
        subparser.add_argument("--lr", type=float, default=1e-2)

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark the current implementation.")
    add_common_arguments(benchmark_parser)
    benchmark_parser.add_argument("--num-threads", type=int, default=1)
    benchmark_parser.add_argument("--min-run-time", type=float, default=0.5)
    benchmark_parser.set_defaults(func=command_benchmark)

    profile_parser = subparsers.add_parser("profile", help="Profile the current implementation.")
    add_common_arguments(profile_parser)
    profile_parser.add_argument("--iterations", type=int, default=3)
    profile_parser.add_argument("--row-limit", type=int, default=30)
    profile_parser.add_argument("--record-shapes", action="store_true")
    profile_parser.add_argument("--sort-by", default="self_cpu_time_total")
    profile_parser.set_defaults(func=command_profile)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
