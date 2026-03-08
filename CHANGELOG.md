# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

## [1.1.0] - 2026-03-08

### Added
- Added the `spflow.zoo.NaiveBayes` model for density estimation and classification, including configurable class priors and classifier-compatible posterior training outputs.

### Changed
- Improved einsum-layer log-likelihood performance and reduced peak memory usage, with A100 benchmarks showing 3.38x to 4.71x speedups on the main workloads and peak allocated memory dropping from 2491.656 MB to 1435.656 MB or from 2339.656 MB to 803.656 MB depending on shape.
- Simplified `PiecewiseLinear` runtime paths to the packed vectorized implementation, with CPU log-likelihood benchmarks improving by 5.3x to 30.7x, GPU log-likelihood improving by 168.4x to 4431.5x, CPU sampling improving by 3.5x to 9.2x, and GPU sampling improving by 11.7x to 15.0x on the recorded workloads.
- Optimized histogram `log_prob` to lower allocations and improve throughput, with CPU benchmarks improving from 0.094 ms to 0.041 ms on the small case and from 7.201 ms to 1.302 ms on the large case, while A100 peak allocated memory dropped from 10.380 MB to 4.148 MB on the target workload and from 31.135 MB to 12.419 MB on the large workload.
- Reused cached classification forwards in gradient descent so `log_posterior` and `log_likelihood` share traversals during training and validation, improving CPU train and validation times by roughly 1.66x to 1.92x, GPU train times by up to 1.81x, and reducing GPU peak allocation from 59.23 MB to 41.52 MB on the target workload.
- Removed the legacy factorize routing path in RAT sampling to reduce allocation overhead and improve throughput, with CPU workloads improving by 1.04x to 1.18x, GPU workloads improving by 1.08x to 2.94x, and GPU peak allocated memory dropping from 4378.008 MB to 811.259 MB on the largest recorded case.

### Fixed
- Removed unnecessary `retain_graph` overhead in expectation-maximization while preserving the cached likelihood gradients needed by the M-step, reducing GPU peak allocation from 488.2 MB to 430.8 MB on the target full-batch workload and from 3040.8 MB to 2688.5 MB on the large full-batch workload while keeping runtime approximately neutral overall.

## [1.0.2] - 2026-03-07

### Fixed
- Fixed CUDA/default-device bugs across learning, inference, and zoo modules so the full pytest suite now passes on GPU hosts with a CUDA default device.
- Corrected device handling for one-hot conversion, learned leaf instantiation, leaf likelihood evaluation, and inferred discrete domains on non-CPU devices.
- Fixed device-compatible generator usage in APC, CNet, and SOS training/build paths.
- Fixed PIC/QPC materialization, tensorized execution, functional sharing, and weighted-sum paths to keep CUDA computations device-consistent.
- Updated sklearn wrapper defaults and related doctest/test expectations to follow the active torch device.

## [1.0.1] - 2026-03-07

### Fixed
- Corrected the published package contents so PyPI distributions only ship the runtime `spflow` package instead of also including repository-only files such as `tests/`, `docs/`, and `scripts`.
- Added the `py.typed` marker to the published wheel so the typed package metadata matches the actual distribution contents.

## [1.0.0] - 2026-03-04

This is a complete rewrite of SPFlow using a modular PyTorch-based design, providing significant performance improvements and GPU acceleration (read [README.md](README.md) and the documentation at [spflow.github.io](https://spflow.github.io) for more details).

### Legacy Support
- The previous non-PyTorch version of SPFlow remains available as `spflow==0.0.48` on PyPI and via the `legacy` branch in the repository.


[Unreleased]: https://github.com/SPFlow/SPFlow/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/SPFlow/SPFlow/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/SPFlow/SPFlow/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/SPFlow/SPFlow/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/SPFlow/SPFlow/releases/tag/v1.0.0
