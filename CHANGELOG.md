# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

## [1.0.1] - 2026-03-07

### Fixed
- Corrected the published package contents so PyPI distributions only ship the runtime `spflow` package instead of also including repository-only files such as `tests/`, `docs/`, and `scripts`.
- Added the `py.typed` marker to the published wheel so the typed package metadata matches the actual distribution contents.

## [1.0.0] - 2026-03-04

This is a complete rewrite of SPFlow using a modular PyTorch-based design, providing significant performance improvements and GPU acceleration (read [README.md](README.md) and the documentation at [spflow.github.io](https://spflow.github.io) for more details).

### Legacy Support
- The previous non-PyTorch version of SPFlow remains available as `spflow==0.0.48` on PyPI and via the `legacy` branch in the repository.


[Unreleased]: https://github.com/SPFlow/SPFlow/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/SPFlow/SPFlow/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/SPFlow/SPFlow/releases/tag/v1.0.0
