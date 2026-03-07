# Benchmarking Methodology

This document describes the standard workflow for benchmarking performance improvements in SPFlow when an implementation change affects runtime or memory behavior.

## Non-Negotiable Rule

Performance improvements must not change output values.

Any benchmark in this repository must treat output parity between the old and new implementation as a hard requirement:

- no benchmark result is valid unless outputs match
- no speedup is acceptable if it introduces a behavioral regression
- correctness checks must run before performance numbers are reported

In short: benchmarking is only used to validate a faster implementation of the same behavior, not a different behavior.

## Principles

- Benchmark correctness first, then speed.
- Compare the old and new implementations in the same codebase.
- Keep the old path temporary and selectable via an environment variable.
- Measure both runtime and memory.
- Run benchmarks on:
  - local CPU
  - remote GPU via `rr`
- Report both parity and performance results together.

## Temporary Implementation Toggle

When benchmarking an optimization, keep both implementations in the tree temporarily:

- `legacy`: current or pre-optimization behavior
- `optimized`: new behavior under evaluation

Select the path with an environment variable read at runtime from `os.environ`.

Current example:

```bash
SPFLOW_SAMPLING_ROUTING_IMPL=legacy
SPFLOW_SAMPLING_ROUTING_IMPL=optimized
```

Rules:

- Default to `legacy` while benchmarking unless there is an explicit reason to switch the default.
- Reject invalid env values with a hard error.
- Keep both code paths behaviorally equivalent.
- Remove the legacy path after the optimization is accepted and no longer needs side-by-side comparison.

## Correctness Requirements

Every benchmark must prove the optimized path preserves outputs.

This is mandatory, not best-effort. If outputs diverge, the benchmark is a failure even if runtime or memory improves.

Required checks:

- Add regression tests that run both implementations with the same seed and identical inputs.
- Compare outputs before trusting any performance number.
- For non-differentiable sampling or deterministic integer outputs, require exact equality.
- For differentiable floating outputs, use `torch.testing.assert_close(...)` with a tight tolerance.

The benchmark harness itself must also compare outputs before printing timings.

## Benchmark Harness

Use a dedicated script in `scripts/` that:

- accepts a workload selector
- runs both `legacy` and `optimized`
- checks output parity first
- performs warmup iterations
- measures timed iterations
- reports memory
- prints a compact summary

For the current sampling-routing benchmark:

- script: `scripts/profile_sampling_routing.py`
- workloads:
  - `factorize-int`
  - `factorize-diff`
  - `rat-int`

## CPU Benchmark Procedure

Run benchmarks locally first.

Example commands:

```bash
.venv/bin/python scripts/profile_sampling_routing.py benchmark \
  --workload factorize-int \
  --batch-size 256 \
  --num-features 256 \
  --num-channels 8 \
  --num-repetitions 16 \
  --depth 6 \
  --warmups 2 \
  --iterations 10

.venv/bin/python scripts/profile_sampling_routing.py benchmark \
  --workload factorize-diff \
  --batch-size 256 \
  --num-features 256 \
  --num-channels 8 \
  --num-repetitions 16 \
  --depth 6 \
  --warmups 2 \
  --iterations 10

.venv/bin/python scripts/profile_sampling_routing.py benchmark \
  --workload rat-int \
  --batch-size 2048 \
  --num-features 128 \
  --depth 3 \
  --region-nodes 5 \
  --num-repetitions 7 \
  --num-leaves 6 \
  --root-nodes 4 \
  --warmups 2 \
  --iterations 5
```

For CPU profiling, use the script's `profile` mode with `torch.profiler`.

## GPU Benchmark Procedure with rr

Use `rr` to run the same benchmark remotely on a GPU machine.

Checklist:

- verify the host is reachable with `rr status`
- verify CUDA is available remotely
- force the benchmark process onto CUDA before running workloads
- use the same workloads as the CPU run where practical

CUDA verification example:

```bash
rr run --host dgxe ".venv/bin/python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'"
```

GPU benchmark wrapper example:

```bash
rr run --host dgxe ".venv/bin/python - <<'PY'
import runpy
import sys
import torch

torch.set_default_device('cuda')

runs = [
    [
        'scripts/profile_sampling_routing.py',
        'benchmark',
        '--workload', 'factorize-int',
        '--batch-size', '256',
        '--num-features', '256',
        '--num-channels', '8',
        '--num-repetitions', '16',
        '--depth', '6',
        '--warmups', '2',
        '--iterations', '10',
    ],
    [
        'scripts/profile_sampling_routing.py',
        'benchmark',
        '--workload', 'factorize-diff',
        '--batch-size', '256',
        '--num-features', '256',
        '--num-channels', '8',
        '--num-repetitions', '16',
        '--depth', '6',
        '--warmups', '2',
        '--iterations', '10',
    ],
    [
        'scripts/profile_sampling_routing.py',
        'benchmark',
        '--workload', 'rat-int',
        '--batch-size', '2048',
        '--num-features', '128',
        '--depth', '3',
        '--region-nodes', '5',
        '--num-repetitions', '7',
        '--num-leaves', '6',
        '--root-nodes', '4',
        '--warmups', '2',
        '--iterations', '5',
    ],
]

for argv in runs:
    sys.argv = argv
    try:
        runpy.run_path('scripts/profile_sampling_routing.py', run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
PY"
```

## Timing Rules

CPU timing:

- `time.perf_counter()` is sufficient.

CUDA timing:

- synchronize before starting each timed iteration
- synchronize after each timed iteration
- otherwise asynchronous kernel launch overhead invalidates the measurement

Current script behavior:

- calls `torch.cuda.synchronize()` before and after each timed iteration when CUDA is available

## Memory Rules

Always capture memory along with runtime.

CPU:

- use `torch.profiler(..., profile_memory=True)` when investigating operator-level allocation churn

CUDA:

- reset peak memory stats after warmups
- record:
  - `torch.cuda.max_memory_allocated()`
  - `torch.cuda.max_memory_reserved()`

Interpretation:

- `allocated` is the most useful measure of actual tensor footprint
- `reserved` is allocator pool size and may stay flat even when the optimization reduces real usage

## Validation Sequence

For every optimization benchmark:

1. Implement temporary `legacy` and `optimized` paths.
2. Add parity tests.
3. Run targeted local tests.
4. Run local CPU benchmarks.
5. Run remote GPU benchmarks with `rr`.
6. Confirm:
   - outputs still match
   - runtime improves or is neutral
   - memory improves or is neutral
7. Produce a final report.

## Final Report Format

Every benchmark summary should include:

- workload name
- hardware
- output parity result
- legacy runtime
- optimized runtime
- speedup
- legacy memory
- optimized memory
- whether the improvement is meaningful at the microbenchmark level, end-to-end level, or both

Suggested template:

```text
Environment
- CPU: <model>
- GPU: <model>

Parity
- Outputs: match / exact / close

CPU Results
- <workload>: <legacy ms> vs <optimized ms>, <speedup>x
- memory: <legacy> vs <optimized>

GPU Results
- <workload>: <legacy ms> vs <optimized ms>, <speedup>x
- peak allocated: <legacy> vs <optimized>
- peak reserved: <legacy> vs <optimized>

Conclusion
- Main win:
- End-to-end impact:
- Follow-up work:
```

## Notes for rr

- `rr` connectivity must be validated from the environment actually running the command.
- If the remote host has shell startup issues, fix the `rr` host shell configuration before trusting benchmark results.
- Record the exact remote host used in the final report.
