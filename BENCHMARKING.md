# Benchmarking Methodology

This document describes the standard workflow for benchmarking performance improvements in SPFlow when an implementation change affects runtime or memory behavior.

## PyTorch References

Useful upstream references:

- profiling recipe source: https://raw.githubusercontent.com/pytorch/tutorials/refs/heads/main/recipes_source/recipes/profiler_recipe.py
- benchmarking recipe source: https://raw.githubusercontent.com/pytorch/tutorials/refs/heads/main/recipes_source/recipes/benchmark.py
- profiler tutorial page: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- benchmark tutorial page: https://pytorch.org/tutorials/recipes/recipes/benchmark.html

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
- Benchmark more than one regime:
  - a small/control workload
  - at least one realistic target workload
  - at least one large stress or crossover workload if the optimization may only help for certain shapes
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
- If the optimization changes reduction order or contraction scheduling, expect tiny floating-point differences even when the math is equivalent:
  - use a tolerance that is still strict enough to catch real regressions
  - report the observed max absolute difference in the benchmark output
- If gradients matter for the optimized path, compare gradients as well as forward outputs before trusting runtime data.

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
- separates cheap benchmark-mode measurements from expensive diagnostic profiling:
  - `benchmark` mode should stay lightweight enough that measurement overhead does not dominate runtime
  - `profile` mode can use `torch.profiler` and other heavy tools for operator-level diagnosis

Choose the measurement tool based on the question:

- use `torch.utils.benchmark.Timer` as the default benchmarking tool for SPFlow performance work, including end-to-end workload runners when they can be expressed as a callable or statement
- use `Timer.blocked_autorange()` by default; the official PyTorch docs recommend it because it warms up, adjusts block size to keep timer overhead small, and yields replicates suitable for summary statistics
- use `Timer.timeit()` only for quick checks when you intentionally want a fixed run count
- use `torch.utils.benchmark.Compare` when comparing several shapes, implementations, or thread counts
- use `Timer.collect_callgrind()` when wall-clock noise is obscuring a small but important CPU-side change and instruction counts would be more informative
- do not use Python `timeit` or raw `time.perf_counter()` as the source of benchmark numbers in final reports when `torch.utils.benchmark` can measure the workload
- never use Python `timeit` alone for CUDA kernels; without synchronization it can mostly measure launch overhead instead of kernel completion

For the current sampling-routing benchmark:

- script: `scripts/profile_sampling_routing.py`
- workloads:
  - `factorize-int`
  - `factorize-diff`
  - `rat-int`

## CPU Benchmark Procedure

Run benchmarks locally first.

Guidance:

- Start with a control workload to verify the optimization is not regressing small cases.
- Add at least one stress workload large enough to expose bandwidth or allocation bottlenecks.
- If the optimization is shape-sensitive, include a crossover matrix rather than a single benchmark point.
- Sweep representative tensor shapes instead of a single shape. The official PyTorch benchmark recipe shows that the faster implementation can flip as shape and thread count change.
- Report the thread count explicitly. `torch.utils.benchmark.Timer` defaults to one thread, which is good for stable apples-to-apples comparison but may not match end-to-end SPFlow usage.
- Where possible, benchmark a prepared callable with `torch.utils.benchmark.Timer(stmt="run()", globals={"run": run}, ...)` so setup stays outside the measured statement.

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

- Use `torch.utils.benchmark.Timer`, not raw Python timers, for benchmark numbers.
- Prefer `blocked_autorange()` for measured results; it is explicitly designed to balance timer overhead against statistical quality.
- Report the measurement summary from PyTorch benchmark output and include mean or median as appropriate for the comparison you are making.
- Keep setup outside the timed statement where possible. Benchmark the operation you care about, not tensor creation or Python bookkeeping unless that overhead is intentionally part of the question.
- Keep heavyweight profiler instrumentation out of the timed path.
- If timings are highly variable, suspect allocator churn, thread-pool effects, or a hidden synchronization artifact before drawing conclusions.

Threading:

- Benchmark with thread settings that match the intended deployment regime.
- Record the value of `torch.get_num_threads()` or the explicit `num_threads` passed to `torch.utils.benchmark.Timer`.
- Do not generalize from a single-thread microbenchmark to a highly threaded workload. The benchmark recipe demonstrates that algorithm rankings can change once threading changes.

CUDA timing:

- Use `torch.utils.benchmark.Timer` for benchmark numbers because it is runtime-aware and synchronizes asynchronous accelerator work when needed.
- Warm up before relying on results. CUDA library initialization, kernel selection, and allocator setup can skew early runs.
- Keep explicit manual synchronization only when building custom diagnostics outside `torch.utils.benchmark`; it should not be the default benchmark path.

## Profiling Rules

Use `torch.profiler` to explain a benchmark result, not to generate the primary timing number.

- keep profiler instrumentation out of the main benchmark timing path
- use `record_function(...)` labels around meaningful SPFlow regions when the profile would otherwise be hard to interpret
- enable `record_shapes=True` when you suspect shape-specific dispatch or unexpected broadcasting, but disable it for the default path because it adds overhead
- use `profile_memory=True` when investigating allocation churn, peak tensor footprint, or unexpected copies
- sort the output by the metric that matches the question:
  - `self_cpu_time_total` to find Python-side or operator self time
  - `cpu_time_total` to find expensive operator families including children
  - `self_cpu_memory_usage` to find operators that allocate directly
  - `cpu_memory_usage` to find operators with large transitive memory impact
  - `cuda_time_total` or `xpu_time_total` when profiling accelerator kernels
- use `group_by_input_shape=True` when the same operator appears under multiple shapes and you need to see crossover points or pathological shapes
- use `with_stack=True` only for targeted investigations; stack capture adds noticeable overhead
- use `record_function(...)` labels to put SPFlow-specific ranges into the profile so operator tables and traces map back to domain concepts instead of only low-level `aten::*` names

Trace export:

- when operator tables are not enough, export a Chrome trace and inspect it in `chrome://tracing` or Perfetto
- traces are especially useful on CUDA for spotting many small kernel launches, host/device gaps, or unintended synchronization
- keep traces short and focused; long unbounded traces become hard to inspect and expensive to capture

Long-running jobs:

- if the workload is a loop or training job, do not trace the entire run by default
- use `torch.profiler.schedule(...)` with `wait`, `warmup`, and `active` windows so startup noise is discarded and trace size stays bounded
- use `on_trace_ready` to print summary tables or export one trace per active window
- call `prof.step()` once per logical iteration so the profiler schedule advances correctly
- prefer scheduled profiling even for SPFlow sampling or likelihood loops if more than a few iterations are needed; it keeps traces inspectable and avoids profiling startup costs dominating the result

## Memory Rules

Always capture memory along with runtime.

CPU:

- in `benchmark` mode, prefer a lightweight process-level metric such as RSS delta if you only need a quick memory signal
- use `torch.profiler(..., profile_memory=True)` in a separate `profile` mode when investigating operator-level allocation churn
- do not put `torch.profiler` memory collection in the main benchmark timing path unless the benchmark is explicitly about profiler output; it can dominate runtime and distort conclusions
- interpret `self_cpu_memory_usage` as direct allocation pressure from an operator and `cpu_memory_usage` as the operator plus its callees
- if you suspect repeated temporary tensors, look for `aten::empty`, `aten::empty_strided`, `aten::clone`, `aten::copy_`, `aten::cat`, and large broadcasted ops in the profiler output

CUDA:

- reset peak memory stats after warmups
- record:
  - `torch.cuda.max_memory_allocated()`
  - `torch.cuda.max_memory_reserved()`
- use profiler traces when peak stats are too coarse to explain where allocation bursts occur

Interpretation:

- `allocated` is the most useful measure of actual tensor footprint
- `reserved` is allocator pool size and may stay flat even when the optimization reduces real usage
- if `reserved` stays flat while `allocated` drops, treat that as a real memory improvement rather than a failed optimization

## Validation Sequence

For every optimization benchmark:

1. Implement temporary `legacy` and `optimized` paths.
2. Add parity tests.
3. Run targeted local tests.
4. Run local CPU benchmarks.
5. Run remote GPU benchmarks with `rr`.
6. Confirm:
   - outputs still match
   - gradients still match when relevant
   - runtime improves or is neutral
   - memory improves or is neutral
   - the improvement holds on the workload regime you actually care about, not just on a single microbenchmark point
7. Produce a final report.
8. Remove the temporary `legacy` path after the optimization is accepted.

## Preferred Defaults

For this repository, the default stance is:

- benchmarking: `torch.utils.benchmark.Timer` with `blocked_autorange()`
- profiling: `torch.profiler.profile(...)`
- long-running profiling: `torch.profiler.schedule(...)` plus `on_trace_ready`
- comparison tables: `torch.utils.benchmark.Compare`
- Python timers: fallback-only for cases `torch.utils.benchmark` cannot reasonably express or measure

## Final Report Format

Every benchmark summary should include:

- workload name
- hardware
- output parity result
- observed max absolute output difference for floating outputs
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
- If `rr` connectivity is flaky, retry host discovery separately from the benchmark command so transport failures are not confused with workload failures.
- Record the exact remote host used in the final report.
