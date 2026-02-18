# AI Agent Guide for SPFlow

## Development Commands
- **Run all tests:** `.venv/bin/pytest -n 4`
- **Run single test:** `.venv/bin/pytest tests/path/to/test_file.py::test_function`
- **Format code:** `.venv/bin/black spflow tests`
- **Build docs:** `cd docs && make html`
- **Generate HTML coverage report:** `.venv/bin/pytest -n 4 --cov=spflow --cov-report=html`
- **List lowest-coverage files:** `.venv/bin/python scripts/coverage_inspect.py list --limit 30`
- **Show missed line chunks (with context):** `.venv/bin/python scripts/coverage_inspect.py show spflow/path/to_file.py --context 3`
- **Coverage runtime note (PyTorch):** Avoid module-targeted `pytest-cov` like `--cov=spflow.learn.prometheus` in this environment; it can trigger `RuntimeError: function '_has_torch_function' already has a docstring` during `import torch`. Use package-level coverage targets instead (for example `--cov=spflow` or `--cov=spflow.learn`).

## Code Style Guidelines
- **Python version:** 3.10+ with type hints required
- **Docstrings:** Google style convention
- **Error handling:** Use custom exceptions from `spflow.exceptions`
- **PyTorch:** All Modules inherit from `nn.Module`, use proper tensor typing
- **Testing:** Use pytest, fixtures in `conftest.py`

## Project Structure
- Core modules in `spflow/modules/` with base classes in `base.py` files
- Leaf distributions in `spflow/modules/leaves/`
- Tests mirror source structure in `tests/`


## Versioning & Commits

* **Versioning:** Semantic Versioning. The version is in `spflow/__init__.py`.
* **Commits:** Use [Conventional Commits](https://www.conventionalcommits.org/). Keep the commit body brief. Don't mention which files changed in detail since we can see this in the git diff anyway.
* **NOTE:** Never `git add` or `git commit` unless I ask you to.

### Commit Message Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Commit Types

| Type       | Description                 | Version Bump |
|------------|-----------------------------|--------------|
| `feat`     | New feature                 | MINOR        |
| `fix`      | Bug fix                     | PATCH        |
| `docs`     | Documentation only          | -            |
| `style`    | Formatting, no logic change | -            |
| `refactor` | Code refactoring            | -            |
| `perf`     | Performance improvements    | -            |
| `test`     | Adding/updating tests       | -            |
| `chore`    | Maintenance tasks           | -            |
| `ci`       | CI/CD changes               | -            |
| `build`    | Build system changes        | -            |

### Scopes (Optional)

`modules`, `learn`, `leaf`, `rat`, `deps`, `tests`, `sum`, `product`

### Examples

```bash
git commit -m "feat(leaf): add Gamma distribution module"
git commit -m "fix(dispatch): correct cache initialization"
git commit -m "docs: update RAT-SPN examples in README"
```

## Programming Practices
* Prefer clarity, simplicity, and explicitness (Zen of Python).
* Write code that is correct, readable, maintainable, and efficient.
* Keep functions small and focused on one task (single responsibility).
* Keep modules cohesive; avoid unnecessary coupling.
* Prefer simple control flow; avoid deeply nested logic.
* Avoid repetition; follow DRY (Don’t Repeat Yourself).
* Never implement silent fallbacks when adding new features; fail fast with explicit errors or warnings.

## Comment Best Practices
* Use comments to explain intent, invariants, assumptions, and non-obvious tradeoffs.
* Prefer self-explanatory names and clear code; do not restate what the code already says.
* Keep comments concise, specific, and close to the code they describe.
* Update or remove comments whenever code changes so comments never become stale.
* Use `TODO(username): short reason` for actionable follow-ups; avoid vague TODOs.

## Tensor/Module Debug Tracing
* Use trace helpers from `spflow.utils.debug` when debugging runtime tensor/module behavior without an interactive interpreter.
* Enable tracing explicitly in code with:
  * `configure_trace(enabled=True, prefix="SPFLOW", max_events=400, max_values=6)`
  * or environment variable `SPFLOW_TRACE=1` (legacy `APC_TRACE=1` also works).
* Core helpers:
  * `trace_tensor(name, tensor)` for shape/dtype/device/finite stats + value preview.
  * `trace_tensor_delta(name, before, after)` for before/after numerical drift summaries.
  * `trace_tensor_tree(name, payload)` for nested dict/list/tuple tensor payloads.
  * `trace_module_state(name, module)` for parameters, gradients, and buffers.
  * `attach_module_trace_hooks(module, name, recurse=...)` + `remove_trace_hooks(handles)` to trace forward inputs/outputs across module calls.
* Keep tracing lightweight:
  * Set bounded `max_events` and `max_values`.
  * Attach hooks in a `try/finally` block and always remove handles.
  * Prefer targeted traces around suspicious modules/tensors instead of whole-model tracing by default.
