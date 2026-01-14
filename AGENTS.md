# AI Agent Guide for SPFlow

## Development Commands
- **Run all tests:** `.venv/bin/pytest -n 4`
- **Run single test:** `.venv/bin/pytest tests/path/to/test_file.py::test_function`
- **Format code:** `.venv/bin/black spflow tests`
- **Build docs:** `cd docs && make html`

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
