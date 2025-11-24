# Contribution Guidelines

Thank you for considering contributing to SPFlow! This document provides guidelines for contributing code, documentation, and reporting issues.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Versioning](#versioning)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone git@github.com:YOUR-USERNAME/SPFlow.git
   cd SPFlow
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream git@github.com:SPFlow/SPFlow.git
   ```

## Development Setup

### Using uv (Recommended)

```bash
# Install SPFlow with development dependencies
uv sync --extra dev

# Activate virtual environment
source .venv/bin/activate
```

### System Dependencies

For visualization features, install [Graphviz](https://graphviz.org/download/):

- **macOS**: `brew install graphviz`
- **Ubuntu/Debian**: `sudo apt-get install graphviz`
- **Windows**: Download from https://graphviz.org/download/

## Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feat/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. Make your changes following our [code standards](#code-standards)

3. Add tests for new functionality

4. Run the test suite to ensure nothing breaks

5. Commit your changes following our [commit conventions](#commit-conventions)

## Commit Conventions

SPFlow follows [Conventional Commits](https://www.conventionalcommits.org/) for clear commit history and automated versioning.

### Commit Message Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Commit Types

- **feat**: New feature (triggers MINOR version bump)
- **fix**: Bug fix (triggers PATCH version bump)
- **docs**: Documentation only changes
- **style**: Code style changes (formatting, no logic change)
- **refactor**: Code refactoring without feature/bug changes
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks (dependencies, tooling)
- **ci**: CI/CD configuration changes
- **build**: Build system changes

### Breaking Changes

Add `!` after the type or `BREAKING CHANGE:` in the footer for breaking changes (triggers MAJOR version bump):

```bash
git commit -m "feat!: remove deprecated Sum module

BREAKING CHANGE: Sum has been removed. Use ElementwiseSum instead."
```

### Commit Scopes

Optional scopes to specify what area changed:

- `modules` - Module implementations
- `learn` - Learning algorithms
- `leaf` - Leaf distributions
- `rat` - RAT-SPN specific
- `deps` - Dependencies

### Examples

```bash
# New feature
git commit -m "feat(leaf): add Gamma distribution module"

# Bug fix
git commit -m "fix(dispatch): correct cache initialization in DispatchContext"

# Documentation
git commit -m "docs: update RAT-SPN usage examples in README"

# Breaking change
git commit -m "feat(modules)!: change signature of log_likelihood

BREAKING CHANGE: log_likelihood now requires dispatch_ctx as keyword argument"

# Multiple changes
git commit -m "feat(modules): add support for conditional distributions

- Add CondGaussian module
- Add CondCategorical module
- Update AutoLeaf to recognize conditional distributions

Closes #123"
```

## Pull Request Process

1. **Update your branch** with latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/dev/torch
   ```

2. **Push your branch** to your fork:
   ```bash
   git push origin your-branch-name
   ```

3. **Open a Pull Request** on GitHub:
   - Target the `dev/torch` branch (or `master` for hotfixes)
   - Use a descriptive title following commit conventions
   - Reference related issues (e.g., "Fixes #123")
   - Fill out the PR template

4. **PR Requirements**:
   - [ ] All tests pass
   - [ ] Code is formatted with Black
   - [ ] Type hints added where applicable
   - [ ] Documentation updated (if needed)
   - [ ] Tests added for new functionality
   - [ ] CHANGELOG.md updated (for significant changes)

5. **Review Process**:
   - Maintainers will review your PR
   - Address feedback and push updates
   - Once approved, a maintainer will merge

## Code Standards

### Formatting

SPFlow uses **Black** for code formatting:

```bash
# Format all code
uv run black spflow tests
```

### Documentation Strings

All public modules, classes, and functions must have docstrings:

```python
def log_likelihood(module: Module, data: torch.Tensor) -> torch.Tensor:
    """Compute log-likelihood of data given module.

    Args:
        module: The probabilistic module to evaluate.
        data: Input data tensor of shape [batch_size, num_features].

    Returns:
        Log-likelihood tensor of shape [batch_size, out_channels].

    Raises:
        ValueError: If data shape doesn't match module scope.
    """
    ...
```

## Testing

### Running Tests

```bash
# Run all tests in parallel
uv run pytest -n 4

# Run specific test file
uv run pytest tests/modules/test_sum.py

# Run specific test
uv run pytest tests/modules/test_sum.py::test_sum_log_likelihood
```

### Writing Tests

- Place tests in `tests/` directory mirroring `spflow/` structure
- Use pytest fixtures for common setup (see `tests/conftest.py`)
- Test with both CPU and GPU when applicable

## Documentation

### Building the Documentation

The project uses **Sphinx** for documentation generation. API documentation is automatically generated from source code
docstrings.

```bash
# Build HTML documentation
cd docs && make html

# View the built documentation
open build/index.html

# Clean build artifacts
cd docs && make clean
```

The generated HTML will be available in `docs/build/index.html`.

### Documentation Standards

- All public modules, classes, and functions must have docstrings
- Use the [Google docstring style](https://numpydoc.readthedocs.io/en/latest/format.html) for consistency
- Include examples in docstrings where appropriate
- Update docs for any API changes or new features

## Versioning

SPFlow follows [Semantic Versioning 2.0.0](https://semver.org/).

See [VERSIONING.md](VERSIONING.md) for a detailed guide.

**Quick Reference:**
- **MAJOR**: Breaking changes (incompatible API changes)
- **MINOR**: New features (backward-compatible)
- **PATCH**: Bug fixes (backward-compatible)

## Code of Conduct

Be respectful and constructive in all interactions. We aim to foster an inclusive and welcoming community.

## License

By contributing to SPFlow, you agree that your contributions will be licensed under the Apache License 2.0.
