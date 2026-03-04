# Versioning Guidelines

SPFlow follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## Version Format

Version numbers follow the format: `MAJOR.MINOR.PATCH`

## When to Bump Versions

### MAJOR version (X.0.0)

Increment when making **incompatible API changes**:

- Removing public APIs, functions, classes, or modules
- Changing function signatures (removing parameters, changing parameter order)
- Changing the behavior of existing APIs in backward-incompatible ways
- Changing data formats that break existing saved models
- Dropping support for Python versions
- Major architectural changes that require user code modifications

**Examples:**
- Removing a module: `spflow.modules.sum.Sum` → deleted
- Changing signature: `log_likelihood(module, data)` → `log_likelihood(module, data, context)`

### MINOR version (x.Y.0)

Increment when adding **backward-compatible functionality**:

- Adding new modules, functions, or classes
- Adding new optional parameters to existing functions (with defaults)
- Adding new dispatch function implementations
- Adding new learning algorithms or inference methods
- Adding support for new distribution types
- Deprecating functionality (but not removing it)
- Performance improvements without API changes
- Adding new backends or backend features

**Examples:**
- New module: Adding `spflow.modules.leaf.gamma.Gamma`
- New parameter: `log_likelihood(module, data, cache=True)` (default `True`)
- New dispatch: Implementing `sample()` for a new module type
- New algorithm: Adding `spflow.learn.boosted_spn.learn_boosted_spn()`

### PATCH version (x.y.Z)

Increment when making **backward-compatible bug fixes**:

- Fixing incorrect calculations or logic errors
- Fixing memory leaks or performance regressions
- Correcting documentation or type hints
- Fixing crashes or exceptions
- Security patches
- Dependency version updates (within compatible ranges)
- Test improvements

**Examples:**
- Bug fix: Correcting NaN handling in `marginalize()`
- Fix: Repairing incorrect gradient computation in EM
- Security: Patching vulnerability in dependencies
- Docs: Fixing incorrect docstrings or type annotations

## Determining the Next Version

To choose the next version number, review the set of changes since the last release and classify them with the MAJOR/MINOR/PATCH rules above.

Release execution steps (editing version files, changelog updates, tagging, build, and publish) are documented in [RELEASE.md](RELEASE.md).

## Commit Conventions

SPFlow uses [Conventional Commits](https://www.conventionalcommits.org/) to make release intent explicit.

The authoritative commit format, types, scopes, and examples are documented in [CONTRIBUTING.md](CONTRIBUTING.md#commit-conventions).

## Dependency Version Policy

### Required Dependencies

Pin lower bounds, avoid upper bounds unless necessary:

```toml
dependencies = [
    "torch>=2.0.1",  # Good: allows newer versions
    "numpy>=1.26.4",
]
```

### Development Dependencies

Can be more relaxed:

```toml
[project.optional-dependencies]
dev = [
    "pytest",  # No version constraint
    "black~=23.12",  # Compatible with 23.x
]
```

## Deprecation Policy

Before removing functionality:

1. Mark as deprecated in current release (MINOR bump)
2. Add deprecation warning in code
3. Document in CHANGELOG under `### Deprecated`
4. Keep deprecated code for at least **2 MINOR versions** or **6 months**
5. Remove in future MAJOR or MINOR release

**Example:**

```python
import warnings

def old_function():
    warnings.warn(
        "old_function() is deprecated and will be removed in version 2.0.0. "
        "Use new_function() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()
```

## Special Cases

### Hotfixes

For critical bugs in production:

1. Create branch from the stable `main` release tag
2. Apply minimal fix
3. Bump PATCH version
4. Create new tag and release
5. Merge back to both `main` and `develop`
