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

## Version Update Process

### 1. Determine Version Bump

Review all changes since the last release:

```bash
# View commits since last tag
git log $(git describe --tags --abbrev=0)..HEAD --oneline

# View changes since last tag
git diff $(git describe --tags --abbrev=0)..HEAD
```

Ask yourself:
- Are there any breaking changes? → **MAJOR**
- Are there new features? → **MINOR**
- Only bug fixes? → **PATCH**

### 2. Update Version Number

Edit `spflow/__init__.py`:

```python
__version__ = "X.Y.Z"
```

### 3. Update CHANGELOG.md

Add entry under `## [Unreleased]` or create new version section:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features and capabilities

### Changed
- Changes to existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security patches
```

### 4. Commit Version Bump

```bash
git add spflow/__init__.py CHANGELOG.md
git commit -m "chore: bump version to X.Y.Z"
```

### 5. Create Git Tag

```bash
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
```

### 6. Build and Publish

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build distribution
python -m build

# Upload to PyPI (requires credentials)
python -m twine upload dist/*
```

## Commit Message Conventions

SPFlow follows [Conventional Commits](https://www.conventionalcommits.org/) for automatic changelog generation and semantic release:

### Format
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat:` - New feature (MINOR version bump)
- `fix:` - Bug fix (PATCH version bump)
- `docs:` - Documentation changes (PATCH version bump)
- `style:` - Code style changes (formatting, no logic change) (PATCH)
- `refactor:` - Code refactoring without feature/bug changes (PATCH)
- `perf:` - Performance improvements (PATCH/MINOR depending on scope)
- `test:` - Adding or updating tests (no version bump)
- `chore:` - Maintenance tasks (no version bump)
- `ci:` - CI/CD changes (no version bump)
- `build:` - Build system changes (no version bump)
- `revert:` - Revert previous commit (depends on reverted change)

### Breaking Changes

Add `BREAKING CHANGE:` in commit footer or `!` after type to indicate MAJOR version bump:

```
feat!: remove deprecated maximum_likelihood_estimation function

BREAKING CHANGE: Use mle() instead of maximum_likelihood_estimation()
```

### Scopes

Optional scopes to specify what changed:

- `modules:` - Changes to module implementations
- `learn:` - Learning algorithms
- `leaf:` - Leaf distributions
- `rat:` - RAT-SPN specific
- `deps:` - Dependency updates

### Examples

```bash
# New feature (MINOR bump)
git commit -m "feat(modules): add Gamma leaf distribution"

# Bug fix (PATCH bump)
git commit -m "fix(dispatch): correct cache initialization bug"

# Breaking change (MAJOR bump)
git commit -m "feat(modules)!: remove deprecated Sum module

BREAKING CHANGE: Use ElementwiseSum instead of Sum"

# Documentation (PATCH bump)
git commit -m "docs: update RAT-SPN usage examples"

# Refactoring (PATCH bump)
git commit -m "refactor(leaf): simplify Normal distribution implementation"

# No version bump
git commit -m "test: add integration tests for EM algorithm"
git commit -m "chore: update development dependencies"
```

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

1. Create branch from release tag
2. Apply minimal fix
3. Bump PATCH version
4. Create new tag and release
5. Merge back to main branch

## Release Checklist

Before each release:

- [ ] All tests pass (`pytest -n 4`)
- [ ] Code is formatted (`black spflow tests`)
- [ ] CHANGELOG.md is updated
- [ ] Version in `spflow/__init__.py` is updated
- [ ] Git tag created
- [ ] Package built and uploaded to PyPI
