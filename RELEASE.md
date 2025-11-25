# Release Process

This document describes the release process for SPFlow maintainers.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Release Types](#release-types)
- [Pre-Release Checklist](#pre-release-checklist)
- [Release Steps](#release-steps)
- [Post-Release Steps](#post-release-steps)
- [Hotfix Releases](#hotfix-releases)
- [Backport Releases](#backport-releases)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Access

- Write access to the SPFlow repository
- PyPI maintainer access for the `spflow` package

### Required Tools

```bash
# Configure PyPI credentials (one-time setup)
# Option 1: Use token (recommended)
# Create token at https://pypi.org/manage/account/token/
# Store in ~/.pypirc:
cat > ~/.pypirc << EOF
[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE
EOF

# Option 2: Use username/password
# twine will prompt for credentials during upload
```

### Development Environment

```bash
# Clone repository
git clone git@github.com:SPFlow/SPFlow.git
cd SPFlow

# Set up development environment
uv sync --extra dev
source .venv/bin/activate
```

## Release Types

### Stable Release (X.Y.Z)

Full production-ready release:
- All features complete and tested
- Documentation updated
- No known critical bugs

## Pre-Release Checklist

Before starting the release process, ensure:

### Code Quality

- [ ] All tests pass on main branch:
  ```bash
  uv run pytest -n 4
  ```

- [ ] Code is properly formatted:
  ```bash
  uv run black spflow tests --check
  ```

### Documentation

- [ ] CHANGELOG.md is updated with all changes since last release
- [ ] README.md reflects current version and features
- [ ] All new features are documented
- [ ] Breaking changes are clearly documented
- [ ] Migration guide exists (for MAJOR versions)

### Dependencies

- [ ] All dependencies are up to date and secure
- [ ] Minimum versions are tested and documented
- [ ] No security vulnerabilities in dependencies:
  ```bash
  pip-audit  # or safety check
  ```

### Version Number

- [ ] Version follows semantic versioning rules (see [VERSIONING.md](VERSIONING.md))
- [ ] Version number is appropriate for changes being released

## Release Steps

### Step 1: Prepare Release Branch

```bash
# Ensure you're on the main development branch
git checkout develop
git pull origin develop

# Create release branch (optional, for preparation)
git checkout -b release/vX.Y.Z
```

### Step 2: Update Version Number

Edit `spflow/__init__.py`:

```python
__version__ = "X.Y.Z"  # or "X.Y.Z-alpha.1", etc.
```

### Step 3: Update CHANGELOG.md

Move items from `[Unreleased]` to new version section:

```markdown
## [Unreleased]

## [X.Y.Z] - YYYY-MM-DD

### Added
- New feature A
- New feature B

### Changed
- Updated behavior of feature C

### Fixed
- Bug fix D

### Breaking Changes
- Breaking change E (if MAJOR version)
```

Add comparison link at bottom:

```markdown
[X.Y.Z]: https://github.com/SPFlow/SPFlow/compare/vX.Y.Z-1...vX.Y.Z
```

### Step 4: Commit Version Bump

```bash
git add spflow/__init__.py CHANGELOG.md
git commit -m "chore: bump version to X.Y.Z"
```

### Step 5: Run Final Tests

```bash
# Run complete test suite
.venv/bin/uv run pytest -n 4 --cov=spflow

# Test on clean install
deactivate
uv run venv test_venv
source test_venv/bin/activate
pip install -e .
uv run pytest -n 4
deactivate
rm -rf test_venv
source .venv/bin/activate
```

### Step 6: Create Git Tag

```bash
# Create annotated tag
git tag -a vX.Y.Z -m "Release version X.Y.Z"

# Verify tag
git tag -l vX.Y.Z
git show vX.Y.Z
```

### Step 7: Push Changes

```bash
# Push commit to remote
git push origin develop

# Push tag
git push origin vX.Y.Z

# If using release branch, merge to develop first
git checkout develop
git merge --no-ff release/vX.Y.Z
git push origin develop
git push origin vX.Y.Z
```

### Step 8: Build Distribution

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
uv run build
```

### Step 9: Test Distribution

```bash
# Test installation from built wheel
python -m venv test_dist
source test_dist/bin/activate
pip install dist/spflow-X.Y.Z-py3-none-any.whl
python -c "import spflow; print(spflow.__version__)"
uv run pytest tests/  # Run subset of tests
deactivate
rm -rf test_dist
```

### Step 10: Upload to PyPI

```bash
# Upload to production PyPI
twine upload dist/*

# Verify upload
# Visit: https://pypi.org/project/spflow/X.Y.Z/
```

## Post-Release Steps

### Step 1: Verify Release

```bash
# Test installation from PyPI in fresh environment
python -m venv verify_release
source verify_release/bin/activate
pip install spflow==X.Y.Z
python -c "import spflow; print(spflow.__version__)"
python -c "import spflow; from spflow import log_likelihood, sample"
deactivate
rm -rf verify_release
```

### Step 2: Update Documentation

- [ ] Update documentation site (if exists)
- [ ] Update any version-specific links
- [ ] Announce release in documentation

### Step 3: Merge to Main (Stable Branch)

```bash
# Merge stable releases to main branch
git checkout main
git merge --no-ff vX.Y.Z
git push origin main
```

### Step 5: Prepare for Next Release

```bash
# Update version to next dev version (optional)
# E.g., after 1.2.0 release, set to "1.3.0-dev"
vim spflow/__init__.py

# Add [Unreleased] section in CHANGELOG.md
cat >> CHANGELOG.md << EOF
## [Unreleased]

### Added

### Changed

### Fixed

EOF

git add spflow/__init__.py CHANGELOG.md
git commit -m "chore: prepare for next development cycle"
git push origin develop
```

