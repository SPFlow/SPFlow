# Release Process

This document describes the release process for SPFlow maintainers.

SPFlow uses a two-branch model:
- `develop`: integration branch for regular feature and bug-fix work
- `main`: stable release branch and public release source

The production PyPI publish flow is automated through
`.github/workflows/publish-to-pypi.yml` using PyPI Trusted Publishing.
Maintainers should no longer upload releases from a local machine with `twine`.

## Table of Contents

- [One-Time Setup](#one-time-setup)
- [Pre-Release Checklist](#pre-release-checklist)
- [Stable Release Flow](#stable-release-flow)
- [TestPyPI Rehearsal](#testpypi-rehearsal)
- [Post-Release Verification](#post-release-verification)
- [Hotfix Releases](#hotfix-releases)

## One-Time Setup

Complete these steps once per repository/environment.

### GitHub

- Create protected environments named `pypi` and `testpypi`
- Require manual approval for `pypi`
- Restrict `pypi` deployments to the `main` branch and release tags
- Ensure branch protection requires the CI workflow to pass before merging to `main`

### PyPI

Configure a Trusted Publisher for the production project:

- Owner: `SPFlow`
- Repository: `SPFlow`
- Workflow file: `publish-to-pypi.yml`
- Environment: `pypi`

### TestPyPI

Configure a Trusted Publisher for the TestPyPI project:

- Owner: `SPFlow`
- Repository: `SPFlow`
- Workflow file: `publish-to-pypi.yml`
- Environment: `testpypi`

## Pre-Release Checklist

Before starting a release:

- [ ] All CI jobs pass on the release commit
- [ ] `CHANGELOG.md` is updated
- [ ] `README.md` and docs reflect the release
- [ ] `spflow/__init__.py` contains the release version
- [ ] The release commit has been merged or promoted to `main`

## Stable Release Flow

### Step 1: Prepare the release on `develop`

```bash
git checkout develop
git pull origin develop
```

Update the release metadata:

- bump `spflow/__init__.py`
- update `CHANGELOG.md`
- commit the release prep changes

### Step 2: Promote the release commit to `main`

```bash
git checkout main
git pull origin main
git merge --ff-only develop
git push origin main
```

If you use a release branch, merge that branch to `main` instead. The key rule is
that the public release commit must already be on `main` before the tag/release is created.

### Step 3: Confirm CI is green on `main`

Wait for the CI workflow to pass on the exact release commit. This includes:

- linting
- formatting
- package build validation
- the test matrix

### Step 4: Create the release tag

```bash
git checkout main
git pull origin main
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
```

### Step 5: Publish the GitHub Release

Create a GitHub Release for `vX.Y.Z`.

This triggers `.github/workflows/publish-to-pypi.yml`, which will:

1. build the sdist and wheel
2. run `twine check`
3. verify the wheel contents
4. smoke-install the built wheel
5. wait for `pypi` environment approval
6. publish the already-built artifact to PyPI

Approve the `pypi` environment deployment when the workflow requests it.

## TestPyPI Rehearsal

Use this before the first public 1.x release or whenever you want a dry run.

1. Open the `Publish Python Package` workflow in GitHub Actions.
2. Choose `Run workflow` on the branch or tag you want to test.
3. The workflow will build and publish to TestPyPI project `spflow` through the `testpypi` environment.
4. Verify the uploaded artifact before creating the production GitHub Release.

## Post-Release Verification

### Step 1: Confirm the GitHub workflow succeeded

- `build-release-dists` passed
- `publish-pypi` passed
- The published version matches the GitHub tag

### Step 2: Verify the package from PyPI

```bash
python -m venv verify_release
source verify_release/bin/activate
python -m pip install --upgrade pip
python -m pip install spflow==X.Y.Z
python - <<'PY'
import importlib.metadata
import spflow

version = importlib.metadata.version("spflow")
assert version == spflow.__version__, (version, spflow.__version__)
print(version)
PY
deactivate
rm -rf verify_release
```

### Step 3: Prepare the next development cycle

If needed:

- add a new `[Unreleased]` section to `CHANGELOG.md`
- bump to the next development version
- merge `main` back into `develop` if your release process created divergence

## Hotfix Releases

For a hotfix release:

1. branch from `main` or the last release tag
2. apply the minimal fix
3. bump the patch version
4. merge the fix to `main`
5. wait for green CI
6. tag `vX.Y.Z`
7. publish a GitHub Release
8. back-merge the hotfix to `develop`
