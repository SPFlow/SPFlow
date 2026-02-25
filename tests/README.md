# SPFlow Test Suite Guide

This is the canonical guide for how tests are organized in SPFlow.
Suite-specific READMEs (for example `tests/modules/README.md`, `tests/zoo/README.md`) may add local rules, but must not conflict with this document.

## Purpose and philosophy
- Keep tests discoverable: a new contributor should find the right location quickly.
- Keep ownership clear: shared contracts and implementation-specific branches live in different files.
- Keep behavior strong: assertions should validate semantics, not only shape/finite checks.
- Keep breadth intentional: exhaustive parameter matrices are acceptable when they test distinct behavior.

## Top-level ownership map
- `tests/modules/**`: module-level behavior (contracts + module-specific branches).
- `tests/zoo/**`: architecture/composed-model behavior (RAT/Einet/APC/etc.).
- `tests/learn/**`: training/learning APIs and algorithm helper branches.
- `tests/utils/**`: utility contracts, numerical identities, and helper internals.
- `tests/interfaces/**`: sklearn/public interface behavior.
- `tests/conditional/**`, `tests/pipelines/**`, `tests/meta/**`, `tests/measures/**`, `tests/devtools/**`: domain-specific suites.

## Where a new test goes
1. Put the test in a `*_contract*` file if behavior must hold across multiple implementations.
2. Put the test in a module/suite-specific file if it targets implementation branches.
3. Put the test in an integration-style file if it validates composed behavior across components.
4. Avoid adding duplicate assertions to multiple files; choose one owner and reference it.

## Naming conventions
- `test_*_contract_*.py` or `test_*_contract.py`: shared contracts.
- `test_<module>.py` or `test_<module>_specific.py`: implementation-specific behavior.
- `test_*_integration.py` (optional): integration-style composed behavior.
- Test function names should state behavior and condition (`test_<behavior>_<condition>`).

## Marker conventions
- `@pytest.mark.contract`: shared behavior contracts.
- `@pytest.mark.integration_style`: composed multi-component behavior.
- `@pytest.mark.slow_matrix`: large parameter matrices kept for breadth.
- `@pytest.mark.numerical`: numerical identity / closed-form validation.
- Existing suite markers like `slow`, `gpu`, `property` remain valid.

## Shared helpers and matrices
- Use shared contract data from `tests/contract_data.py` for recurring parameter grids.
- Use reusable builders/assertion helpers under `tests/test_helpers/*`.
- Keep legacy helpers in `tests/utils/*` backward compatible while migrating call sites.

## Test design checklist
- Keep each test focused on one behavior/invariant.
- Prefer explicit arrange/act/assert structure.
- Use descriptive test names with expected behavior and condition.
- Rely on shared autouse seeding in `conftest.py` unless a test needs a custom seed.
- Avoid runtime `pytest.skip()` for discoverable invalid combinations; filter params ahead of time.
- Use `torch.testing.assert_close` / `np.testing.assert_allclose` for numeric comparisons.
- Assert explicit error types with `pytest.raises(...)` for invalid paths.

## Validation commands
- Collection parity: `.venv/bin/pytest --collect-only -q tests`
- Focused suite run: `.venv/bin/pytest -q tests/<suite_or_file>`
- Full suite parity: `.venv/bin/pytest -n 4 tests`

## PR checklist for test changes
- Added/updated tests are in the correct ownership location.
- File naming follows conventions.
- Relevant markers are applied.
- Shared matrices/helpers were reused where possible.
- Removed tests (if any) include replacement mapping.
- `collect-only` passes and targeted/full runs were executed as appropriate.
