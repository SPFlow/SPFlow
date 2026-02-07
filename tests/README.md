# Test Quality Checklist

This checklist documents baseline expectations for new and updated tests in SPFlow.

## Test design
- Keep each test focused on one behavior/invariant.
- Prefer explicit arrange/act/assert structure.
- Use descriptive test names with expected behavior and condition.

## Determinism
- Rely on shared autouse seeding in `conftest.py` unless a test needs a custom seed.
- When random behavior is asserted, verify deterministic equivalence by fixing seed.
- Avoid runtime `pytest.skip()` for discoverable invalid combinations; filter param sets ahead of time.

## Assertions
- Use `torch.testing.assert_close` / `np.testing.assert_allclose` for numeric comparisons.
- Include at least one semantic assertion (not only shape/finite checks) when feasible.
- Assert error type with `pytest.raises(...)` for invalid input behavior.

## Parametrization
- Build reusable param matrices with clear names.
- Keep param spaces valid and intentional.
- Prefer parametrization over duplicated test bodies.

## Reuse and contracts
- Put shared test helpers in `tests/utils/`.
- Put reusable module contracts in dedicated contract test modules.
- Keep implementation-specific behavior in small `*_specific.py` files.

## CI progression
- Stage A: report metrics only.
- Stage B: soft gate on regressions.
- Stage C: hard gate for agreed thresholds.
