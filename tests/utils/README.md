# Utils Test Map

## Ownership
- `test_*_contract.py`: cross-utility contracts for stable public utility behavior.
- `test_*_math.py`: numerical/analytic identity checks (`@pytest.mark.numerical`).
- `test_*_core.py`: implementation branches, guardrails, cache-path behavior.

## Where New Tests Go
- Put reusable behavior invariants in contract files.
- Put closed-form and Monte-Carlo validation in `*_math.py`.
- Put private helper branch checks in `*_core.py`.
