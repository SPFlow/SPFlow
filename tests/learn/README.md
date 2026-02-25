# Learn Test Map

## Ownership
- `test_*_contract.py`: public learning API contracts (input validation, deterministic behavior, output invariants).
- `test_*.py`: algorithm-specific branches and helper-path coverage.

## Where New Tests Go
- If behavior should hold regardless of internal helper path, add it to a contract file.
- If a test validates a private helper or branch-specific fallback, keep it in module-specific files.

## Current Contract Owners
- EM: `tests/learn/test_expectation_maximization_contract.py`
- Prometheus: `tests/learn/test_prometheus_contract.py`
