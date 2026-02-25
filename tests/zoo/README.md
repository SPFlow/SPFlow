# Zoo Test Map

## Ownership
- `test_*_contract_*.py`: cross-architecture behavior contracts (construction/log-likelihood/sampling).
- `test_*_module_specific.py`: architecture-specific branches and edge cases.
- other `test_*.py` files: implementation/integration coverage for their specific zoo family.

## Where New Tests Go
- Put shared behavior that should hold for multiple zoo architectures in a `*_contract_*` file.
- Put RAT-only/Einet-only/APC-only internals and branch guards in the family-specific file.
- Use `@pytest.mark.integration_style` for composed multi-module behaviors.

## Current Contract Owners
- RAT: `tests/zoo/rat/test_rat_contract_*.py`
- EiNet: `tests/zoo/einet/test_einet_contract_*.py`
