# Module Test Map (Non-Leaf)

This directory separates module tests by ownership so generic contracts do not drift into implementation-specific suites.
Scope note: this map covers `tests/modules` non-leaf tests only.

## Ownership Matrix
- `sum`/`elementwise_sum`:
  - Generic behavior owner: `test_sum_contract_*.py`
  - Branch/implementation owner: `test_sum.py`, `test_elementwise_sum.py`
- `product`/base products:
  - Generic behavior owner: `test_product_contract_*.py`
  - Branch/implementation owner: `test_product.py`, `test_base_products.py`, `test_elementwise_product.py`, `test_outer_product.py`
- `ops/cat`:
  - Generic behavior owner: `ops/test_ops_cat_contract.py`
  - Branch/implementation owner: `ops/test_cat.py`, `ops/test_cat_permuted_scope.py`
- `ops/split`:
  - Generic behavior owner: `ops/test_ops_split_contract.py`
  - Branch/implementation owner: `ops/test_split.py`, `ops/test_half_split.py`, `ops/test_alt_split.py`, `ops/test_split_by_index.py`
  - Integration-style owner: `ops/test_split_integration.py`
- `einsum`/`linsum`:
  - Generic behavior owner: `einsum/contracts.py` + `test_einsum_contract.py` + `test_linsum_contract.py`
  - Branch/implementation owner: `test_*_specific.py`, `test_*_layer_*.py`
  - Equivalence/integration owner: `test_*_equivalence.py`

## Contract tests (cross-module behavior)
- Sum family:
  - [test_sum_contract_loglikelihood.py](/Users/steven/projects/SPFlow/tests/modules/test_sum_contract_loglikelihood.py)
  - [test_sum_contract_sampling.py](/Users/steven/projects/SPFlow/tests/modules/test_sum_contract_sampling.py)
  - [test_sum_contract_training.py](/Users/steven/projects/SPFlow/tests/modules/test_sum_contract_training.py)
  - [test_sum_contract_marginalization.py](/Users/steven/projects/SPFlow/tests/modules/test_sum_contract_marginalization.py)
  - [test_sum_contract_weights_and_constructor.py](/Users/steven/projects/SPFlow/tests/modules/test_sum_contract_weights_and_constructor.py)
- Product family:
  - [test_product_contract_loglikelihood.py](/Users/steven/projects/SPFlow/tests/modules/test_product_contract_loglikelihood.py)
  - [test_product_contract_sampling.py](/Users/steven/projects/SPFlow/tests/modules/test_product_contract_sampling.py)
  - [test_product_contract_training_marginalization.py](/Users/steven/projects/SPFlow/tests/modules/test_product_contract_training_marginalization.py)
- Ops family:
  - [ops/test_ops_cat_contract.py](/Users/steven/projects/SPFlow/tests/modules/ops/test_ops_cat_contract.py)
  - [ops/test_ops_split_contract.py](/Users/steven/projects/SPFlow/tests/modules/ops/test_ops_split_contract.py)
- Einsum/Linsum family:
  - [einsum/contracts.py](/Users/steven/projects/SPFlow/tests/modules/einsum/contracts.py)
  - [einsum/test_einsum_contract.py](/Users/steven/projects/SPFlow/tests/modules/einsum/test_einsum_contract.py)
  - [einsum/test_linsum_contract.py](/Users/steven/projects/SPFlow/tests/modules/einsum/test_linsum_contract.py)

Contract tests should be marked with `@pytest.mark.contract`.

## Module-specific tests
- Keep `test_<module>.py` files for module-specific branches and edge paths.
- Examples:
  - [test_sum.py](/Users/steven/projects/SPFlow/tests/modules/test_sum.py)
  - [test_elementwise_sum.py](/Users/steven/projects/SPFlow/tests/modules/test_elementwise_sum.py)
  - [test_product.py](/Users/steven/projects/SPFlow/tests/modules/test_product.py)
  - [ops/test_cat.py](/Users/steven/projects/SPFlow/tests/modules/ops/test_cat.py)
  - [ops/test_split.py](/Users/steven/projects/SPFlow/tests/modules/ops/test_split.py)
  - [einsum/test_einsum_specific.py](/Users/steven/projects/SPFlow/tests/modules/einsum/test_einsum_specific.py)
  - [einsum/test_linsum_specific.py](/Users/steven/projects/SPFlow/tests/modules/einsum/test_linsum_specific.py)
  - [conv/test_sum_conv.py](/Users/steven/projects/SPFlow/tests/modules/conv/test_sum_conv.py)
  - [conv/test_prod_conv.py](/Users/steven/projects/SPFlow/tests/modules/conv/test_prod_conv.py)

## Integration-style tests inside modules
- Keep integration-style composed-operator checks in dedicated files, not in contract files.
- Examples:
  - [ops/test_split_integration.py](/Users/steven/projects/SPFlow/tests/modules/ops/test_split_integration.py)
  - [einsum/test_*_equivalence.py](/Users/steven/projects/SPFlow/tests/modules/einsum/test_einsum_equivalence.py)

## Test Routing Rules
1. If behavior must hold for multiple module families, add/extend a contract file first.
2. If behavior depends on one class-specific branch, keep it in `test_<module>.py`.
3. If behavior validates composition of modules (not a single class contract), place it in integration-style files.
4. If a new test duplicates an existing contract assertion matrix, do not add it to module-specific files.

## Contract Markers
- Cross-module contract tests should carry `@pytest.mark.contract`.
- Module-specific and integration-style tests should not use the contract marker unless they define shared behavior across implementations.

## Naming conventions
- `test_*_contract_*.py`: cross-module contracts.
- `test_<module>.py`: module-specific logic and branches.
- `tests/modules/test_helpers/*`: reusable builders and sampling helpers.
- `tests/modules/module_contract_data.py`: shared matrix definitions.

## Where new tests go
1. Behavior expected across module families: add to a contract file.
2. Behavior unique to one implementation branch: add to that module file.
3. Composed/integration behavior: add to dedicated integration-style files.

## Anti-drift rules
1. Do not duplicate matrix-heavy generic tests in both contract and module-specific files.
2. Source shared grids from [module_contract_data.py](/Users/steven/projects/SPFlow/tests/modules/module_contract_data.py).
3. Reuse helpers from `tests/modules/test_helpers/` for new contract tests.

## Quick Checks Before Opening A PR
1. `uv run pytest --collect-only -q tests/modules`
2. `uv run pytest -n 4 tests/modules`
3. `rg "Cross-module.*moved to|contracts moved to" tests/modules -g 'test_*.py'` to verify ownership hints remain explicit in module-specific files.
4. For removed tests, document replacement owner(s) in PR notes (contract file or specific module file).
