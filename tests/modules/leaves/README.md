# Leaf Test Map

This directory separates leaf tests by responsibility to keep ownership clear.

## Ownership matrix
- Contract tests (`test_leaf_contract_*.py`, all marked `@pytest.mark.contract`): cross-leaf API behavior and invariants.
  - [`test_leaf_contract_loglikelihood.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_leaf_contract_loglikelihood.py)
  - [`test_leaf_contract_sampling.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_leaf_contract_sampling.py)
  - [`test_leaf_contract_marginalization.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_leaf_contract_marginalization.py)
  - [`test_leaf_contract_training.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_leaf_contract_training.py)
  - [`test_leaf_contract_constructor_sanity.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_leaf_contract_constructor_sanity.py)
  - [`test_leaf_contract_constructor_domains.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_leaf_contract_constructor_domains.py)
  - [`test_leaf_contract_conditional.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_leaf_contract_conditional.py)
- Base leaf API behavior (`test_leaf_base.py`, `test_leaf_conditional_mpe.py`): abstract/base class mechanics and dispatch behavior.
- Leaf-specific implementation suites (`test_<leaf>.py`): behavior unique to one distribution/family and edge branches not shared by most leaves.
  - Examples: [`test_cltree.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_cltree.py), [`test_histogram.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_histogram.py), [`test_hypergeometric.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_hypergeometric.py), [`test_piecewise_linear.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_piecewise_linear.py), [`test_uniform.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_uniform.py)
- Differentiable implementation internals: [`test_differentiable_distribution_implementations.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_differentiable_distribution_implementations.py) for implementation-specific `rsample` branches.
- Scope semantics cross-checks: [`test_scope_flexibility.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_scope_flexibility.py) and [`test_scope_order_invariance.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/test_scope_order_invariance.py)

## Integration and math suites outside this folder
These are intentionally not leaf-contract tests. They validate subsystem math and model-level integration:
- [`tests/utils/test_inner_product_math.py`](/Users/steven/projects/SPFlow/tests/utils/test_inner_product_math.py): analytic/closed-form math checks.
- [`tests/utils/test_inner_product_core.py`](/Users/steven/projects/SPFlow/tests/utils/test_inner_product_core.py): core operator behavior and guardrails.
- [`tests/zoo/sos/test_leaf.py`](/Users/steven/projects/SPFlow/tests/zoo/sos/test_leaf.py): SOS integration-level leaf math checks.

Do not move these three files into `tests/modules/leaves/` unless they stop testing integration/math behavior and become pure leaf API contracts.

## Naming conventions
- `test_leaf_contract_*.py`: cross-leaf contract behavior. Must use shared fixtures/matrices from [`conftest.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/conftest.py) and [`leaf_contract_data.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/leaf_contract_data.py).
- `test_<leaf>.py`: one leaf class or tightly coupled family, only for leaf-specific behavior.
- `test_differentiable_distribution_*.py`: implementation-level differentiable distribution tests, not cross-leaf contracts.

## Where a new test goes
1. Cross-leaf invariant expected to hold for most leaves: add to `test_leaf_contract_*.py` and mark as `contract`.
2. Constructor/domain validation common across multiple leaves: add to constructor contract files, not leaf-specific files.
3. Unique leaf branch (special helper, unsupported path, bespoke edge case): add to `test_<leaf>.py`.
4. Differentiable distribution internals (`rsample` algorithm variants): add to `test_differentiable_distribution_implementations.py` unless the behavior is truly leaf-file-specific.
5. Inner-product/triple-product closed forms or subsystem integration behavior: add to `tests/utils/` or `tests/zoo/sos/`, not to leaf contract files.

## Anti-drift checklist (before opening a PR)
1. If the same assertion appears in a contract file and a leaf-specific file, keep it in the contract file unless the leaf file adds unique branch coverage.
2. If adding new cross-leaf parametrization, source the matrix from [`leaf_contract_data.py`](/Users/steven/projects/SPFlow/tests/modules/leaves/leaf_contract_data.py) instead of duplicating local `product(...)` grids.
3. Run `pytest --collect-only -q tests/modules/leaves` and confirm contract tests remain discoverable by filename and marker.
