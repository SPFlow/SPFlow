import pytest

from spflow.meta import Scope
from tests.modules.leaves.leaf_contract_data import MARGINALIZE_LEAF_PARAMS
from tests.utils.leaves import make_leaf

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "leaf_cls, out_channels, prune, marg_rvs, num_reps",
    MARGINALIZE_LEAF_PARAMS,
)
def test_marginalize(leaf_cls, out_channels: int, prune: bool, marg_rvs, num_reps):
    # prune is part of shared contract cases, but leaves currently ignore pruning policy.
    del prune
    out_features = 3
    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )
    marginalized_module = module.marginalize(marg_rvs)

    if len(marg_rvs) == out_features:
        # Fully marginalized leaves collapse to a scalar factor represented as None.
        assert marginalized_module is None
        return

    for _, param in marginalized_module.named_parameters():
        assert param.shape[0] == out_features - len(marg_rvs)

    # Scope equality is set-based; ordering differences should not change semantics.
    marg_scope = Scope(list(set(module.scope.query) - set(marg_rvs)))
    assert marginalized_module.scope == marg_scope
