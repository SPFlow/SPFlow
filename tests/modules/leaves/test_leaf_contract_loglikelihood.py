import pytest

from tests.modules.leaves.leaf_contract_data import LEAF_PARAMS
from tests.utils.leaves import evaluate_log_likelihood, make_data, make_leaf

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("leaf_cls, out_features, out_channels, num_reps", LEAF_PARAMS)
def test_log_likelihood(leaf_cls, out_features: int, out_channels: int, num_reps):
    module = make_leaf(
        cls=leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )
    data = make_data(cls=leaf_cls, out_features=out_features, n_samples=5)
    # Central helper enforces the shared leaf contract (shape, finiteness, and broadcast rules).
    evaluate_log_likelihood(module, data)
