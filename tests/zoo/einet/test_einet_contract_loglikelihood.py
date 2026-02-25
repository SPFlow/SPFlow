from __future__ import annotations

import pytest
import torch

from spflow.utils.cache import Cache
from tests.contract_data import EINET_PARAMS_FULL
from tests.test_helpers.assertions import assert_finite_tensor
from tests.test_helpers.builders import make_einet

pytestmark = [pytest.mark.contract, pytest.mark.slow_matrix]


@pytest.mark.parametrize("num_sums,num_leaves,depth,num_reps,layer_type,structure", EINET_PARAMS_FULL)
def test_parametrized_log_likelihood_contract(num_sums, num_leaves, depth, num_reps, layer_type, structure):
    # Keep feature count compatible with deep region splits while still checking
    # small models that are common in unit tests.
    num_features = max(4, 2**depth)
    batch_size = 10

    model = make_einet(
        num_features=num_features,
        num_classes=2,
        num_sums=num_sums,
        num_leaves=num_leaves,
        depth=depth,
        num_repetitions=num_reps,
        layer_type=layer_type,
        structure=structure,
    )

    lls = model.log_likelihood(torch.randn(batch_size, num_features))
    assert lls.shape[0] == batch_size
    assert_finite_tensor(lls)


def test_log_likelihood_cached_contract():
    model = make_einet(num_features=4, num_classes=1, num_sums=5, num_leaves=3, depth=1, num_repetitions=2)
    data = torch.randn(10, 4)
    cache = Cache()

    lls = model.log_likelihood(data, cache=cache)
    # Caching this key is part of the public contract used by repeated inference calls.
    assert "log_likelihood" in cache
    assert_finite_tensor(lls)
