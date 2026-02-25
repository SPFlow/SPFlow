from __future__ import annotations

import pytest
import torch

from spflow.utils.cache import Cache
from tests.contract_data import RAT_PARAMS
from tests.test_helpers.builders import make_rat_spn
from tests.test_helpers.sampling import make_dense_sampling_context

pytestmark = [pytest.mark.contract, pytest.mark.slow_matrix]


@pytest.mark.parametrize(
    "leaf_cls, d, region_nodes, leaves, num_reps, root_nodes, outer_product, split_mode", RAT_PARAMS
)
def test_sampling_contract(
    leaf_cls, d, region_nodes, leaves, num_reps, root_nodes, outer_product, split_mode
):
    n_samples = 100
    num_features = 64
    module = make_rat_spn(
        leaf_cls=leaf_cls,
        depth=d,
        n_region_nodes=region_nodes,
        num_leaves=leaves,
        num_repetitions=num_reps,
        n_root_nodes=root_nodes,
        num_features=num_features,
        outer_product=outer_product,
        split_mode=split_mode,
    )

    data = torch.full((n_samples, num_features), torch.nan)
    # Dense context pre-populates channel/repetition routing so this test exercises
    # the internal _sample contract directly, not just high-level defaults.
    sampling_ctx = make_dense_sampling_context(
        n_samples=n_samples,
        n_features=module.out_shape.features,
        n_channels=module.out_shape.channels,
        n_repetitions=num_reps,
    )
    samples = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())
    assert samples.shape == data.shape
    # Only the module scope is guaranteed to be written; out-of-scope columns may remain NaN.
    assert torch.isfinite(samples[:, module.scope.query]).all()
