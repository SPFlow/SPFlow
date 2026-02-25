from __future__ import annotations

import pytest
import torch

from tests.contract_data import RAT_MULTI_DIST_PARAMS, RAT_PARAMS
from tests.test_helpers.assertions import assert_finite_tensor
from tests.test_helpers.builders import make_rat_spn
from tests.utils.leaves import make_data, make_leaf
from spflow.meta import Scope
from spflow.modules.leaves import Bernoulli, Normal
from spflow.zoo.rat import RatSPN

pytestmark = [pytest.mark.contract, pytest.mark.slow_matrix]


@pytest.mark.parametrize(
    "leaf_cls, d, region_nodes, leaves, num_reps, root_nodes, outer_product, split_mode", RAT_PARAMS
)
def test_log_likelihood_contract(
    leaf_cls, d, region_nodes, leaves, num_reps, root_nodes, outer_product, split_mode
):
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
    assert len(module.scope) == num_features

    data = make_data(cls=leaf_cls, out_features=num_features, n_samples=10)
    lls = module.log_likelihood(data)

    # RAT internals propagate [N, F, C, R] tensors; preserving rank here protects
    # downstream sum/product layers that assume explicit feature/channel axes.
    assert lls.ndim == 4
    assert lls.shape[0] == data.shape[0]
    assert lls.shape[1] == module.out_shape.features
    assert lls.shape[2] == module.out_shape.channels
    assert lls.shape[3] == 1
    assert_finite_tensor(lls)


@pytest.mark.parametrize(
    "region_nodes, leaves, num_reps, root_nodes, outer_product, split_mode", RAT_MULTI_DIST_PARAMS
)
def test_multidistribution_input_contract(
    region_nodes, leaves, num_reps, root_nodes, outer_product, split_mode
):
    out_features_1 = 8
    out_features_2 = 10
    depth = 2

    scope_1 = Scope(list(range(0, out_features_1)))
    scope_2 = Scope(list(range(out_features_1, out_features_1 + out_features_2)))

    module_1 = make_leaf(cls=Normal, out_channels=leaves, scope=scope_1, num_repetitions=num_reps)
    data_1 = make_data(cls=Normal, out_features=out_features_1, n_samples=5)

    module_2 = make_leaf(cls=Bernoulli, out_channels=leaves, scope=scope_2, num_repetitions=num_reps)
    data_2 = make_data(cls=Bernoulli, out_features=out_features_2, n_samples=5)

    data = torch.cat((data_1, data_2), dim=1)

    # This mixed-leaf setup verifies that heterogeneous distributions can coexist
    # as long as their scopes partition the input feature space.
    model = RatSPN(
        leaf_modules=[module_1, module_2],
        n_root_nodes=root_nodes,
        n_region_nodes=region_nodes,
        num_repetitions=num_reps,
        depth=depth,
        outer_product=outer_product,
        split_mode=split_mode,
    )

    lls = model.log_likelihood(data)
    assert lls.shape == (data.shape[0], model.out_shape.features, model.out_shape.channels, 1)

    samples = model.sample()
    assert samples.shape == (1, out_features_1 + out_features_2)
