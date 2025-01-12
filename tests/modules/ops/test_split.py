
# create tests for split module

from spflow.modules.ops.split import Split
from spflow.modules.ops.split_halves import SplitHalves
from spflow.modules.ops.split_alternate import SplitAlternate
from itertools import product
from spflow.meta.data import Scope
import pytest
from spflow.meta.dispatch import init_default_sampling_context, SamplingContext
from spflow import log_likelihood, sample, marginalize
from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.modules import Sum
from spflow.modules.leaf import Categorical, Binomial
from spflow.modules.ops.cat import Cat
from tests.utils.leaves import make_normal_leaf, make_normal_data
import torch

out_channels_values = [1, 5]
features_values_multiplier = [1, 6]
num_splits = [2,3]
split_type = [SplitHalves,SplitAlternate]
params = list(product(out_channels_values, features_values_multiplier, num_splits, split_type))

def make_split(out_channels=3, out_features=3, num_splits=2, split_type=SplitHalves):
    scope = Scope(list(range(0, out_features)))

    inputs_a = make_normal_leaf(scope, out_channels=out_channels)

    return split_type(inputs=inputs_a, num_splits=num_splits, dim=1)

@pytest.mark.parametrize("out_channels,features_values_multiplier,num_splits,split_type", params)
def test_log_likelihood(out_channels: int, features_values_multiplier: int, num_splits: int, split_type):
    out_channels = 3
    module = make_split(
        out_channels=out_channels,
        out_features=features_values_multiplier * num_splits,
        num_splits=num_splits,
        split_type=split_type
    )
    data = make_normal_data(out_features=module.out_features)
    lls = log_likelihood(module, data)
    assert len(lls) == num_splits
    for ll in lls:
        assert ll.shape == (data.shape[0], module.out_features // num_splits, module.out_channels)


@pytest.mark.parametrize("out_channels,features_values_multiplier,num_splits,split_type", params)
def test_sample(out_channels: int, features_values_multiplier: int, num_splits: int, split_type):
    n_samples = 10
    out_channels = 3
    module = make_split(
        out_channels=out_channels,
        out_features=features_values_multiplier * num_splits,
        num_splits=num_splits,
        split_type=split_type
    )
    for i in range(module.out_channels):
        data = torch.full((n_samples, module.out_features), torch.nan)
        channel_index = torch.randint(low=0, high=module.out_channels, size=(n_samples, module.out_features))
        mask = torch.full((n_samples, module.out_features), True, dtype=torch.bool)
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)
        samples = sample(module, data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, module.scope.query]
        assert torch.isfinite(samples_query).all()
        
