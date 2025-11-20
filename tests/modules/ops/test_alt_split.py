from itertools import product

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.ops import SplitAlternate
from spflow.modules.ops import SplitHalves
from spflow.modules.products import ElementwiseProduct
from spflow.modules.products import OuterProduct
from tests.utils.leaves import make_normal_leaf, make_normal_data

out_channels_values = [1, 6]
features_values_multiplier = [1, 6]
num_splits = [2, 3]
split_type = [SplitHalves, SplitAlternate]
params = list(product(out_channels_values, features_values_multiplier, num_splits, split_type))


cls = [ElementwiseProduct, OuterProduct]

out_channels_values = [1, 3]
out_features_values = [2, 4]
num_repetition_values = [1, 5]

@pytest.mark.parametrize("cls,out_channels,out_features,num_repetitions", product(cls, out_channels_values, out_features_values, num_repetition_values))
def test_split_result(device, cls, out_channels: int, out_features: int, num_repetitions: int):
    out_channels = 10
    num_features = 6
    scope = Scope(list(range(0, num_features)))

    scope_1 = Scope(list(range(0, num_features))[0::2])
    scope_2 = Scope(list(range(0, num_features))[1::2])
    mean = torch.randn(num_features, out_channels, num_repetitions)
    mean_1 = mean[0::2]
    mean_2 = mean[1::2]
    std = torch.rand(num_features, out_channels, num_repetitions)
    std_1 = std[0::2]
    std_2 = std[1::2]
    leaf = make_normal_leaf(scope=scope, mean=mean, std=std)
    leaf_half_1 = make_normal_leaf(scope=scope_1, mean=mean_1, std=std_1)
    leaf_half_2 = make_normal_leaf(scope=scope_2, mean=mean_2, std=std_2)
    split = SplitAlternate(inputs=leaf, num_splits=2, dim=1)
    spn1 = ElementwiseProduct(inputs=split).to(device)
    spn2 = ElementwiseProduct(inputs=[leaf_half_1, leaf_half_2]).to(device)
    assert spn1.out_channels == spn2.out_channels
    assert spn1.out_features == spn2.out_features
    data = make_normal_data(out_features=num_features).to(device)
    ll_1 = spn1.log_likelihood(data)
    ll_2 = spn2.log_likelihood(data)
    assert torch.allclose(ll_1, ll_2)
