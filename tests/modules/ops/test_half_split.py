import pytest
import torch

from spflow.meta import Scope
from spflow.modules.ops import SplitHalves
from spflow.modules.products import ElementwiseProduct, OuterProduct
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf, make_normal_data

cls = [ElementwiseProduct, OuterProduct]


@pytest.mark.parametrize("cls", cls)
def test_split_result(cls, device):
    torch.manual_seed(0)
    out_channels = 10
    num_features = 6
    scope = Scope(list(range(0, num_features)))
    scope_1 = Scope(list(range(0, num_features // 2)))
    scope_2 = Scope(list(range(num_features // 2, num_features)))
    mean = torch.randn(num_features, out_channels)
    std = torch.rand(num_features, out_channels)
    leaf = make_normal_leaf(scope=scope, mean=mean, std=std).to(device)
    leaf_half_1 = make_normal_leaf(
        scope=scope_1, mean=mean[: num_features // 2], std=std[: num_features // 2]
    ).to(device)
    leaf_half_2 = make_normal_leaf(
        scope=scope_2, mean=mean[num_features // 2 :], std=std[num_features // 2 :]
    ).to(device)
    split = SplitHalves(inputs=leaf, num_splits=2, dim=1).to(device)
    spn1 = cls(inputs=split).to(device)
    spn2 = cls(inputs=[leaf_half_1, leaf_half_2]).to(device)
    assert spn1.out_channels == spn2.out_channels
    assert spn1.out_features == spn2.out_features
    data = make_normal_data(out_features=num_features).to(device)
    ll_1 = spn1.log_likelihood(data)
    ll_2 = spn2.log_likelihood(data)
    assert torch.allclose(ll_1, ll_2)

    n_samples = 100
    num_inputs = 2

    data1 = torch.full((n_samples, spn1.out_features * num_inputs), torch.nan).to(device)
    data2 = torch.full((n_samples, spn1.out_features * num_inputs), torch.nan).to(device)
    mask = torch.full((n_samples, spn1.out_features), True, dtype=torch.bool).to(device)
    channel_index = torch.randint(low=0, high=spn1.out_channels, size=(n_samples, spn1.out_features)).to(
        device
    )
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)
    sampling_ctx2 = SamplingContext(channel_index=channel_index, mask=mask)

    s1 = spn1.sample(data=data1, sampling_ctx=sampling_ctx, is_mpe=True)
    s2 = spn2.sample(data=data2, sampling_ctx=sampling_ctx2, is_mpe=True)

    assert torch.allclose(s1, s2)
