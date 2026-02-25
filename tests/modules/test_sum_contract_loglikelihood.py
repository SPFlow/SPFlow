import pytest
import torch

from tests.modules.module_contract_data import SUM_PARAMS
from tests.modules.test_helpers.builders import build_elementwise_sum, build_sum
from tests.utils.leaves import make_normal_data

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", SUM_PARAMS)
@pytest.mark.parametrize("builder", [build_sum, build_elementwise_sum])
def test_sum_family_log_likelihood_shape_and_finite(
    in_channels: int, out_channels: int, out_features: int, num_reps: int, builder
):
    module = builder(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    data = make_normal_data(out_features=out_features)
    lls = module.log_likelihood(data)
    # Sum-family layers preserve feature count while remapping channel probabilities.
    assert lls.shape == (data.shape[0], module.out_shape.features, module.out_shape.channels, num_reps)
    assert torch.isfinite(lls).all()
