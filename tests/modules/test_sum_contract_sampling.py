import pytest
import torch

from spflow.utils.cache import Cache
from tests.modules.module_contract_data import SUM_PARAMS
from tests.modules.test_helpers.builders import build_elementwise_sum, build_sum
from tests.modules.test_helpers.sampling import make_sampling_context_diff, make_sampling_context_int
from tests.utils.sampling_context_helpers import patch_simple_as_categorical_one_hot

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", SUM_PARAMS)
@pytest.mark.parametrize("builder", [build_sum, build_elementwise_sum])
def test_sum_family_sample_shape_and_finite(
    in_channels: int, out_channels: int, out_features: int, num_reps: int, builder
):
    n_samples = 32
    module = builder(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    ctx = make_sampling_context_int(
        num_samples=n_samples,
        num_features=module.out_shape.features,
        num_channels=module.out_shape.channels,
        num_repetitions=num_reps,
    )
    samples = module._sample(
        # NaN input forces the sampler to populate values through its own routing path.
        data=torch.full((n_samples, module.out_shape.features), torch.nan),
        sampling_ctx=ctx,
        cache=Cache(),
    )
    assert samples.shape == (n_samples, module.out_shape.features)
    # Only variables in the module scope are guaranteed to be written by this node.
    assert torch.isfinite(samples[:, module.scope.query]).all()


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", SUM_PARAMS)
@pytest.mark.parametrize("builder", [build_sum, build_elementwise_sum])
def test_sum_family_diff_sampling_matches_non_diff(
    in_channels: int, out_channels: int, out_features: int, num_reps: int, builder, monkeypatch
):
    n_samples = 20
    module = builder(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    data = torch.full((n_samples, module.out_shape.features), torch.nan)
    int_ctx = make_sampling_context_int(
        num_samples=n_samples,
        num_features=module.out_shape.features,
        num_channels=module.out_shape.channels,
        num_repetitions=num_reps,
    )
    diff_ctx = make_sampling_context_diff(
        num_samples=n_samples,
        num_features=module.out_shape.features,
        num_channels=module.out_shape.channels,
        num_repetitions=num_reps,
    )

    # Patch keeps differentiable routing deterministic so seeded runs are comparable.
    patch_simple_as_categorical_one_hot(monkeypatch)
    torch.manual_seed(1337)
    non_diff = module._sample(data=data.clone(), sampling_ctx=int_ctx, cache=Cache())
    torch.manual_seed(1337)
    diff = module._sample(data=data.clone(), sampling_ctx=diff_ctx, cache=Cache())
    assert non_diff.shape == diff.shape
    assert torch.isfinite(diff).all()
