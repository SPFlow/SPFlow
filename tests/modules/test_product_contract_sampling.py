import pytest
import torch

from spflow.meta import Scope
from spflow.modules.products import ElementwiseProduct, OuterProduct
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.modules.module_contract_data import PRODUCT_PARAMS
from tests.modules.test_helpers.builders import build_product
from tests.utils.leaves import DummyLeaf, make_leaf

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("in_channels,out_features,num_reps", PRODUCT_PARAMS)
def test_product_sample_contract(in_channels: int, out_features: int, num_reps: int):
    n_samples = 12
    module = build_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)
    sampling_ctx = SamplingContext(
        channel_index=torch.randint(0, module.out_shape.channels, (n_samples, module.out_shape.features)),
        mask=torch.ones((n_samples, module.out_shape.features), dtype=torch.bool),
        repetition_index=torch.randint(0, num_reps, (n_samples,)),
    )
    samples = module._sample(
        # Start from NaNs to ensure every scoped position is actively written.
        data=torch.full((n_samples, out_features), torch.nan),
        sampling_ctx=sampling_ctx,
        cache=Cache(),
    )
    assert samples.shape == (n_samples, out_features)
    # Contract only requires finite values on this module's scope.
    assert torch.isfinite(samples[:, module.scope.query]).all()


@pytest.mark.parametrize("in_channels,out_features,num_reps", PRODUCT_PARAMS)
def test_product_diff_sampling_equals_non_diff_contract(in_channels: int, out_features: int, num_reps: int):
    n_samples = 16
    module = build_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)
    channel_index = torch.randint(0, module.out_shape.channels, (n_samples, 1))
    repetition_index = torch.randint(0, num_reps, (n_samples,))
    sampling_ctx_a = SamplingContext(
        channel_index=channel_index,
        mask=torch.ones((n_samples, 1), dtype=torch.bool),
        repetition_index=repetition_index,
    )
    sampling_ctx_b = SamplingContext(
        channel_index=to_one_hot(channel_index, dim=-1, dim_size=module.out_shape.channels),
        mask=torch.ones((n_samples, 1), dtype=torch.bool),
        repetition_index=to_one_hot(repetition_index, dim=-1, dim_size=num_reps),
        is_differentiable=True,
        hard=True,
    )

    # Matching seeds make stochastic tie-breaking identical across both routes.
    torch.manual_seed(1337)
    a = module._sample(torch.full((n_samples, out_features), torch.nan), sampling_ctx_a, Cache())
    torch.manual_seed(1337)
    b = module._sample(torch.full((n_samples, out_features), torch.nan), sampling_ctx_b, Cache())
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("cls", [ElementwiseProduct, OuterProduct])
def test_base_products_diff_sampling_smoke_contract(cls):
    n_samples = 16
    scope_a = Scope([0, 1, 2])
    scope_b = Scope([3, 4, 5])
    inputs = [
        make_leaf(cls=DummyLeaf, out_channels=3, scope=scope_a, num_repetitions=1),
        make_leaf(cls=DummyLeaf, out_channels=3, scope=scope_b, num_repetitions=1),
    ]
    module = cls(inputs=inputs)
    ctx = SamplingContext(
        channel_index=to_one_hot(
            torch.randint(0, module.out_shape.channels, (n_samples, module.out_shape.features)),
            dim=-1,
            dim_size=module.out_shape.channels,
        ),
        mask=torch.ones((n_samples, module.out_shape.features), dtype=torch.bool),
        repetition_index=to_one_hot(torch.zeros((n_samples,), dtype=torch.long), dim=-1, dim_size=1),
        is_differentiable=True,
        hard=True,
    )
    out = module._sample(
        data=torch.full((n_samples, 6), torch.nan),
        sampling_ctx=ctx,
        cache=Cache(),
    )
    # Smoke test guards that differentiable indexing does not produce NaNs.
    assert torch.isfinite(out[:, module.scope.query]).all()
