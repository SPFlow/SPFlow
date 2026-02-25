import pytest
import torch

from spflow.learn import expectation_maximization, train_gradient_descent
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext
from tests.modules.module_contract_data import CAT_PARAMS
from tests.modules.test_helpers.builders import build_cat
from tests.utils.leaves import make_normal_data

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("out_channels,out_features,num_reps,dim", CAT_PARAMS)
def test_cat_loglikelihood_contract(out_channels: int, out_features: int, num_reps: int, dim: int):
    module = build_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    )
    data = make_normal_data(out_features=module.out_shape.features)
    lls = module.log_likelihood(data)
    # This tensor shape is the stable interface consumed by downstream operators.
    assert lls.shape == (data.shape[0], module.out_shape.features, module.out_shape.channels, num_reps)


@pytest.mark.parametrize("out_channels,out_features,num_reps,dim", CAT_PARAMS)
def test_cat_sampling_contract(out_channels: int, out_features: int, num_reps: int, dim: int):
    n = 12
    module = build_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    )
    ctx = SamplingContext(
        channel_index=torch.randint(0, module.out_shape.channels, (n, module.out_shape.features)),
        mask=torch.ones((n, module.out_shape.features), dtype=torch.bool),
        repetition_index=torch.randint(0, num_reps, (n,)),
    )
    samples = module._sample(
        data=torch.full((n, module.out_shape.features), torch.nan),
        sampling_ctx=ctx,
        cache=Cache(),
    )
    assert samples.shape == (n, module.out_shape.features)
    # Restrict finiteness check to in-scope RVs; out-of-scope entries may remain NaN by design.
    assert torch.isfinite(samples[:, module.scope.query]).all()


@pytest.mark.parametrize("out_channels,out_features,num_reps,dim", CAT_PARAMS)
def test_cat_training_contract(out_channels: int, out_features: int, num_reps: int, dim: int):
    module = build_cat(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        dim=dim,
    )
    data = make_normal_data(out_features=module.out_shape.features, num_samples=20)
    expectation_maximization(module, data, max_steps=2)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data), batch_size=10)
    train_gradient_descent(module, loader, epochs=1)
