import pytest
import torch

from spflow.exceptions import MissingCacheError
from spflow.learn import expectation_maximization, train_gradient_descent
from spflow.utils.cache import Cache
from tests.modules.module_contract_data import SUM_PARAMS
from tests.modules.test_helpers.builders import build_elementwise_sum, build_sum
from tests.utils.leaves import make_normal_data

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", SUM_PARAMS)
@pytest.mark.parametrize("builder", [build_sum, build_elementwise_sum])
def test_sum_family_em_returns_finite_history(
    in_channels: int, out_channels: int, out_features: int, num_reps: int, builder
):
    module = builder(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    ll_history = expectation_maximization(module, make_normal_data(out_features=out_features), max_steps=2)
    assert ll_history.ndim == 1
    assert ll_history.numel() >= 1
    assert ll_history.isfinite().all()


@pytest.mark.parametrize("in_channels,out_channels,out_features,num_reps", SUM_PARAMS)
@pytest.mark.parametrize("builder", [build_sum, build_elementwise_sum])
def test_sum_family_gradient_descent_executes(
    in_channels: int, out_channels: int, out_features: int, num_reps: int, builder
):
    module = builder(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
    )
    dataset = torch.utils.data.TensorDataset(make_normal_data(out_features=out_features, num_samples=20))
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    train_gradient_descent(module, loader, epochs=1)


@pytest.mark.parametrize("builder", [build_sum, build_elementwise_sum])
def test_sum_family_em_requires_cached_lls_contract(builder):
    module = builder(in_channels=2, out_channels=2, out_features=2, num_repetitions=1)
    with pytest.raises(MissingCacheError):
        module._expectation_maximization_step(torch.randn(3, 2), cache=Cache())


@pytest.mark.parametrize("builder", [build_sum, build_elementwise_sum])
def test_sum_family_em_requires_module_lls_contract(builder):
    module = builder(in_channels=2, out_channels=2, out_features=2, num_repetitions=1)
    cache = Cache()
    try:
        inputs = list(module.inputs)
        if len(inputs) == 0:
            inputs = [module.inputs]
    except TypeError:
        inputs = [module.inputs]

    for inp in inputs:
        cache.set(
            "log_likelihood",
            inp,
            torch.zeros(
                (3, module.out_shape.features, module.in_shape.channels, module.out_shape.repetitions)
            ),
        )

    with pytest.raises(MissingCacheError):
        module._expectation_maximization_step(torch.randn(3, 2), cache=cache)
