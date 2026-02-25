from itertools import product

import pytest
import torch

from spflow.learn import train_gradient_descent
from spflow.learn.expectation_maximization import expectation_maximization
from tests.modules.leaves.leaf_contract_data import (
    LEAF_CLS_VALUES,
    LEAF_PARAMS,
    OUT_FEATURES_VALUES,
    TRAINABLE_LEAF_PARAMS,
)
from tests.utils.leaves import make_data, make_leaf

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "leaf_cls, out_features, bias_correction",
    list(product(LEAF_CLS_VALUES, OUT_FEATURES_VALUES, [True, False])),
)
def test_maximum_likelihood_estimation(leaf_cls, out_features: int, bias_correction: bool):
    out_channels = 1
    num_reps = 1
    leaf_module = make_leaf(
        cls=leaf_cls, out_channels=out_channels, num_repetitions=num_reps, out_features=out_features
    )
    # Keep a separate generator so target parameters are independent of the model under test.
    leaf_sampler = make_leaf(
        cls=leaf_cls, out_channels=out_channels, num_repetitions=num_reps, out_features=out_features
    )

    data = leaf_sampler.distribution().sample((5000,)).squeeze(-1).squeeze(-1)
    leaf_module.maximum_likelihood_estimation(data, bias_correction=bias_correction)

    for param_name, param_module in leaf_module.named_parameters():
        param_sampler = getattr(leaf_sampler, param_name)
        # Finite-sample MLE is approximate, so tolerance is intentionally loose on absolute error.
        torch.testing.assert_close(param_module, param_sampler, rtol=1e-5, atol=1e-1)


@pytest.mark.parametrize("leaf_cls, out_features, out_channels, num_reps", LEAF_PARAMS)
def test_requires_grad(leaf_cls, out_features: int, out_channels: int, num_reps):
    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    # Contract: trainable leaves expose differentiable parameters by default.
    for param in module.parameters():
        assert param.requires_grad


@pytest.mark.parametrize("leaf_cls,out_features,out_channels, num_reps", TRAINABLE_LEAF_PARAMS)
def test_gradient_descent_optimization(
    leaf_cls,
    out_features: int,
    out_channels: int,
    num_reps,
):
    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )
    data = make_data(cls=leaf_cls, out_features=out_features, n_samples=20)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)

    # Snapshot params so we can assert optimization moved at least one trainable tensor.
    params_before = {k: v.clone() for (k, v) in module.params().items()}

    train_gradient_descent(module, data_loader, epochs=2)

    for param_name, param in module.params().items():
        if param.requires_grad:
            assert not torch.allclose(param, params_before[param_name], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("leaf_cls,out_features,out_channels, num_reps", TRAINABLE_LEAF_PARAMS)
def test_expectation_maximization(
    leaf_cls,
    out_features: int,
    out_channels: int,
    num_reps,
):
    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )
    data = make_data(cls=leaf_cls, out_features=out_features, n_samples=20)

    # Snapshot params so EM is required to produce an actual parameter update.
    params_before = {k: v.clone() for (k, v) in module.params().items()}

    expectation_maximization(module, data, max_steps=1)

    for param_name, param in module.params().items():
        if param.requires_grad:
            assert not torch.allclose(param, params_before[param_name], rtol=0.0, atol=0.0)
