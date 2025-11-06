import unittest
from itertools import product

from spflow.exceptions import InvalidParameterCombinationError
from spflow.learn import train_gradient_descent
from tests.fixtures import auto_set_test_seed, auto_set_test_device

from spflow.meta.dispatch import init_default_sampling_context, SamplingContext
from tests.utils.leaves import evaluate_log_likelihood
import pytest
import torch

from spflow import maximum_likelihood_estimation, sample, marginalize
from spflow.meta.data import Scope
from spflow.modules import leaf
from tests.utils.leaves import make_leaf, make_data
from spflow.learn.expectation_maximization import expectation_maximization

out_channels_values = [1, 3]
out_features_values = [1, 4]
num_repetitions = [None, 5]
leaf_cls_values = [
    leaf.Bernoulli,
    leaf.Binomial,
    leaf.Categorical,
    leaf.Exponential,
    leaf.Gamma,
    leaf.Geometric,
    leaf.Hypergeometric,
    leaf.LogNormal,
    leaf.NegativeBinomial,
    leaf.Normal,
    leaf.Poisson,
    leaf.Uniform,
]
params = list(product(leaf_cls_values, out_features_values, out_channels_values, num_repetitions))


@pytest.mark.parametrize("leaf_cls, out_features, out_channels, num_reps", params)
def test_log_likelihood(leaf_cls, out_features: int, out_channels: int, num_reps):
    """Test the log likelihood of a normal distribution."""
    module = make_leaf(
        cls=leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )
    data = make_data(cls=leaf_cls, out_features=out_features, n_samples=5)
    evaluate_log_likelihood(module, data)


@pytest.mark.parametrize(
    "leaf_cls, out_features, out_channels, num_reps, is_mpe",
    list(product(leaf_cls_values, out_features_values, out_channels_values, num_repetitions, [True, False])),
)
def test_sample(leaf_cls, out_features: int, out_channels: int, num_reps, is_mpe: bool):
    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    # Setup sampling context
    n_samples = 10
    data = torch.full((n_samples, out_features), torch.nan)
    channel_index = torch.randint(low=0, high=out_channels, size=(n_samples, out_features))
    mask = torch.full((n_samples, out_features), True, dtype=torch.bool)
    if num_reps is not None:
        repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    else:
        repetition_index = None
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)

    # Sample
    samples = sample(module, data, is_mpe=is_mpe, check_support=True, sampling_ctx=sampling_ctx)

    assert samples.shape == (n_samples, out_features)

    # Check finite
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize(
    "leaf_cls, out_features, out_channels, bias_correction, num_reps",
    list(product(leaf_cls_values, out_features_values, out_channels_values, [True, False], num_repetitions)),
)
def test_maximum_likelihood_estimation(
    leaf_cls, out_features: int, out_channels: int, bias_correction: bool, num_reps
):
    # Construct leaf module

    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    # Construct sampler
    scope = Scope(list(range(0, out_features)))
    sampler = make_leaf(cls=leaf_cls, scope=scope, out_channels=1)
    data = sampler.distribution.distribution.sample((100000,)).squeeze(-1)

    maximum_likelihood_estimation(module, data, bias_correction=bias_correction)

    # Check that module and sampler params are equal
    for param_name, param_module in module.distribution.named_parameters():
        param_sampler = getattr(sampler.distribution, param_name)
        if num_reps:
            assert torch.allclose(param_module, param_sampler.unsqueeze(2), atol=3e-1)
        else:
            assert torch.allclose(param_module, param_sampler, atol=3e-1)


@pytest.mark.parametrize("leaf_cls, out_features, out_channels, num_reps", params)
def test_requires_grad(leaf_cls, out_features: int, out_channels: int, num_reps):
    """Test whether the mean and std of a normal distribution require gradients."""
    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    for param in module.distribution.parameters():
        assert param.requires_grad


@pytest.mark.parametrize("leaf_cls,out_features,out_channels, num_reps", params)
def test_gradient_descent_optimization(
    leaf_cls,
    out_features: int,
    out_channels: int,
    num_reps,
):
    # Skip leaves without parameters
    if leaf_cls in [leaf.Hypergeometric, leaf.Uniform]:
        return

    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )
    data = make_data(cls=leaf_cls, out_features=out_features, n_samples=20)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)

    # Clone module parameters before training
    params_before = {k: v.clone() for (k, v) in module.distribution.params().items()}

    train_gradient_descent(module, data_loader, epochs=2)

    # Check that the parameters have changed
    for param_name, param in module.distribution.params().items():
        if param.requires_grad:
            assert not torch.allclose(param, params_before[param_name])


@pytest.mark.parametrize("leaf_cls,out_features,out_channels, num_reps", params)
def test_expectation_maximization(
    leaf_cls,
    out_features: int,
    out_channels: int,
    num_reps,
):
    # Skip leaves without parameters
    if leaf_cls in [leaf.Hypergeometric, leaf.Uniform]:
        return

    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )
    data = make_data(cls=leaf_cls, out_features=out_features, n_samples=20)

    # Clone module parameters before training
    params_before = {k: v.clone() for (k, v) in module.distribution.params().items()}

    expectation_maximization(module, data, max_steps=1)

    # Check that the parameters have changed
    for param_name, param in module.distribution.params().items():
        if param.requires_grad:
            assert not torch.allclose(param, params_before[param_name])


@pytest.mark.parametrize(
    "leaf_cls, out_channels, prune, marg_rvs, num_reps",
    list(
        product(
            leaf_cls_values,
            out_channels_values,
            [True, False],
            [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]],
            num_repetitions,
        )
    ),
)
def test_marginalize(leaf_cls, out_channels: int, prune: bool, marg_rvs, num_reps):
    """Test marginalization of a normal distribution."""
    out_features = 3
    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )
    marginalizeed_module = marginalize(module, marg_rvs)

    # If the number of marginalized rvs is equal to the number of out_features, the module should be None
    if len(marg_rvs) == out_features:
        assert marginalizeed_module is None
        return

    # Check, that the parameters have len(marg_rvs) fewer scopes
    for param_name, param in marginalizeed_module.distribution.named_parameters():
        assert param.shape[0] == out_features - len(marg_rvs)

    # Check, that the correct scopes were marginalized
    marg_scope = Scope(list(set(module.scope.query) - set(marg_rvs)))
    assert marginalizeed_module.scope == marg_scope


@pytest.mark.parametrize(
    "leaf_cls,out_features,out_channels, num_reps",
    product(leaf_cls_values, out_features_values, out_channels_values, num_repetitions),
)
def test_constructor_valid_params(leaf_cls, out_features: int, out_channels: int, num_reps):
    """Test the constructor of the distribution with valid parameters."""

    # Construct module A
    module_a = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    # Get parameters of module A
    module_a_param_dict = module_a.distribution.params()

    # Construct module B with parameters of module A is initialization
    module_b = leaf_cls(scope=module_a.scope, **module_a_param_dict)

    # Check that the parameters are the same
    for name, param in module_b.distribution.params().items():
        assert torch.isfinite(param).all()
        assert torch.allclose(param, module_a_param_dict[name])


@pytest.mark.parametrize(
    "leaf_cls,out_features,out_channels, num_reps",
    product(leaf_cls_values, out_features_values, out_channels_values, num_repetitions),
)
def test_constructor_nan_param(leaf_cls, out_features: int, out_channels: int, num_reps):
    """Test the constructor of a Normal distribution with NaN mean."""
    # Construct module A
    module_a = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    # Get parameters of module A
    nan_params = module_a.distribution.params()
    for key, value in nan_params.items():
        nan_params[key] = torch.full_like(value, torch.nan)

    # Construct module B with parameters of module A is initialization
    with pytest.raises(ValueError):
        module_b = leaf_cls(scope=module_a.scope, **nan_params)


@pytest.mark.parametrize(
    "leaf_cls,out_features,out_channels, num_reps",
    product(leaf_cls_values, out_features_values, out_channels_values, num_repetitions),
)
def test_constructor_inf_param(leaf_cls, out_features: int, out_channels: int, num_reps):
    """Test the constructor of a Normal distribution with NaN mean."""
    # Construct module A
    module_a = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    # Get parameters of module A
    nan_params = module_a.distribution.params()
    for key, value in nan_params.items():
        nan_params[key] = torch.full_like(value, torch.inf)

    # Construct module B with parameters of module A is initialization
    with pytest.raises(ValueError):
        module_b = leaf_cls(scope=module_a.scope, **nan_params)


@pytest.mark.parametrize(
    "leaf_cls,out_features,out_channels, num_reps",
    product(leaf_cls_values, out_features_values, out_channels_values, num_repetitions),
)
def test_constructor_neginf_param(leaf_cls, out_features: int, out_channels: int, num_reps):
    """Test the constructor of a Normal distribution with NaN mean."""
    # Construct module A
    module_a = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    # Get parameters of module A
    nan_params = module_a.distribution.params()
    for key, value in nan_params.items():
        nan_params[key] = torch.full_like(value, -1 * torch.inf)

    # Construct module B with parameters of module A is initialization
    with pytest.raises(ValueError):
        module_b = leaf_cls(scope=module_a.scope, **nan_params)


@pytest.mark.parametrize(
    "leaf_cls,out_features,out_channels, num_reps",
    product(leaf_cls_values, out_features_values, out_channels_values, num_repetitions),
)
def test_constructor_missing_param_and_out_channels(leaf_cls, out_features: int, out_channels: int, num_reps):
    """Test the constructor of a Normal distribution with NaN mean."""
    # Construct module A
    module_a = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    # Get parameters of module A
    none_params = module_a.distribution.params()
    for key, value in none_params.items():
        none_params[key] = None

    # Construct module B with parameters of module A is initialization
    with pytest.raises(InvalidParameterCombinationError):
        module_b = leaf_cls(scope=module_a.scope, **none_params)
