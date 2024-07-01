import unittest
from itertools import product

from spflow.exceptions import InvalidParameterCombinationError
from spflow.learn import train_gradient_descent
from tests.fixtures import auto_set_test_seed

from spflow.meta.dispatch import init_default_sampling_context
from tests.utils.leaves import evaluate_log_likelihood
import pytest
import torch

from spflow import maximum_likelihood_estimation, sample, marginalize
from spflow.meta.data import Scope
from spflow.modules import leaf
from tests.utils.leaves import make_leaf, make_data

out_channels_values = [1, 3]
out_features_values = [1, 4]
cls_values = [
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
params = list(product(cls_values, out_features_values, out_channels_values))


@pytest.mark.parametrize("cls, out_features, out_channels", params)
def test_log_likelihood(cls, out_features: int, out_channels: int):
    """Test the log likelihood of a normal distribution."""
    module = make_leaf(cls=cls, out_channels=out_channels, out_features=out_features)
    data = make_data(cls=cls, out_features=out_features, n_samples=5)
    evaluate_log_likelihood(module, data)


@pytest.mark.parametrize(
    "cls, out_features, out_channels, is_mpe",
    list(product(cls_values, out_features_values, out_channels_values, [True, False])),
)
def test_sample(cls, out_features: int, out_channels: int, is_mpe: bool):
    module = make_leaf(cls, out_channels=out_channels, out_features=out_features)

    # Setup sampling context
    n_samples = 100
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)
    data = torch.full((n_samples, out_features), torch.nan)
    sampling_ctx.output_ids = torch.randint(low=0, high=out_channels, size=(n_samples, out_features))

    # Sample
    samples = sample(module, data, is_mpe=is_mpe, check_support=True, sampling_ctx=sampling_ctx)

    # Check shape
    assert samples.shape == (n_samples, out_features)

    # Check finite
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize(
    "cls, out_features, out_channels, bias_correction",
    list(product(cls_values, out_features_values, out_channels_values, [True, False])),
)
def test_maximum_likelihood_estimation(cls, out_features: int, out_channels: int, bias_correction: bool):
    # Construct leaf module
    module = make_leaf(cls, out_channels=out_channels, out_features=out_features)

    # Construct sampler
    scope = Scope(list(range(0, out_features)))
    sampler = make_leaf(cls=cls, scope=scope, out_channels=1)
    data = sampler.distribution.distribution.sample((100000,)).squeeze(-1)

    maximum_likelihood_estimation(module, data, bias_correction=bias_correction)

    # Check that module and sampler params are equal
    for param_name, param_module in module.distribution.named_parameters():
        param_sampler = getattr(sampler.distribution, param_name)
        assert torch.allclose(param_module, param_sampler, atol=1e-1)


@pytest.mark.parametrize("cls, out_features, out_channels", params)
def test_requires_grad(cls, out_features: int, out_channels: int):
    """Test whether the mean and std of a normal distribution require gradients."""
    module = make_leaf(cls, out_channels=out_channels, out_features=out_features)

    for param in module.distribution.parameters():
        assert param.requires_grad


@pytest.mark.parametrize("cls,out_features,out_channels", params)
def test_gradient_descent_optimization(
    cls, out_features: int,out_channels: int,
):

    # Skip leaves without parameters
    if cls in [leaf.Hypergeometric, leaf.Uniform]:
        return

    module = make_leaf(cls, out_channels=out_channels, out_features=out_features)
    data = make_data(cls=cls, out_features=out_features, n_samples=20)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    train_gradient_descent(module, data_loader, epochs=1)


@pytest.mark.parametrize(
    "cls, out_channels, prune, marg_rvs",
    list(
        product(
            cls_values,
            out_channels_values,
            [True, False],
            [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]],
        )
    ),
)
def test_marginalize(cls, out_channels: int, prune: bool, marg_rvs):
    """Test marginalization of a normal distribution."""
    out_features = 3
    module = make_leaf(cls, out_channels=out_channels, out_features=out_features)
    marg_rvs = [1, 2]
    marginalizeed_module = marginalize(module, marg_rvs)

    # If the number of marginalized rvs is equal to the number of out_features, the module should be None
    if len(marg_rvs) == out_features:
        assert marginalizeed_module is None

    # Check, that the parameters have len(marg_rvs) fewer scopes
    for param_name, param in marginalizeed_module.distribution.named_parameters():
        assert param.shape[0] == out_features - len(marg_rvs)


@pytest.mark.parametrize(
    "cls,out_features,out_channels", product(cls_values, out_features_values, out_channels_values)
)
def test_normal_constructor_valid_params(cls, out_features: int, out_channels: int):
    """Test the constructor of the distribution with valid parameters."""

    # Construct module A
    module_a = make_leaf(cls, out_channels=out_channels, out_features=out_features)

    # Get parameters of module A
    module_a_param_dict = module_a.distribution.params()

    # Construct module B with parameters of module A is initialization
    module_b = cls(scope=module_a.scope, **module_a_param_dict)

    # Check that the parameters are the same
    for name, param in module_b.distribution.params().items():
        assert torch.isfinite(param).all()
        assert torch.allclose(param, module_a_param_dict[name])


@pytest.mark.parametrize(
    "cls,out_features,out_channels", product(cls_values, out_features_values, out_channels_values)
)
def test_normal_constructor_nan_param(cls, out_features: int, out_channels: int):
    """Test the constructor of a Normal distribution with NaN mean."""
    # Construct module A
    module_a = make_leaf(cls, out_channels=out_channels, out_features=out_features)

    # Get parameters of module A
    nan_params = module_a.distribution.params()
    for key, value in nan_params.items():
        nan_params[key] = torch.full_like(value, torch.nan)

    # Construct module B with parameters of module A is initialization
    with pytest.raises(ValueError):
        module_b = cls(scope=module_a.scope, **nan_params)


@pytest.mark.parametrize(
    "cls,out_features,out_channels", product(cls_values, out_features_values, out_channels_values)
)
def test_normal_constructor_inf_param(cls, out_features: int, out_channels: int):
    """Test the constructor of a Normal distribution with NaN mean."""
    # Construct module A
    module_a = make_leaf(cls, out_channels=out_channels, out_features=out_features)

    # Get parameters of module A
    nan_params = module_a.distribution.params()
    for key, value in nan_params.items():
        nan_params[key] = torch.full_like(value, torch.inf)

    # Construct module B with parameters of module A is initialization
    with pytest.raises(ValueError):
        module_b = cls(scope=module_a.scope, **nan_params)


@pytest.mark.parametrize(
    "cls,out_features,out_channels", product(cls_values, out_features_values, out_channels_values)
)
def test_normal_constructor_neginf_param(cls, out_features: int, out_channels: int):
    """Test the constructor of a Normal distribution with NaN mean."""
    # Construct module A
    module_a = make_leaf(cls, out_channels=out_channels, out_features=out_features)

    # Get parameters of module A
    nan_params = module_a.distribution.params()
    for key, value in nan_params.items():
        nan_params[key] = -1 * torch.full_like(value, torch.inf)

    # Construct module B with parameters of module A is initialization
    with pytest.raises(ValueError):
        module_b = cls(scope=module_a.scope, **nan_params)


@pytest.mark.parametrize(
    "cls,out_features,out_channels", product(cls_values, out_features_values, out_channels_values)
)
def test_normal_constructor_missing_param_and_out_channels(cls, out_features: int, out_channels: int):
    """Test the constructor of a Normal distribution with NaN mean."""
    # Construct module A
    module_a = make_leaf(cls, out_channels=out_channels, out_features=out_features)

    # Get parameters of module A
    nan_params = module_a.distribution.params()
    for key, value in nan_params.items():
        nan_params[key] = None

    # Construct module B with parameters of module A is initialization
    with pytest.raises(InvalidParameterCombinationError):
        module_b = cls(scope=module_a.scope, **nan_params)


if __name__ == "__main__":
    unittest.main()
