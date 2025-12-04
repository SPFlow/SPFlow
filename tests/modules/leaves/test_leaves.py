from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.learn import train_gradient_descent
from spflow.learn.expectation_maximization import expectation_maximization
from spflow.meta import Scope
from spflow.modules import leaves
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import evaluate_log_likelihood
from tests.utils.leaves import make_leaf, make_data, create_conditional_parameter_fn, make_leaf_args

out_channels_values = [1, 3]
out_features_values = [1, 4]
num_repetition_values = [1, 2]
leaf_cls_values = [
    leaves.Bernoulli,
    leaves.Binomial,
    leaves.Categorical,
    leaves.Exponential,
    leaves.Gamma,
    leaves.Geometric,
    leaves.Hypergeometric,
    leaves.LogNormal,
    leaves.NegativeBinomial,
    leaves.Normal,
    leaves.Poisson,
    leaves.Uniform,
]
params = list(product(leaf_cls_values, out_features_values, out_channels_values, num_repetition_values))


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
    list(product(leaf_cls_values, out_features_values, out_channels_values, num_repetition_values, [True, False])),
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
    # Always provide repetition_index with num_repetitions dimension
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)

    # Sample
    samples = module.sample(data=data, is_mpe=is_mpe, sampling_ctx=sampling_ctx)

    assert samples.shape == (n_samples, out_features)

    # Check finite
    assert torch.isfinite(samples).all()


def _getattr_nested(obj, name):
    """Get nested attribute using dot notation (e.g., '_distribution.log_p')."""
    parts = name.split(".")
    for part in parts:
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize(
    "leaf_cls, out_features, bias_correction",
    list(product(leaf_cls_values, out_features_values, [True, False])),
)
def test_maximum_likelihood_estimation(
        leaf_cls, out_features: int, bias_correction: bool):
    # Construct leaves module

    out_channels = 1
    num_reps = 1
    leaf_module = make_leaf(cls=leaf_cls, out_channels=out_channels, num_repetitions=num_reps,
                            out_features=out_features)
    # Construct sampler
    leaf_sampler = make_leaf(cls=leaf_cls, out_channels=out_channels, num_repetitions=num_reps,
                             out_features=out_features)

    data = leaf_sampler.distribution.sample((50000,)).squeeze(-1).squeeze(-1)
    leaf_module.maximum_likelihood_estimation(data, bias_correction=bias_correction)

    # Check that module and sampler params are equal
    for param_name, param_module in leaf_module.named_parameters():
        param_sampler = getattr(leaf_sampler, param_name)
        assert torch.allclose(param_module, param_sampler, atol=1e-1)


@pytest.mark.parametrize("leaf_cls, out_features, out_channels, num_reps", params)
def test_requires_grad(leaf_cls, out_features: int, out_channels: int, num_reps):
    """Test whether the mean and std of a normal distribution require gradients."""
    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    for param in module.parameters():
        assert param.requires_grad


@pytest.mark.parametrize("leaf_cls,out_features,out_channels, num_reps", params)
def test_gradient_descent_optimization(
        leaf_cls,
        out_features: int,
        out_channels: int,
        num_reps,
):
    # Skip leaves without parameters
    if leaf_cls in [leaves.Hypergeometric, leaves.Uniform]:
        return

    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )
    data = make_data(cls=leaf_cls, out_features=out_features, n_samples=20)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)

    # Clone module parameters before training
    params_before = {k: v.clone() for (k, v) in module.params().items()}

    train_gradient_descent(module, data_loader, epochs=2)

    # Check that the parameters have changed
    for param_name, param in module.params().items():
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
    if leaf_cls in [leaves.Hypergeometric, leaves.Uniform]:
        return

    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )
    data = make_data(cls=leaf_cls, out_features=out_features, n_samples=20)

    # Clone module parameters before training
    params_before = {k: v.clone() for (k, v) in module.params().items()}

    expectation_maximization(module, data, max_steps=1)

    # Check that the parameters have changed
    for param_name, param in module.params().items():
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
            num_repetition_values,
        )
    ),
)
def test_marginalize(leaf_cls, out_channels: int, prune: bool, marg_rvs, num_reps):
    """Test marginalization of a normal distribution."""
    out_features = 3
    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )
    marginalizeed_module = module.marginalize(marg_rvs)

    # If the number of marginalized rvs is equal to the number of out_features, the module should be None
    if len(marg_rvs) == out_features:
        assert marginalizeed_module is None
        return

    # Check, that the parameters have len(marg_rvs) fewer scopes
    for param_name, param in marginalizeed_module.named_parameters():
        assert param.shape[0] == out_features - len(marg_rvs)

    # Check, that the correct scopes were marginalized
    marg_scope = Scope(list(set(module.scope.query) - set(marg_rvs)))
    assert marginalizeed_module.scope == marg_scope


@pytest.mark.parametrize(
    "leaf_cls,out_features,out_channels, num_reps",
    product(leaf_cls_values, out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_valid_params(leaf_cls, out_features: int, out_channels: int, num_reps):
    """Test the constructor of the distribution with valid parameters."""
    # Construct module A
    module_a = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    # Get parameters of module A
    module_a_param_dict = module_a.params()

    # Construct module B with parameters of module A is initialization
    module_b = leaf_cls(scope=module_a.scope, **module_a_param_dict)

    # Check that the parameters are the same
    for name, param in module_b.params().items():
        assert torch.isfinite(param).all()
        assert torch.allclose(param, module_a_param_dict[name])


@pytest.mark.parametrize(
    "leaf_cls,out_features,out_channels, num_reps",
    product(leaf_cls_values, out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_nan_param(leaf_cls, out_features: int, out_channels: int, num_reps):
    """Test the constructor of a Normal distribution with NaN mean."""
    # Construct module A
    module_a = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    # Get parameters of module A
    nan_params = module_a.params()
    for key, value in nan_params.items():
        nan_params[key] = torch.full_like(value, torch.nan)

    # Construct module B with parameters of module A is initialization
    with pytest.raises(ValueError):
        leaf_cls(scope=module_a.scope, **nan_params).distribution


@pytest.mark.parametrize(
    "leaf_cls,out_features,out_channels, num_reps",
    product(leaf_cls_values, out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_inf_param(leaf_cls, out_features: int, out_channels: int, num_reps):
    """Test the constructor of a Normal distribution with NaN mean."""
    # Construct module A
    module_a = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    # Get parameters of module A
    nan_params = module_a.params()
    for key, value in nan_params.items():
        nan_params[key] = torch.full_like(value, torch.inf)

    # Construct module B with parameters of module A is initialization
    with pytest.raises(ValueError):
        leaf_cls(scope=module_a.scope, **nan_params).distribution


@pytest.mark.parametrize(
    "leaf_cls,out_features,out_channels, num_reps",
    product(leaf_cls_values, out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_neginf_param(leaf_cls, out_features: int, out_channels: int, num_reps):
    """Test the constructor of a Normal distribution with NaN mean."""
    # Construct module A
    module_a = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    # Get parameters of module A
    nan_params = module_a.params()
    for key, value in nan_params.items():
        nan_params[key] = torch.full_like(value, -1 * torch.inf)

    # Construct module B with parameters of module A is initialization
    with pytest.raises(ValueError):
        leaf_cls(scope=module_a.scope, **nan_params).distribution


@pytest.mark.parametrize(
    "leaf_cls,out_features,out_channels, num_reps",
    product(leaf_cls_values, out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_missing_param_and_out_channels(leaf_cls, out_features: int, out_channels: int, num_reps):
    """Test the constructor of a Normal distribution with NaN mean."""
    # Construct module A
    module_a = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    # Get parameters of module A
    none_params = module_a.params()
    for key, value in none_params.items():
        none_params[key] = None

    # Construct module B with parameters of module A is initialization
    with pytest.raises(InvalidParameterCombinationError):
        module_b = leaf_cls(scope=module_a.scope, **none_params)


# Conditional distribution tests
# Note: Hypergeometric and Uniform are excluded as they have no trainable parameters
# and therefore don't support conditional parameter networks
conditional_leaf_cls_values = [
    leaves.Bernoulli,
    leaves.Binomial,
    leaves.Categorical,
    leaves.Exponential,
    leaves.Gamma,
    leaves.Geometric,
    leaves.LogNormal,
    leaves.NegativeBinomial,
    leaves.Normal,
    leaves.Poisson,
]


class TestConditionalLeaves:
    @pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps",
                             product(conditional_leaf_cls_values, out_features_values, out_channels_values,
                                     num_repetition_values))
    def test_conditional_leaf_is_conditional(self, leaf_cls, out_features, out_channels, num_reps):
        """Test that a leaf with parameter network is marked as conditional."""
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        # Non-conditional leaf
        scope = Scope(query)
        leaf = make_leaf(cls=leaf_cls, scope=scope, out_channels=out_channels, num_repetitions=num_reps)
        assert not leaf.is_conditional

        # Conditional leaf
        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope_cond = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(
            cls=leaf_cls, out_channels=out_channels, scope=scope_cond, num_repetitions=num_reps
        )
        leaf_cond = leaf_cls(
            scope=scope_cond, out_channels=out_channels, parameter_fn=param_net, **leaf_args
        )
        assert leaf_cond.is_conditional

    @pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps",
                             product(conditional_leaf_cls_values, out_features_values, out_channels_values,
                                     num_repetition_values))
    def test_conditional_leaf_distribution_with_evidence(self, leaf_cls, out_features, out_channels: int, num_reps):
        """Test that conditional leaf generates distribution from evidence."""
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(cls=leaf_cls, out_channels=out_channels, scope=scope, num_repetitions=num_reps)
        leaf = leaf_cls(scope=scope, out_channels=out_channels, parameter_fn=param_net, **leaf_args)

        # Create evidence tensor
        batch_size = 4
        evidence_tensor = torch.randn(batch_size, len(evidence))

        # Get distribution
        dist = leaf.conditional_distribution(evidence=evidence_tensor)
        assert dist is not None

    @pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps",
                             product(conditional_leaf_cls_values, out_features_values, out_channels_values,
                                     num_repetition_values))
    def test_conditional_leaf_distribution_requires_evidence(self, leaf_cls, out_features, out_channels, num_reps):
        """Test that conditional leaf requires evidence when needed."""
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(cls=leaf_cls, out_channels=out_channels, scope=scope, num_repetitions=num_reps)
        leaf = leaf_cls(scope=scope, out_channels=out_channels, parameter_fn=param_net, **leaf_args)

        # Should raise error if evidence is not provided
        # Note: Gamma has a custom conditional_distribution that doesn't check for None,
        # so it raises AttributeError instead of ValueError
        if leaf_cls == leaves.Gamma:
            with pytest.raises(AttributeError):
                leaf.conditional_distribution(evidence=None)
        else:
            with pytest.raises(ValueError, match="Evidence tensor must be provided"):
                leaf.conditional_distribution(evidence=None)

    @pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps",
                             product(conditional_leaf_cls_values, out_features_values, out_channels_values,
                                     num_repetition_values))
    def test_conditional_leaf_likelihood(self, leaf_cls, out_features, out_channels, num_reps):
        """Test likelihood computation with conditional leaf."""
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(cls=leaf_cls, out_channels=out_channels, scope=scope, num_repetitions=num_reps)
        leaf = leaf_cls(scope=scope, out_channels=out_channels, parameter_fn=param_net, **leaf_args)

        # Create random evidence data
        batch_size = 4
        evidence_data = torch.randn(batch_size, len(evidence))

        # Create query data by sampling from the conditional distribution
        dist = leaf.conditional_distribution(evidence=evidence_data)
        query_data = dist.sample()

        # Verify the sample shape contains batch_size and number of features
        # Different distributions may have different shapes with num_repetitions
        assert query_data.shape[0] == batch_size
        assert query_data.shape[1] == len(query)
        assert torch.isfinite(query_data).all()

    @pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps",
                             product(conditional_leaf_cls_values, out_features_values, out_channels_values,
                                     num_repetition_values))
    def test_conditional_leaf_sampling(self, leaf_cls, out_features, out_channels, num_reps):
        """Test sampling from conditional leaf."""
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(cls=leaf_cls, out_channels=out_channels, scope=scope, num_repetitions=num_reps)
        leaf = leaf_cls(scope=scope, out_channels=out_channels, parameter_fn=param_net, **leaf_args)

        # Create random evidence data
        batch_size = 4
        evidence_data = torch.randn(batch_size, len(evidence))

        # Test that we can get the conditional distribution and it has samples
        dist = leaf.conditional_distribution(evidence=evidence_data)
        assert dist is not None

    @pytest.mark.parametrize(
        "leaf_cls,out_features,out_channels,num_reps",
        product(conditional_leaf_cls_values, [1], [1], [1]),
    )
    def test_conditional_leaf_parameter_fn_gradients(self, leaf_cls, out_features, out_channels, num_reps):
        """Ensure a gradient step populates .grad for parameter network params."""
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(cls=leaf_cls, out_channels=out_channels, scope=scope, num_repetitions=num_reps)
        leaf = leaf_cls(scope=scope, out_channels=out_channels, parameter_fn=param_net, **leaf_args)

        batch_size = 3
        evidence_data = torch.randn(batch_size, len(evidence))
        dist = leaf.conditional_distribution(evidence=evidence_data)
        query_data = dist.sample().reshape(batch_size, -1).float()
        query_data = query_data[:, : len(query)]
        data = torch.cat([query_data, evidence_data], dim=1)

        optimizer = torch.optim.SGD(leaf.parameters(), lr=1e-2)
        optimizer.zero_grad()
        loss = -leaf.log_likelihood(data).sum()
        loss.backward()
        optimizer.step()

        grads = [param.grad for param in leaf.parameter_fn.parameters() if param.requires_grad]
        assert grads
        for grad in grads:
            assert grad is not None
            assert torch.isfinite(grad).all()

    @pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps",
                             product(conditional_leaf_cls_values, out_features_values, out_channels_values,
                                     num_repetition_values))
    def test_conditional_leaf_mle_not_supported(self, leaf_cls, out_features, out_channels, num_reps):
        """Test that MLE raises error for conditional leaf."""
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(cls=leaf_cls, out_channels=out_channels, scope=scope, num_repetitions=num_reps)
        leaf = leaf_cls(scope=scope, out_channels=out_channels, parameter_fn=param_net, **leaf_args)

        batch_size = 4
        n_vars = max(query + evidence) + 1
        data = torch.randn(batch_size, n_vars)

        with pytest.raises(RuntimeError, match="MLE not supported for conditional"):
            leaf.maximum_likelihood_estimation(data=data)

    @pytest.mark.parametrize("leaf_cls,out_features,out_channels,num_reps",
                             product(conditional_leaf_cls_values, out_features_values, out_channels_values,
                                     num_repetition_values))
    def test_conditional_leaf_marginalization_not_supported(self, leaf_cls, out_features, out_channels, num_reps):
        """Test that marginalization raises error for conditional leaf."""
        query = list(range(out_features))
        evidence = [out_features, out_features + 1]

        param_net = create_conditional_parameter_fn(
            distribution_class=leaf_cls,
            out_features=out_features,
            out_channels=out_channels,
            evidence_size=len(evidence),
            num_repetitions=num_reps,
        )
        scope = Scope(query, evidence=evidence)
        leaf_args = make_leaf_args(cls=leaf_cls, out_channels=out_channels, scope=scope, num_repetitions=num_reps)
        leaf = leaf_cls(scope=scope, out_channels=out_channels, parameter_fn=param_net, **leaf_args)

        with pytest.raises(RuntimeError, match="Marginalization not supported for conditional"):
            leaf.marginalize(marg_rvs=[0])
