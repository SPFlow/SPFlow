import torch

from spflow import log_likelihood, sample
from spflow.meta.data import Scope
from spflow.modules import leaf
from spflow.modules.leaf import Normal

from spflow.modules.leaf.leaf_module import LeafModule


def evaluate_log_likelihood(module: LeafModule, data: torch.Tensor):
    lls = log_likelihood(module, data, check_support=True)
    assert lls.shape == (data.shape[0], len(module.scope.query), module.out_channels)
    assert torch.isfinite(lls).all()


def evaluate_samples(node: LeafModule, data: torch.Tensor, is_mpe: bool, sampling_ctx):
    samples = sample(node, data, is_mpe=is_mpe, check_support=True, sampling_ctx=sampling_ctx)
    assert samples.shape == data.shape
    s_query = samples[:, node.scope.query]
    assert s_query.shape == (data.shape[0], len(node.scope.query))
    assert torch.isfinite(s_query).all()


def make_normal_leaf(scope=None, out_features=None, out_channels=None, mean=None, std=None) -> Normal:
    """
    Create a Normal leaf module.

    Args:
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.
    """

    if mean is not None:
        out_features = mean.shape[0]
    assert (scope is None) ^ (out_features is None), "Either scope or out_features must be given"

    if scope is None:
        scope = Scope(list(range(0, out_features)))
    elif isinstance(scope, int):
        scope = Scope([scope])
    elif isinstance(scope, list):
        scope = Scope(scope)
    elif isinstance(scope, Scope):
        pass
    else:
        out_features = len(scope.query)

    mean = mean if mean is not None else torch.randn(len(scope.query), out_channels)
    std = std if std is not None else torch.rand(len(scope.query), out_channels) + 1e-8
    return Normal(scope=scope, mean=mean, std=std)


def make_normal_data(mean=0.0, std=1.0, num_samples=10, out_features=2):
    torch.manual_seed(0)
    return torch.randn(num_samples, out_features) * std + mean


def make_leaf(cls, out_channels: int = None, out_features: int = None, scope: Scope = None) -> LeafModule:
    assert (out_features is None) ^ (scope is None), "Either out_features or scope must be provided"

    if scope is None:
        scope = Scope(list(range(0, out_features)))

    # Check special cases
    if cls == leaf.Binomial:
        return leaf.Binomial(scope=scope, out_channels=out_channels, n=torch.ones(1) * 3)
    elif cls == leaf.NegativeBinomial:
        return leaf.NegativeBinomial(scope=scope, out_channels=out_channels, n=torch.ones(1) * 3)
    elif cls == leaf.Categorical:
        return leaf.Categorical(
            scope=scope,
            out_channels=out_channels,
            K=3,
        )
    elif cls == leaf.Hypergeometric:
        return leaf.Hypergeometric(
            scope=scope,
            n=torch.ones((len(scope.query), out_channels)) * 3,
            N=torch.ones((len(scope.query), out_channels)) * 10,
            K=torch.ones((len(scope.query), out_channels)) * 5,
        )
    elif cls == leaf.Uniform:
        return leaf.Uniform(
            scope=scope,
            start=torch.zeros((len(scope.query), out_channels)),
            end=torch.ones((len(scope.query), out_channels)),
        )
    else:
        # Default case: just call the class
        return cls(scope=scope, out_channels=out_channels)


def make_data(cls, out_features: int, n_samples: int = 5) -> torch.Tensor:
    scope = Scope(list(range(0, out_features)))
    return (
        make_leaf(cls=cls, scope=scope, out_channels=1)
        .distribution.distribution.sample((n_samples,))
        .squeeze(-1)
    )
