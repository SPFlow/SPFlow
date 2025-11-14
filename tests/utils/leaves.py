import torch

from spflow.meta import Scope
from spflow.modules import leaves
from spflow.modules.leaves import Normal
from spflow.modules.leaves.base import LeafModule


def evaluate_log_likelihood(module: LeafModule, data: torch.Tensor):
    lls = module.log_likelihood(data)
    if module.num_repetitions is not None:
        assert lls.shape == (
            data.shape[0],
            len(module.scope.query),
            module.out_channels,
            module.num_repetitions,
        )
    else:
        assert lls.shape == (data.shape[0], len(module.scope.query), module.out_channels)
    assert torch.isfinite(lls).all()


def evaluate_samples(node: LeafModule, data: torch.Tensor, is_mpe: bool, sampling_ctx):
    samples = node.sample(data=data, is_mpe=is_mpe, sampling_ctx=sampling_ctx)
    assert samples.shape == data.shape
    s_query = samples[:, node.scope.query]
    assert s_query.shape == (data.shape[0], len(node.scope.query))
    assert torch.isfinite(s_query).all()


def make_normal_leaf(
    scope=None, out_features=None, out_channels=None, num_repetitions=None, mean=None, std=None
) -> Normal:
    """Create a Normal leaves module.

    Args:
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.
    """
    if mean is not None:
        out_features = mean.shape[0]
    # assert (scope is None) ^ (out_features is None), "Either scope or out_features must be given"

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

    if out_features and scope:
        assert len(scope.query) == out_features, "scope and out_features must have the same length"

    if num_repetitions is not None:
        mean = mean if mean is not None else torch.randn(len(scope.query), out_channels, num_repetitions)
        std = std if std is not None else torch.rand(len(scope.query), out_channels, num_repetitions) + 1e-8
    else:
        mean = mean if mean is not None else torch.randn(len(scope.query), out_channels)
        std = std if std is not None else torch.rand(len(scope.query), out_channels) + 1e-8
    return Normal(scope=scope, mean=mean, std=std, num_repetitions=num_repetitions)


def make_normal_data(mean=0.0, std=1.0, num_samples=10, out_features=2):
    torch.manual_seed(0)
    return torch.randn(num_samples, out_features) * std + mean


def make_leaf(
    cls, out_channels: int = None, out_features: int = None, scope: Scope = None, num_repetitions=None
) -> LeafModule:
    assert (out_features is None) ^ (scope is None), "Either out_features or scope must be provided"

    if scope is None:
        scope = Scope(list(range(0, out_features)))

    # Check special cases
    if cls == leaves.Binomial:
        return leaves.Binomial(
            scope=scope, out_channels=out_channels, n=torch.ones(1) * 3, num_repetitions=num_repetitions
        )
    elif cls == leaves.NegativeBinomial:
        return leaves.NegativeBinomial(
            scope=scope, out_channels=out_channels, n=torch.ones(1) * 3, num_repetitions=num_repetitions
        )
    elif cls == leaves.Categorical:
        return leaves.Categorical(
            scope=scope,
            out_channels=out_channels,
            K=3,
            num_repetitions=num_repetitions,
        )
    elif cls == leaves.Hypergeometric:
        if num_repetitions is None:
            return leaves.Hypergeometric(
                scope=scope,
                n=torch.ones((len(scope.query), out_channels)) * 3,
                N=torch.ones((len(scope.query), out_channels)) * 10,
                K=torch.ones((len(scope.query), out_channels)) * 5,
                num_repetitions=num_repetitions,
            )
        else:
            return leaves.Hypergeometric(
                scope=scope,
                n=torch.ones((len(scope.query), out_channels, num_repetitions)) * 3,
                N=torch.ones((len(scope.query), out_channels, num_repetitions)) * 10,
                K=torch.ones((len(scope.query), out_channels, num_repetitions)) * 5,
                num_repetitions=num_repetitions,
            )
    elif cls == leaves.Uniform:
        if num_repetitions is None:
            return leaves.Uniform(
                scope=scope,
                start=torch.zeros((len(scope.query), out_channels)),
                end=torch.ones((len(scope.query), out_channels)),
                num_repetitions=num_repetitions,
            )
        else:
            return leaves.Uniform(
                scope=scope,
                start=torch.zeros((len(scope.query), out_channels, num_repetitions)),
                end=torch.ones((len(scope.query), out_channels, num_repetitions)),
                num_repetitions=num_repetitions,
            )
    else:
        # Default case: just call the class
        return cls(scope=scope, out_channels=out_channels, num_repetitions=num_repetitions)


def make_leaf_args(cls, out_channels: int = None, scope: Scope = None, num_repetitions=None) -> dict:
    # Check special cases
    if cls == leaves.Binomial or cls == leaves.NegativeBinomial:
        return {"n": torch.ones(1) * 3}
    elif cls == leaves.Categorical:
        return {"K": 3}
    elif cls == leaves.Hypergeometric:
        if num_repetitions is None:
            return {
                "n": torch.ones((len(scope.query), out_channels)) * 3,
                "N": torch.ones((len(scope.query), out_channels)) * 10,
                "K": torch.ones((len(scope.query), out_channels)) * 5,
            }
        else:
            return {
                "n": torch.ones((len(scope.query), out_channels, num_repetitions)) * 3,
                "N": torch.ones((len(scope.query), out_channels, num_repetitions)) * 10,
                "K": torch.ones((len(scope.query), out_channels, num_repetitions)) * 5,
            }
    elif cls == leaves.Uniform:
        if num_repetitions is None:
            return {
                "start": torch.zeros((len(scope.query), out_channels)),
                "end": torch.ones((len(scope.query), out_channels)),
            }
        else:
            return {
                "start": torch.zeros((len(scope.query), out_channels, num_repetitions)),
                "end": torch.ones((len(scope.query), out_channels, num_repetitions)),
            }
    else:
        return {}


def make_cond_leaf(
    cls, out_channels: int = None, out_features: int = None, scope: Scope = None
) -> LeafModule:
    assert (out_features is None) ^ (scope is None), "Either out_features or scope must be provided"

    if scope is None:
        scope = Scope(list(range(0, out_features)))

    event_shape = (len(scope.query), out_channels)
    """
    # Check special cases
    if cls == leaves.CondBinomial:
        return leaves.CondBinomial(scope=scope, out_channels=out_channels, n=torch.ones(1) * 3)
    elif cls == leaves.NegativeBinomial:
        return leaves.NegativeBinomial(scope=scope, out_channels=out_channels, n=torch.ones(1) * 3)
    elif cls == leaves.CondCategorical:
        return leaves.CondCategorical(
            scope=scope,
            out_channels=out_channels,
            K=3,
        )
    elif cls == leaves.CondHypergeometric:
        return leaves.CondHypergeometric(
            scope=scope,
            n=torch.ones((len(scope.query), out_channels)) * 3,
            N=torch.ones((len(scope.query), out_channels)) * 10,
            K=torch.ones((len(scope.query), out_channels)) * 5,
        )
    elif cls == leaves.CondUniform:
        return leaves.CondUniform(
            scope=scope,
            start=torch.zeros((len(scope.query), out_channels)),
            end=torch.ones((len(scope.query), out_channels)),
        )
    else:
    """
    # Default case: just call the class
    mean = torch.randn(event_shape)
    std = torch.rand(event_shape)
    cond_f = lambda data: {"mean": mean, "std": std}
    return cls(scope=scope, cond_f=cond_f)


def make_data(cls, out_features: int, n_samples: int = 5) -> torch.Tensor:
    scope = Scope(list(range(0, out_features)))
    return make_leaf(cls=cls, scope=scope, out_channels=1).distribution.sample((n_samples,)).squeeze(-1)


def make_cond_data(cls, out_features: int, n_samples: int = 5) -> torch.Tensor:
    scope = Scope(list(range(0, out_features)))
    return make_cond_leaf(cls=cls, scope=scope, out_channels=1).distribution.sample((n_samples,)).squeeze(-1)
