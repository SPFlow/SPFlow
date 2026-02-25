from __future__ import annotations

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves import (
    Bernoulli,
    Binomial,
    Categorical,
    Exponential,
    Gamma,
    Geometric,
    Laplace,
    LogNormal,
    NegativeBinomial,
    Normal,
    Poisson,
)
from tests.modules.leaves.leaf_contract_data import CTOR_PARAMS

pytestmark = pytest.mark.contract


def _scope(out_features: int) -> Scope:
    return Scope(list(range(out_features)))


@pytest.mark.parametrize("out_features,out_channels,num_repetitions", CTOR_PARAMS)
def test_loc_scale_leaves_reject_non_positive_scale(
    out_features: int, out_channels: int, num_repetitions: int
):
    shape = (out_features, out_channels, num_repetitions)
    loc = torch.randn(shape)
    scale = torch.rand(shape)

    # These leaves share the same positive-scale invariant.
    for leaf_cls in (Normal, Laplace, LogNormal):
        with pytest.raises(ValueError):
            leaf_cls(scope=_scope(out_features), loc=loc, scale=-1.0 * scale).distribution()
        with pytest.raises(ValueError):
            leaf_cls(scope=_scope(out_features), loc=loc, scale=torch.zeros_like(scale)).distribution()


@pytest.mark.parametrize("out_features,out_channels,num_repetitions", CTOR_PARAMS)
def test_loc_scale_leaves_require_loc_and_scale(out_features: int, out_channels: int, num_repetitions: int):
    shape = (out_features, out_channels, num_repetitions)
    loc = torch.randn(shape)
    scale = torch.rand(shape)

    # Constructor should fail early when only one half of a coupled parameter pair is provided.
    for leaf_cls in (Normal, Laplace, LogNormal):
        with pytest.raises(InvalidParameterCombinationError):
            leaf_cls(scope=_scope(out_features), loc=None, scale=scale)
        with pytest.raises(InvalidParameterCombinationError):
            leaf_cls(scope=_scope(out_features), loc=loc, scale=None)


@pytest.mark.parametrize("out_features,out_channels,num_repetitions", CTOR_PARAMS)
def test_log_normal_rejects_nan_loc_or_scale(out_features: int, out_channels: int, num_repetitions: int):
    shape = (out_features, out_channels, num_repetitions)
    loc = torch.randn(shape)
    scale = torch.rand(shape)

    with pytest.raises(ValueError):
        LogNormal(scope=_scope(out_features), loc=loc * torch.nan, scale=scale).distribution()
    with pytest.raises(ValueError):
        LogNormal(scope=_scope(out_features), loc=loc, scale=scale * torch.nan).distribution()


@pytest.mark.parametrize("out_features,out_channels,num_repetitions", CTOR_PARAMS)
def test_rate_leaves_reject_negative_rate(out_features: int, out_channels: int, num_repetitions: int):
    shape = (out_features, out_channels, num_repetitions)
    rate = torch.rand(shape)

    for leaf_cls in (Exponential, Poisson):
        with pytest.raises(ValueError):
            leaf_cls(scope=_scope(out_features), rate=torch.full_like(rate, -1.0)).distribution()


@pytest.mark.parametrize("out_features,out_channels,num_repetitions", CTOR_PARAMS)
def test_exponential_rejects_zero_rate(out_features: int, out_channels: int, num_repetitions: int):
    shape = (out_features, out_channels, num_repetitions)
    rate = torch.rand(shape)

    with pytest.raises(ValueError):
        Exponential(scope=_scope(out_features), rate=torch.zeros_like(rate)).distribution()


@pytest.mark.parametrize("out_features,out_channels,num_repetitions", CTOR_PARAMS)
def test_gamma_requires_valid_concentration_and_rate(
    out_features: int, out_channels: int, num_repetitions: int
):
    shape = (out_features, out_channels, num_repetitions)
    concentration = torch.rand(shape)
    rate = torch.rand(shape)

    with pytest.raises(ValueError):
        Gamma(
            scope=_scope(out_features), concentration=torch.full_like(concentration, -1.0), rate=rate
        ).distribution()
    with pytest.raises(ValueError):
        Gamma(
            scope=_scope(out_features), concentration=concentration, rate=torch.full_like(rate, -1.0)
        ).distribution()
    with pytest.raises(ValueError):
        Gamma(
            scope=_scope(out_features), concentration=torch.zeros_like(concentration), rate=rate
        ).distribution()
    with pytest.raises(ValueError):
        Gamma(
            scope=_scope(out_features), concentration=concentration, rate=torch.zeros_like(rate)
        ).distribution()

    with pytest.raises(InvalidParameterCombinationError):
        Gamma(scope=_scope(out_features), concentration=None, rate=rate)
    with pytest.raises(InvalidParameterCombinationError):
        Gamma(scope=_scope(out_features), concentration=concentration, rate=None)


@pytest.mark.parametrize("out_features,out_channels,num_repetitions", CTOR_PARAMS)
def test_prob_or_logits_leaves_accept_parameterizations(
    out_features: int, out_channels: int, num_repetitions: int
):
    shape = (out_features, out_channels, num_repetitions)
    scope = _scope(out_features)

    bernoulli_probs = torch.rand(shape)
    Bernoulli(scope=scope, probs=bernoulli_probs)
    Bernoulli(scope=scope, logits=torch.randn(shape))

    geometric_probs = torch.rand(shape)
    Geometric(scope=scope, probs=geometric_probs)
    Geometric(scope=scope, logits=torch.randn(shape))

    categorical_shape = (out_features, out_channels, num_repetitions, 4)
    Categorical(scope=scope, probs=torch.rand(categorical_shape))
    Categorical(scope=scope, logits=torch.randn(categorical_shape))

    total_count = torch.randint(1, 10, shape)
    probs = torch.rand(shape)
    Binomial(scope=scope, total_count=total_count, probs=probs)
    Binomial(scope=scope, total_count=total_count, logits=torch.randn(shape))
    NegativeBinomial(scope=scope, total_count=total_count, probs=probs)
    NegativeBinomial(scope=scope, total_count=total_count, logits=torch.randn(shape))


@pytest.mark.parametrize("out_features,out_channels,num_repetitions", CTOR_PARAMS)
def test_binomial_family_rejects_invalid_probs_and_counts(
    out_features: int, out_channels: int, num_repetitions: int
):
    shape = (out_features, out_channels, num_repetitions)
    scope = _scope(out_features)
    total_count = torch.randint(1, 10, shape)
    probs = torch.rand(shape)

    for leaf_cls in (Binomial, NegativeBinomial):
        with pytest.raises(ValueError):
            leaf_cls(scope=scope, total_count=total_count, probs=1.5 + probs).distribution()
        with pytest.raises(ValueError):
            leaf_cls(scope=scope, total_count=total_count, probs=probs - 1.5).distribution()
        with pytest.raises(ValueError):
            leaf_cls(scope=scope, total_count=torch.full_like(total_count, -1.0), probs=probs).distribution()


def test_prob_or_logits_leaves_reject_invalid_parameter_combinations():
    scope = Scope([0])

    # Probability and logits are mutually exclusive parameterizations.
    with pytest.raises(InvalidParameterCombinationError):
        Bernoulli(scope=scope, probs=torch.tensor([0.5]), logits=torch.tensor([0.0]))
    with pytest.raises(InvalidParameterCombinationError):
        Geometric(scope=scope, probs=torch.tensor([0.5]), logits=torch.tensor([0.0]))
    with pytest.raises(InvalidParameterCombinationError):
        Categorical(scope=scope, probs=torch.tensor([[0.25, 0.75]]), logits=torch.tensor([[0.0, 0.0]]))

    with pytest.raises(InvalidParameterCombinationError):
        Binomial(
            scope=scope,
            total_count=torch.tensor([10]),
            probs=torch.tensor([0.5]),
            logits=torch.tensor([0.0]),
        )
    with pytest.raises(InvalidParameterCombinationError):
        NegativeBinomial(
            scope=scope,
            total_count=torch.tensor([10]),
            probs=torch.tensor([0.5]),
            logits=torch.tensor([0.0]),
        )


def test_binomial_family_requires_total_count():
    scope = Scope([0])
    probs = torch.tensor([0.5])

    with pytest.raises(InvalidParameterCombinationError):
        Binomial(scope=scope, total_count=None, probs=probs)
    with pytest.raises(InvalidParameterCombinationError):
        NegativeBinomial(scope=scope, total_count=None, probs=probs)
