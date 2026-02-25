"""Shared leaf contract metadata and parameter grids for tests."""

from __future__ import annotations

from itertools import product

import torch

from spflow.meta import Scope
from spflow.modules import leaves

OUT_CHANNELS_VALUES = [1, 3]
OUT_FEATURES_VALUES = [1, 4]
NUM_REPETITION_VALUES = [1, 2]
CTOR_OUT_CHANNELS_VALUES = [1, 5]
CTOR_OUT_FEATURES_VALUES = [1, 6]
CTOR_NUM_REPETITION_VALUES = [1, 4]

LEAF_CLS_VALUES = [
    leaves.Bernoulli,
    leaves.Binomial,
    leaves.Categorical,
    leaves.Exponential,
    leaves.Gamma,
    leaves.Geometric,
    leaves.Hypergeometric,
    leaves.Laplace,
    leaves.LogNormal,
    leaves.NegativeBinomial,
    leaves.Normal,
    leaves.Poisson,
    leaves.Uniform,
]

DIFFERENTIABLE_EQ_LEAF_CLS_VALUES = [
    leaves.Binomial,
    leaves.Bernoulli,
    leaves.Categorical,
    leaves.Exponential,
    leaves.Gamma,
    leaves.Laplace,
    leaves.LogNormal,
    leaves.Normal,
    leaves.Uniform,
]

DIFF_SAMPLING_SUPPORTED_LEAF_CLS_VALUES = [
    leaves.Bernoulli,
    leaves.Binomial,
    leaves.Categorical,
    leaves.Exponential,
    leaves.Gamma,
    leaves.Geometric,
    leaves.Hypergeometric,
    leaves.Laplace,
    leaves.LogNormal,
    leaves.NegativeBinomial,
    leaves.Normal,
    leaves.Poisson,
    leaves.Uniform,
]

NON_TRAINABLE_LEAF_CLS_VALUES = [leaves.Hypergeometric, leaves.Uniform]

CONDITIONAL_LEAF_CLS_VALUES = [
    leaves.Bernoulli,
    leaves.Binomial,
    leaves.Categorical,
    leaves.Exponential,
    leaves.Gamma,
    leaves.Geometric,
    leaves.Laplace,
    leaves.LogNormal,
    leaves.NegativeBinomial,
    leaves.Normal,
    leaves.Poisson,
]

LEAF_PARAMS = list(product(LEAF_CLS_VALUES, OUT_FEATURES_VALUES, OUT_CHANNELS_VALUES, NUM_REPETITION_VALUES))

TRAINABLE_LEAF_PARAMS = list(
    product(
        [leaf_cls for leaf_cls in LEAF_CLS_VALUES if leaf_cls not in NON_TRAINABLE_LEAF_CLS_VALUES],
        OUT_FEATURES_VALUES,
        OUT_CHANNELS_VALUES,
        NUM_REPETITION_VALUES,
    )
)

LEAF_SAMPLE_PARAMS = list(
    product(LEAF_CLS_VALUES, OUT_FEATURES_VALUES, OUT_CHANNELS_VALUES, NUM_REPETITION_VALUES, [True, False])
)

CONDITIONAL_LEAF_PARAMS = list(
    product(CONDITIONAL_LEAF_CLS_VALUES, OUT_FEATURES_VALUES, OUT_CHANNELS_VALUES, NUM_REPETITION_VALUES)
)

CONDITIONAL_LEAF_GRAD_PARAMS = list(product(CONDITIONAL_LEAF_CLS_VALUES, [1], [1], [1]))

MARGINALIZE_LEAF_PARAMS = list(
    product(
        LEAF_CLS_VALUES,
        OUT_CHANNELS_VALUES,
        [True, False],
        [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]],
        NUM_REPETITION_VALUES,
    )
)
CTOR_PARAMS = list(product(CTOR_OUT_FEATURES_VALUES, CTOR_OUT_CHANNELS_VALUES, CTOR_NUM_REPETITION_VALUES))

DIFF_DISTRIBUTION_LEAF_CASES = [
    (lambda: leaves.Normal(scope=Scope([0])), True),
    (lambda: leaves.Laplace(scope=Scope([0])), True),
    (lambda: leaves.LogNormal(scope=Scope([0])), True),
    (lambda: leaves.Gamma(scope=Scope([0])), True),
    (lambda: leaves.Exponential(scope=Scope([0])), True),
    (
        lambda: leaves.Uniform(
            scope=Scope([0]),
            low=torch.tensor([[[0.0]]]),
            high=torch.tensor([[[1.0]]]),
        ),
        False,
    ),
    (lambda: leaves.Bernoulli(scope=Scope([0])), True),
    (lambda: leaves.Binomial(scope=Scope([0]), total_count=torch.tensor([[[4.0]]])), True),
    (lambda: leaves.Categorical(scope=Scope([0]), K=3), True),
    (lambda: leaves.Geometric(scope=Scope([0])), True),
    (lambda: leaves.Poisson(scope=Scope([0])), True),
    (lambda: leaves.NegativeBinomial(scope=Scope([0]), total_count=torch.tensor([[[4.0]]])), True),
    (
        lambda: leaves.Hypergeometric(
            scope=Scope([0]),
            K=torch.tensor([[[2.0]]]),
            N=torch.tensor([[[5.0]]]),
            n=torch.tensor([[[2.0]]]),
        ),
        False,
    ),
    (lambda: leaves.Histogram(scope=Scope([0]), bin_edges=torch.tensor([0.0, 1.0, 2.0])), True),
]

UNSUPPORTED_DIFF_DISTRIBUTION_LEAF_CASES = [
    lambda: leaves.PiecewiseLinear(scope=Scope([0])),
    lambda: leaves.CLTree(scope=Scope([0, 1]), K=2),
]
