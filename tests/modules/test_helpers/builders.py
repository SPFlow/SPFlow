"""Builder helpers for non-leaf contract tests."""

from __future__ import annotations

import torch

from spflow.meta import Scope
from spflow.modules.ops import Cat
from spflow.modules.products import Product
from spflow.modules.sums import Sum
from spflow.modules.sums.elementwise_sum import ElementwiseSum
from tests.utils.leaves import DummyLeaf, make_leaf, make_normal_leaf


def build_sum(*, in_channels: int, out_channels: int, out_features: int, num_repetitions: int) -> Sum:
    inputs = make_normal_leaf(
        out_features=out_features,
        out_channels=in_channels,
        num_repetitions=num_repetitions,
    )
    return Sum(
        inputs=inputs,
        out_channels=out_channels,
        num_repetitions=num_repetitions,
    )


def build_elementwise_sum(
    *,
    in_channels: int,
    out_channels: int,
    out_features: int,
    num_repetitions: int,
) -> ElementwiseSum:
    scope = Scope(list(range(out_features)))
    inputs = [
        make_leaf(
            cls=DummyLeaf,
            out_channels=in_channels,
            scope=scope,
            num_repetitions=num_repetitions,
        ),
        make_leaf(
            cls=DummyLeaf,
            out_channels=in_channels,
            scope=scope,
            num_repetitions=num_repetitions,
        ),
    ]
    return ElementwiseSum(
        inputs=inputs,
        out_channels=out_channels,
        num_repetitions=num_repetitions,
    )


def build_product(*, in_channels: int, out_features: int, num_repetitions: int) -> Product:
    inputs = make_normal_leaf(
        out_features=out_features,
        out_channels=in_channels,
        num_repetitions=num_repetitions,
    )
    return Product(inputs=inputs)


def build_cat(*, out_channels: int, out_features: int, num_repetitions: int, dim: int) -> Cat:
    if dim == 1:
        scope_a = Scope(list(range(0, out_features)))
        scope_b = Scope(list(range(out_features, 2 * out_features)))
    else:
        scope_a = Scope(list(range(0, out_features)))
        scope_b = Scope(list(range(0, out_features)))

    inputs = [
        make_normal_leaf(scope_a, out_channels=out_channels, num_repetitions=num_repetitions),
        make_normal_leaf(scope_b, out_channels=out_channels, num_repetitions=num_repetitions),
    ]
    return Cat(inputs=inputs, dim=dim)


def build_split(
    *,
    out_channels: int,
    out_features: int,
    num_repetitions: int,
    num_splits: int,
    split_type,
):
    input_leaf = make_normal_leaf(
        out_features=out_features,
        out_channels=out_channels,
        num_repetitions=num_repetitions,
    )
    return split_type(inputs=input_leaf, num_splits=num_splits, dim=1)


def normalized_sum_weights(
    *,
    out_features: int,
    in_channels: int,
    out_channels: int,
    num_repetitions: int,
) -> torch.Tensor:
    weights = torch.ones((out_features, in_channels, out_channels, num_repetitions))
    return weights / weights.sum(dim=1, keepdim=True)
