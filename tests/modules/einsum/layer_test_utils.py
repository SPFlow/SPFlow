"""Shared helpers for Einsum/Linsum test modules."""

from itertools import product

from spflow.meta import Scope
from spflow.modules.einsum import EinsumLayer, LinsumLayer
from tests.utils.leaves import DummyLeaf, make_leaf, make_normal_leaf

in_channels_values = [1, 3]
out_channels_values = [1, 4]
in_features_values = [2, 4, 8]
num_repetitions_values = [1, 2]

params = list(product(in_channels_values, out_channels_values, in_features_values, num_repetitions_values))


def make_einsum_single_input(
    in_channels: int, out_channels: int, in_features: int, num_repetitions: int
) -> EinsumLayer:
    inputs = make_normal_leaf(
        out_features=in_features,
        out_channels=in_channels,
        num_repetitions=num_repetitions,
    )
    return EinsumLayer(inputs=inputs, out_channels=out_channels, num_repetitions=num_repetitions)


def make_einsum_two_inputs(
    in_channels: int,
    out_channels: int,
    in_features: int,
    num_repetitions: int,
    left_channels: int | None = None,
    right_channels: int | None = None,
) -> EinsumLayer:
    left_ch = left_channels if left_channels is not None else in_channels
    right_ch = right_channels if right_channels is not None else in_channels
    left_scope = Scope(list(range(0, in_features)))
    right_scope = Scope(list(range(in_features, in_features * 2)))
    left_input = make_leaf(
        cls=DummyLeaf,
        out_channels=left_ch,
        scope=left_scope,
        num_repetitions=num_repetitions,
    )
    right_input = make_leaf(
        cls=DummyLeaf,
        out_channels=right_ch,
        scope=right_scope,
        num_repetitions=num_repetitions,
    )
    return EinsumLayer(
        inputs=[left_input, right_input], out_channels=out_channels, num_repetitions=num_repetitions
    )


def make_linsum_single_input(
    in_channels: int, out_channels: int, in_features: int, num_repetitions: int
) -> LinsumLayer:
    inputs = make_normal_leaf(
        out_features=in_features,
        out_channels=in_channels,
        num_repetitions=num_repetitions,
    )
    return LinsumLayer(inputs=inputs, out_channels=out_channels, num_repetitions=num_repetitions)


def make_linsum_two_inputs(
    in_channels: int, out_channels: int, in_features: int, num_repetitions: int
) -> LinsumLayer:
    left_scope = Scope(list(range(0, in_features)))
    right_scope = Scope(list(range(in_features, in_features * 2)))
    left_input = make_leaf(
        cls=DummyLeaf,
        out_channels=in_channels,
        scope=left_scope,
        num_repetitions=num_repetitions,
    )
    right_input = make_leaf(
        cls=DummyLeaf,
        out_channels=in_channels,
        scope=right_scope,
        num_repetitions=num_repetitions,
    )
    return LinsumLayer(
        inputs=[left_input, right_input], out_channels=out_channels, num_repetitions=num_repetitions
    )
