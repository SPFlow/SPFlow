"""Shared helpers for Einsum/Linsum test modules."""

from itertools import product
from typing import Callable

import torch

from spflow.modules.module import Module
from spflow.meta import Scope
from spflow.modules.einsum import EinsumLayer, LinsumLayer
from spflow.modules.ops.split import Split
from spflow.modules.ops.split_consecutive import SplitConsecutive
from spflow.modules.ops.split_interleaved import SplitInterleaved
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.utils.leaves import DummyLeaf, make_leaf, make_normal_data, make_normal_leaf

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


def assert_log_weights_consistent(make_single_input: Callable[..., Module]) -> None:
    module = make_single_input(3, 4, 6, 2)
    expected = torch.log(module.weights)
    actual = module.log_weights
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)


def assert_set_weights_round_trip(
    make_single_input: Callable[..., Module], *, normalize_dims: tuple[int, ...]
) -> None:
    module = make_single_input(2, 3, 4, 1)
    new_weights = torch.rand(module.weights_shape) + 1e-8
    new_weights = new_weights / new_weights.sum(dim=normalize_dims, keepdim=True)
    module.weights = new_weights
    torch.testing.assert_close(module.weights, new_weights, rtol=1e-5, atol=1e-5)


def assert_marginalize_partial_single_input(make_single_input: Callable[..., Module]) -> None:
    module = make_single_input(2, 3, 4, 1)
    marg_module = module.marginalize([0])
    assert marg_module is not None


def assert_marginalize_full_single_input(make_single_input: Callable[..., Module]) -> None:
    module = make_single_input(2, 3, 4, 1)
    all_vars = list(module.scope.query)
    marg_module = module.marginalize(all_vars)
    assert marg_module is None


def assert_extra_repr_contains_weights_shape(make_single_input: Callable[..., Module]) -> None:
    module = make_single_input(2, 3, 4, 1)
    repr_str = module.extra_repr()
    assert "weights=" in repr_str
    assert str(module.weights_shape) in repr_str


def assert_split_input_not_wrapped(layer_cls: type[Module]) -> None:
    leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
    split_mode = SplitConsecutive(leaf)
    module = layer_cls(inputs=split_mode, out_channels=3)

    assert module.inputs is split_mode
    assert isinstance(module.inputs, Split)
    assert not isinstance(module.inputs.inputs, Split)


def assert_split_input_produces_same_output(layer_cls: type[Module]) -> None:
    leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
    data = make_normal_data(out_features=4)
    split_mode = SplitConsecutive(leaf)
    wrapped = layer_cls(inputs=leaf, out_channels=3)
    direct = layer_cls(inputs=split_mode, out_channels=3)

    direct.logits.data = wrapped.logits.data.clone()

    lls_wrapped = wrapped.log_likelihood(data)
    lls_direct = direct.log_likelihood(data)
    torch.testing.assert_close(lls_wrapped, lls_direct, rtol=1e-5, atol=1e-8)


def assert_split_sampling_works(layer_cls: type[Module]) -> None:
    leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
    split_mode = SplitConsecutive(leaf)
    module = layer_cls(inputs=split_mode, out_channels=3)

    num_samples = 20
    data = torch.full((num_samples, 4), torch.nan)
    channel_index = torch.zeros((num_samples, 2), dtype=torch.long)
    mask = torch.ones((num_samples, 2), dtype=torch.bool)
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)
    samples = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

    assert samples.shape == (num_samples, 4)
    assert torch.isfinite(samples).all()


def assert_split_alternate_input_works(layer_cls: type[Module]) -> None:
    leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=1)
    split_alt = SplitInterleaved(leaf)
    module = layer_cls(inputs=split_alt, out_channels=3)

    assert module.inputs is split_alt
    assert isinstance(module.inputs, Split)

    data = make_normal_data(out_features=4)
    lls = module.log_likelihood(data)
    assert torch.isfinite(lls).all()

    num_samples = 20
    sample_data = torch.full((num_samples, 4), torch.nan)
    channel_index = torch.zeros((num_samples, 2), dtype=torch.long)
    mask = torch.ones((num_samples, 2), dtype=torch.bool)
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)
    samples = module._sample(data=sample_data, sampling_ctx=sampling_ctx, cache=Cache())
    assert samples.shape == (num_samples, 4)
    assert torch.isfinite(samples).all()


def assert_split_alternate_input_differentiable_sampling(layer_cls: type[Module]) -> None:
    leaf = make_normal_leaf(out_features=4, out_channels=2, num_repetitions=2)
    split_alt = SplitInterleaved(leaf)
    module = layer_cls(inputs=split_alt, out_channels=3, num_repetitions=2)

    num_samples = 20
    sample_data = torch.full((num_samples, 4), torch.nan)
    channel_index = torch.randint(0, module.out_shape.channels, (num_samples, module.out_shape.features))
    repetition_index = torch.randint(0, module.out_shape.repetitions, (num_samples,))
    sampling_ctx = SamplingContext(
        channel_index=to_one_hot(channel_index, dim=-1, dim_size=module.out_shape.channels),
        repetition_index=to_one_hot(repetition_index, dim=-1, dim_size=module.out_shape.repetitions),
        mask=torch.ones((num_samples, module.out_shape.features), dtype=torch.bool),
        is_differentiable=True,
    )

    samples = module._sample(data=sample_data, sampling_ctx=sampling_ctx, cache=Cache())
    assert samples.shape == (num_samples, 4)
    assert torch.isfinite(samples).all()
