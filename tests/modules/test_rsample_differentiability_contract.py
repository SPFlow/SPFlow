"""Differentiability contract tests for targeted rsample modules."""

from collections.abc import Callable

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.einsum import EinsumLayer, LinsumLayer
from spflow.modules.leaves import Normal
from spflow.modules.module import Module
from spflow.modules.ops import Cat, SplitByIndex, SplitConsecutive, SplitInterleaved
from spflow.modules.products import ElementwiseProduct
from spflow.modules.products.outer_product import OuterProduct
from spflow.modules.rat import Factorize
from spflow.modules.sums import RepetitionMixingLayer, Sum
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import DifferentiableSamplingContext
from tests.modules.einsum.layer_test_utils import make_einsum_two_inputs, make_linsum_two_inputs


def _random_simplex(*shape: int) -> torch.Tensor:
    probs = torch.rand(*shape)
    return probs / probs.sum(dim=-1, keepdim=True)


def _rsample_loss(samples: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(
        0.2,
        1.2,
        steps=samples.numel(),
        dtype=samples.dtype,
        device=samples.device,
    ).reshape_as(samples)
    return (samples * weights).mean() + (samples - 0.37).pow(2).mean()


def _assert_nonzero_finite_trainable_gradients(module: Module) -> None:
    trainable = 0
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        trainable += 1
        assert param.grad is not None, f"Missing gradient for parameter '{name}'"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for parameter '{name}'"
        assert param.grad.abs().sum().item() > 0.0, f"Zero gradient magnitude for parameter '{name}'"

    assert trainable > 0, "Expected at least one trainable parameter"


def _build_sum_case() -> tuple[Module, torch.Tensor, DifferentiableSamplingContext]:
    module = Sum(inputs=Normal(scope=Scope([0, 1, 2]), out_channels=3, num_repetitions=1), out_channels=2)
    batch_size = 6
    data = torch.full((batch_size, 3), torch.nan)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=_random_simplex(batch_size, module.out_shape.features, module.out_shape.channels),
        mask=torch.ones((batch_size, module.out_shape.features), dtype=torch.bool),
    )
    return module, data, sampling_ctx


def _build_repetition_mixing_case() -> tuple[Module, torch.Tensor, DifferentiableSamplingContext]:
    module = RepetitionMixingLayer(
        inputs=Normal(scope=Scope([0, 1, 2]), out_channels=2, num_repetitions=3),
        out_channels=2,
        num_repetitions=3,
    )
    batch_size = 5
    data = torch.full((batch_size, 3), torch.nan)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=_random_simplex(batch_size, module.out_shape.features, module.out_shape.channels),
        mask=torch.ones((batch_size, module.out_shape.features), dtype=torch.bool),
    )
    return module, data, sampling_ctx


def _build_split_consecutive_case() -> tuple[Module, torch.Tensor, DifferentiableSamplingContext]:
    module = SplitConsecutive(
        inputs=Normal(scope=Scope([0, 1, 2, 3, 4, 5]), out_channels=2, num_repetitions=1),
        num_splits=2,
        dim=1,
    )
    batch_size = 4
    data = torch.full((batch_size, 6), torch.nan)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=_random_simplex(batch_size, 3, module.out_shape.channels),
        mask=torch.ones((batch_size, 3), dtype=torch.bool),
    )
    return module, data, sampling_ctx


def _build_split_interleaved_case() -> tuple[Module, torch.Tensor, DifferentiableSamplingContext]:
    module = SplitInterleaved(
        inputs=Normal(scope=Scope([0, 1, 2, 3, 4, 5]), out_channels=2, num_repetitions=1),
        num_splits=2,
        dim=1,
    )
    batch_size = 4
    data = torch.full((batch_size, 6), torch.nan)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=_random_simplex(batch_size, 3, module.out_shape.channels),
        mask=torch.ones((batch_size, 3), dtype=torch.bool),
    )
    return module, data, sampling_ctx


def _build_split_by_index_case() -> tuple[Module, torch.Tensor, DifferentiableSamplingContext]:
    module = SplitByIndex(
        inputs=Normal(scope=Scope([0, 1, 2, 3]), out_channels=2, num_repetitions=1),
        indices=[[0, 2], [1, 3]],
    )
    batch_size = 4
    data = torch.full((batch_size, 4), torch.nan)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=_random_simplex(batch_size, 2, module.out_shape.channels),
        mask=torch.ones((batch_size, 2), dtype=torch.bool),
    )
    return module, data, sampling_ctx


def _build_cat_case() -> tuple[Module, torch.Tensor, DifferentiableSamplingContext]:
    scope = Scope([0, 1])
    left = Normal(
        scope=scope,
        out_channels=1,
        num_repetitions=1,
        loc=torch.tensor([[-1.0], [0.5]], dtype=torch.float32).view(2, 1, 1),
        scale=torch.full((2, 1, 1), 0.7, dtype=torch.float32),
    )
    right = Normal(
        scope=scope,
        out_channels=2,
        num_repetitions=1,
        loc=torch.tensor([[0.0, 2.0], [1.0, 3.0]], dtype=torch.float32).unsqueeze(-1),
        scale=torch.full((2, 2, 1), 0.5, dtype=torch.float32),
    )
    module = Cat(inputs=[left, right], dim=2)
    batch_size = 6
    data = torch.full((batch_size, 2), torch.nan)
    channel_probs = torch.zeros((batch_size, module.out_shape.features, module.out_shape.channels))
    for sample_idx, channel_idx in enumerate(torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)):
        channel_probs[sample_idx, :, channel_idx] = 1.0
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=channel_probs,
        mask=torch.ones((batch_size, module.out_shape.features), dtype=torch.bool),
    )
    return module, data, sampling_ctx


def _build_factorize_case() -> tuple[Module, torch.Tensor, DifferentiableSamplingContext]:
    module = Factorize(
        inputs=[Normal(scope=Scope([0, 1, 2, 3]), out_channels=2, num_repetitions=1)],
        depth=1,
        num_repetitions=1,
    )
    batch_size = 4
    data = torch.full((batch_size, 4), torch.nan)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=_random_simplex(batch_size, module.out_shape.features, module.out_shape.channels),
        mask=torch.ones((batch_size, module.out_shape.features), dtype=torch.bool),
    )
    return module, data, sampling_ctx


def _build_elementwise_product_case() -> tuple[Module, torch.Tensor, DifferentiableSamplingContext]:
    module = ElementwiseProduct(
        inputs=[
            Normal(scope=Scope([0, 1]), out_channels=2, num_repetitions=1),
            Normal(scope=Scope([2, 3]), out_channels=2, num_repetitions=1),
        ]
    )
    batch_size = 5
    data = torch.full((batch_size, 4), torch.nan)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=_random_simplex(batch_size, module.out_shape.features, module.out_shape.channels),
        mask=torch.ones((batch_size, module.out_shape.features), dtype=torch.bool),
    )
    return module, data, sampling_ctx


def _build_outer_product_case() -> tuple[Module, torch.Tensor, DifferentiableSamplingContext]:
    module = OuterProduct(
        inputs=[
            Normal(scope=Scope([0, 1]), out_channels=2, num_repetitions=1),
            Normal(scope=Scope([2, 3]), out_channels=2, num_repetitions=1),
        ]
    )
    batch_size = 5
    data = torch.full((batch_size, 4), torch.nan)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=_random_simplex(batch_size, module.out_shape.features, module.out_shape.channels),
        mask=torch.ones((batch_size, module.out_shape.features), dtype=torch.bool),
    )
    return module, data, sampling_ctx


def _build_linsum_case() -> tuple[Module, torch.Tensor, DifferentiableSamplingContext]:
    module: LinsumLayer = make_linsum_two_inputs(
        in_channels=2, out_channels=3, in_features=2, num_repetitions=2
    )
    batch_size = 5
    data = torch.full((batch_size, 4), torch.nan)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=_random_simplex(batch_size, module.out_shape.features, module.out_shape.channels),
        mask=torch.ones((batch_size, module.out_shape.features), dtype=torch.bool),
        repetition_probs=_random_simplex(batch_size, module.out_shape.repetitions),
    )
    return module, data, sampling_ctx


def _build_einsum_case() -> tuple[Module, torch.Tensor, DifferentiableSamplingContext]:
    module: EinsumLayer = make_einsum_two_inputs(
        in_channels=2, out_channels=3, in_features=2, num_repetitions=2
    )
    batch_size = 5
    data = torch.full((batch_size, 4), torch.nan)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=_random_simplex(batch_size, module.out_shape.features, module.out_shape.channels),
        mask=torch.ones((batch_size, module.out_shape.features), dtype=torch.bool),
        repetition_probs=_random_simplex(batch_size, module.out_shape.repetitions),
    )
    return module, data, sampling_ctx


def _build_leaf_case() -> tuple[Module, torch.Tensor, DifferentiableSamplingContext]:
    module = Normal(scope=Scope([0, 1]), out_channels=3, num_repetitions=1)
    batch_size = 6
    data = torch.full((batch_size, 2), torch.nan)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=_random_simplex(batch_size, module.out_shape.features, module.out_shape.channels),
        mask=torch.ones((batch_size, module.out_shape.features), dtype=torch.bool),
    )
    return module, data, sampling_ctx


CaseBuilder = Callable[[], tuple[Module, torch.Tensor, DifferentiableSamplingContext]]


@pytest.mark.parametrize(
    "build_case",
    [
        pytest.param(_build_sum_case, id="sum"),
        pytest.param(_build_repetition_mixing_case, id="repetition-mixing"),
        pytest.param(_build_split_consecutive_case, id="split-consecutive"),
        pytest.param(_build_split_interleaved_case, id="split-interleaved"),
        pytest.param(_build_split_by_index_case, id="split-by-index"),
        pytest.param(_build_cat_case, id="cat"),
        pytest.param(_build_factorize_case, id="factorize"),
        pytest.param(_build_elementwise_product_case, id="elementwise-product"),
        pytest.param(_build_outer_product_case, id="outer-product"),
        pytest.param(_build_linsum_case, id="linsum-layer"),
        pytest.param(_build_einsum_case, id="einsum-layer"),
        pytest.param(_build_leaf_case, id="normal-leaf"),
    ],
)
def test_rsample_gradients_are_nonzero_and_finite_for_targeted_modules(build_case: CaseBuilder):
    torch.manual_seed(7)
    module, data, sampling_ctx = build_case()
    sampling_ctx.diff_method = "gumbel"
    sampling_ctx.hard = False

    samples = module._rsample(
        data=data,
        sampling_ctx=sampling_ctx,
        cache=Cache(),
        is_mpe=False,
    )
    if sampling_ctx.sample_accum is not None and sampling_ctx.sample_mass is not None:
        samples = sampling_ctx.finalize_with_evidence(samples)
    assert torch.isfinite(samples).all()

    loss = _rsample_loss(samples)
    loss.backward()

    _assert_nonzero_finite_trainable_gradients(module)
