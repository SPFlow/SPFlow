import pytest
import torch

from spflow.exceptions import UnsupportedOperationError
from spflow.modules import leaves
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.modules.leaves.leaf_contract_data import (
    DIFFERENTIABLE_EQ_LEAF_CLS_VALUES,
    DIFF_DISTRIBUTION_LEAF_CASES,
    DIFF_SAMPLING_SUPPORTED_LEAF_CLS_VALUES,
    LEAF_SAMPLE_PARAMS,
    UNSUPPORTED_DIFF_DISTRIBUTION_LEAF_CASES,
)
from tests.utils.leaves import make_leaf
from tests.utils.sampling_context_helpers import patch_simple_as_categorical_one_hot

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "leaf_cls, out_features, out_channels, num_reps, is_mpe",
    LEAF_SAMPLE_PARAMS,
)
def test_sample(leaf_cls, out_features: int, out_channels: int, num_reps, is_mpe: bool):
    # Uniform has no unique finite mode; MPE semantics are intentionally undefined there.
    if leaf_cls == leaves.Uniform and is_mpe:
        return

    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    n_samples = 10
    data = torch.full((n_samples, out_features), torch.nan)
    channel_index = torch.randint(low=0, high=out_channels, size=(n_samples, out_features))
    mask = torch.full((n_samples, out_features), True, dtype=torch.bool)
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx = SamplingContext(
        channel_index=channel_index,
        mask=mask,
        repetition_index=repetition_index,
        is_mpe=is_mpe,
    )

    samples = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

    assert samples.shape == (n_samples, out_features)
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize(
    "leaf_cls, out_features, out_channels, num_reps, is_mpe",
    LEAF_SAMPLE_PARAMS,
)
def test_diff_sampling_eq_non_diff(
    leaf_cls, out_features: int, out_channels: int, num_reps, is_mpe: bool, monkeypatch
):
    # Uniform has no unique finite mode; MPE semantics are intentionally undefined there.
    if leaf_cls == leaves.Uniform and is_mpe:
        return

    module = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    n_samples = 10
    data = torch.full((n_samples, out_features), torch.nan)
    channel_index = torch.randint(low=0, high=out_channels, size=(n_samples, out_features))
    mask = torch.full((n_samples, out_features), True, dtype=torch.bool)
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx = SamplingContext(
        channel_index=channel_index,
        mask=mask,
        repetition_index=repetition_index,
        is_mpe=is_mpe,
        is_differentiable=False,
    )

    # Reuse the same seed for both paths so differences reflect routing, not RNG drift.
    torch.manual_seed(1337)
    samples = module._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

    sampling_ctx_diff = SamplingContext(
        channel_index=to_one_hot(channel_index, dim=-1, dim_size=out_channels),
        mask=mask,
        repetition_index=to_one_hot(repetition_index, dim=-1, dim_size=num_reps),
        is_mpe=is_mpe,
        is_differentiable=True,
    )

    if leaf_cls not in DIFF_SAMPLING_SUPPORTED_LEAF_CLS_VALUES:
        # Unsupported leaves should fail fast rather than silently falling back.
        with pytest.raises(NotImplementedError):
            module._sample(data=data, sampling_ctx=sampling_ctx_diff, cache=Cache())
        return

    patch_simple_as_categorical_one_hot(monkeypatch)

    # Use the same RNG state as the non-differentiable branch for parity checks.
    torch.manual_seed(1337)
    samples_diff = module._sample(data=data, sampling_ctx=sampling_ctx_diff, cache=Cache())

    assert samples_diff.shape == (n_samples, out_features)
    assert torch.isfinite(samples).all()
    assert torch.isfinite(samples_diff).all()

    if leaf_cls in DIFFERENTIABLE_EQ_LEAF_CLS_VALUES:
        torch.testing.assert_close(samples, samples_diff, rtol=1e-6, atol=1e-6)
    # Sampling should not mutate routing metadata owned by the caller.
    torch.testing.assert_close(
        sampling_ctx_diff.channel_index,
        to_one_hot(sampling_ctx.channel_index, dim=-1, dim_size=out_channels),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(("leaf_ctor", "expects_grad"), DIFF_DISTRIBUTION_LEAF_CASES)
def test_distribution_rsample_produces_finite_values_and_gradients(leaf_ctor, expects_grad: bool):
    torch.manual_seed(0)
    leaf = leaf_ctor()
    dist = leaf.distribution(with_differentiable_sampling=True)
    samples = dist.rsample((7,))

    assert torch.isfinite(samples).all()

    if not expects_grad:
        # Some rsample adapters are value-only checks and are not expected to backprop.
        return

    trainable_params = [p for p in leaf.parameters() if p.requires_grad]
    assert trainable_params, f"{leaf.__class__.__name__} expected to have trainable parameters."

    loss = samples.mean()
    assert loss.requires_grad, (
        f"{leaf.__class__.__name__}.distribution(...).rsample() must be differentiable."
    )
    loss.backward()

    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable_params), (
        f"{leaf.__class__.__name__} did not receive finite gradients from rsample()."
    )


@pytest.mark.parametrize("leaf_ctor", UNSUPPORTED_DIFF_DISTRIBUTION_LEAF_CASES)
def test_unsupported_leaves_raise_for_differentiable_distribution(leaf_ctor):
    leaf = leaf_ctor()

    with pytest.raises((NotImplementedError, UnsupportedOperationError)):
        leaf.distribution(with_differentiable_sampling=True)
