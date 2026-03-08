"""Tests for EinsumLayer's current log-likelihood implementation details."""

from __future__ import annotations

import torch

from spflow.modules.einsum import EinsumLayer
from tests.utils.leaves import make_normal_data, make_normal_leaf


def _make_single_input_module() -> EinsumLayer:
    torch.manual_seed(1337)
    leaf = make_normal_leaf(
        out_features=8,
        out_channels=16,
        num_repetitions=2,
    )
    return EinsumLayer(
        inputs=leaf,
        out_channels=8,
        num_repetitions=2,
    )


def test_log_likelihood_uses_cached_weights_only_in_eval_no_grad() -> None:
    module = _make_single_input_module().eval()

    with torch.no_grad():
        first = module.weights
        second = module.weights

    assert first is second

    module.train()
    train_first = module.weights
    train_second = module.weights
    assert train_first is not train_second


def test_log_likelihood_weight_cache_invalidates_after_parameter_update() -> None:
    module = _make_single_input_module().eval()

    with torch.no_grad():
        cached_before = module.weights
        module.logits.add_(0.1)
        cached_after = module.weights

    assert cached_before is not cached_after


def test_log_likelihood_stays_finite_for_large_channel_products() -> None:
    torch.manual_seed(1337)
    leaf = make_normal_leaf(
        out_features=16,
        out_channels=32,
        num_repetitions=2,
    )
    module = EinsumLayer(inputs=leaf, out_channels=8, num_repetitions=2)
    data = make_normal_data(num_samples=6, out_features=16)

    lls = module.log_likelihood(data)

    assert lls.shape == (6, 8, 8, 2)
    assert torch.isfinite(lls).all()
