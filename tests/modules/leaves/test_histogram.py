from __future__ import annotations

from itertools import product

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.leaves import Histogram


def _make_histogram(*, bin_edges: torch.Tensor, probs: torch.Tensor) -> Histogram:
    scope = Scope([0])
    return Histogram(scope=scope, bin_edges=bin_edges, probs=probs)


def test_log_likelihood_matches_manual_bin_density():
    bin_edges = torch.tensor([0.0, 1.0, 3.0])
    probs = torch.tensor([[[[0.2, 0.8]]]])  # (F=1, C=1, R=1, B=2)
    leaf = _make_histogram(bin_edges=bin_edges, probs=probs)

    data = torch.tensor([[0.5], [2.5]])
    ll = leaf.log_likelihood(data).squeeze(-1).squeeze(-1).squeeze(1)  # (N,)

    widths = torch.tensor([1.0, 2.0])
    densities = torch.tensor([0.2, 0.8]) / widths
    expected = torch.log(torch.tensor([densities[0], densities[1]]))
    torch.testing.assert_close(ll, expected, rtol=1e-6, atol=1e-6)


def test_nan_inputs_marginalize_to_zero_log_contribution():
    bin_edges = torch.tensor([0.0, 1.0, 2.0])
    probs = torch.tensor([[[[0.5, 0.5]]]])
    leaf = _make_histogram(bin_edges=bin_edges, probs=probs)

    data = torch.tensor([[float("nan")]])
    ll = leaf.log_likelihood(data)
    assert ll.shape == (1, 1, 1, 1)
    torch.testing.assert_close(ll, torch.zeros_like(ll))


@pytest.mark.parametrize("x", [-0.1, 2.0])
def test_out_of_support_is_neg_inf(x: float):
    bin_edges = torch.tensor([0.0, 1.0, 2.0])
    probs = torch.tensor([[[[0.5, 0.5]]]])
    leaf = _make_histogram(bin_edges=bin_edges, probs=probs)

    data = torch.tensor([[x]])
    ll = leaf.log_likelihood(data)
    assert torch.isneginf(ll).all()


def test_sampling_values_in_support():
    bin_edges = torch.tensor([0.0, 1.0, 2.0, 3.0])
    probs = torch.tensor([[[[0.1, 0.7, 0.2]]]])
    leaf = _make_histogram(bin_edges=bin_edges, probs=probs)

    samples = leaf.sample(num_samples=512)
    assert samples.shape == (512, 1)
    assert torch.isfinite(samples).all()
    assert (samples >= bin_edges[0]).all()
    assert (samples < bin_edges[-1]).all()


def test_mpe_returns_mode_value():
    bin_edges = torch.tensor([0.0, 1.0, 3.0])
    probs = torch.tensor([[[[0.2, 0.8]]]])
    leaf = _make_histogram(bin_edges=bin_edges, probs=probs)

    mpe = leaf.sample(num_samples=8, is_mpe=True)
    assert mpe.shape == (8, 1)
    expected_mode = torch.tensor([(1.0 + 3.0) / 2.0])
    torch.testing.assert_close(mpe[:, 0], expected_mode.expand_as(mpe[:, 0]))


@pytest.mark.parametrize("num_repetitions", [1, 3])
@pytest.mark.parametrize("out_channels", [1, 4])
def test_log_likelihood_output_shape(out_channels: int, num_repetitions: int):
    bin_edges = torch.tensor([0.0, 1.0, 2.0])
    probs = torch.rand(1, out_channels, num_repetitions, 2)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    leaf = Histogram(
        scope=Scope([0]),
        bin_edges=bin_edges,
        probs=probs,
        out_channels=out_channels,
        num_repetitions=num_repetitions,
    )

    x = torch.tensor([[0.25], [1.25]])
    ll = leaf.log_likelihood(x)
    assert ll.shape == (2, 1, out_channels, num_repetitions)
    assert torch.isfinite(ll).all()


def test_mle_update_increases_training_ll():
    torch.manual_seed(0)

    bin_edges = torch.tensor([0.0, 1.0, 2.0, 3.0])
    true_probs = torch.tensor([0.1, 0.7, 0.2])

    n = 4000
    bin_idx = torch.distributions.Categorical(probs=true_probs).sample((n,))
    left = bin_edges[bin_idx]
    right = bin_edges[bin_idx + 1]
    x = (left + torch.rand(n) * (right - left)).unsqueeze(1)

    init_probs = torch.full((1, 1, 1, 3), 1.0 / 3.0)
    leaf = _make_histogram(bin_edges=bin_edges, probs=init_probs)

    ll_before = leaf.log_likelihood(x).mean().item()
    leaf.maximum_likelihood_estimation(x, nan_strategy="ignore")
    ll_after = leaf.log_likelihood(x).mean().item()

    assert ll_after > ll_before + 0.05
