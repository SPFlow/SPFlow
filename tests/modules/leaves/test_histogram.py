from __future__ import annotations

from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError, InvalidParameterError
from spflow.meta import Scope
from spflow.modules.leaves import Histogram
from spflow.modules.leaves.histogram import HistogramDist
from spflow.utils.histogram import _get_outer_edges_torch, get_bin_edges_torch


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


def test_get_outer_edges_range_validation_and_special_cases():
    with pytest.raises(InvalidParameterError):
        _get_outer_edges_torch(torch.tensor([0.0, 1.0]), range_bounds=(2.0, 1.0))

    with pytest.raises(InvalidParameterError):
        _get_outer_edges_torch(torch.tensor([0.0, 1.0]), range_bounds=(0.0, float("inf")))

    first, last = _get_outer_edges_torch(torch.tensor([]))
    assert (first, last) == (0.0, 1.0)

    with pytest.raises(InvalidParameterError):
        _get_outer_edges_torch(torch.tensor([0.0, float("nan")]))

    first2, last2 = _get_outer_edges_torch(torch.tensor([3.0, 3.0]))
    assert (first2, last2) == (2.5, 3.5)


def test_get_bin_edges_range_filtering_and_zero_width_fallback():
    x = torch.tensor([0.0, 0.0, 0.0])
    edges, (first, last, n_bins) = get_bin_edges_torch(x)
    assert n_bins == 1
    assert first < last
    assert edges.shape[0] == 2

    y = torch.tensor([-10.0, 10.0])
    edges2, (_first2, _last2, n_bins2) = get_bin_edges_torch(y, range_bounds=(0.0, 1.0))
    assert n_bins2 == 1
    torch.testing.assert_close(edges2, torch.tensor([0.0, 1.0]))


def test_histogram_dist_validation_and_accessors():
    with pytest.raises(InvalidParameterError):
        HistogramDist(bin_edges=torch.tensor([[0.0, 1.0]]), logits=torch.zeros(1, 1, 1, 1))
    with pytest.raises(InvalidParameterError):
        HistogramDist(bin_edges=torch.tensor([0.0]), logits=torch.zeros(1, 1, 1, 1))
    with pytest.raises(InvalidParameterError):
        HistogramDist(bin_edges=torch.tensor([0.0, float("inf")]), logits=torch.zeros(1, 1, 1, 1))
    with pytest.raises(InvalidParameterError):
        HistogramDist(bin_edges=torch.tensor([0.0, 0.0]), logits=torch.zeros(1, 1, 1, 1))
    with pytest.raises(InvalidParameterError):
        HistogramDist(bin_edges=torch.tensor([0.0, 1.0]), logits=torch.zeros(1, 1, 1))

    dist = HistogramDist(bin_edges=torch.tensor([0.0, 1.0, 2.0]), logits=torch.zeros(1, 2, 3, 2))
    assert dist.nbins == 2
    torch.testing.assert_close(dist.bin_edges, torch.tensor([0.0, 1.0, 2.0]))


def test_histogram_dist_align_x_and_log_prob_errors():
    dist = HistogramDist(bin_edges=torch.tensor([0.0, 1.0, 2.0]), logits=torch.zeros(1, 1, 1, 2))

    assert dist._align_x(torch.zeros(2, 1)).shape == (2, 1, 1, 1)
    assert dist._align_x(torch.zeros(2, 1, 1)).shape == (2, 1, 1, 1)
    assert dist._align_x(torch.zeros(2, 1, 1, 1)).shape == (2, 1, 1, 1)

    with pytest.raises(InvalidParameterError):
        dist._align_x(torch.zeros(2, 1, 1, 1, 1))

    with pytest.raises(InvalidParameterError):
        dist.log_prob(torch.zeros(4, 2))


def test_histogram_dist_sample_with_torch_size_shape():
    dist = HistogramDist(bin_edges=torch.tensor([0.0, 1.0, 2.0]), logits=torch.zeros(1, 1, 1, 2))
    samples = dist.sample(torch.Size([5]))
    assert samples.shape == (5, 1, 1, 1)

    samples_default = dist.sample(torch.Size())
    assert samples_default.shape == (1, 1, 1, 1)


def test_histogram_constructor_validation_errors():
    with pytest.raises(InvalidParameterCombinationError):
        Histogram(
            scope=Scope([0]),
            bin_edges=torch.tensor([0.0, 1.0]),
            probs=torch.tensor([[[[1.0]]]]),
            logits=torch.tensor([[[[0.0]]]]),
        )

    with pytest.raises(InvalidParameterError):
        Histogram(scope=Scope([0, 1]), bin_edges=torch.tensor([0.0, 1.0]))

    with pytest.raises(InvalidParameterError):
        Histogram(scope=Scope([0]), bin_edges=torch.tensor([[0.0, 1.0]]))
    with pytest.raises(InvalidParameterError):
        Histogram(scope=Scope([0]), bin_edges=torch.tensor([0.0]))
    with pytest.raises(InvalidParameterError):
        Histogram(scope=Scope([0]), bin_edges=torch.tensor([0.0, float("nan")]))
    with pytest.raises(InvalidParameterError):
        Histogram(scope=Scope([0]), bin_edges=torch.tensor([0.0, 0.0]))

    with pytest.raises(InvalidParameterError):
        Histogram(
            scope=Scope([0]),
            bin_edges=torch.tensor([0.0, 1.0, 2.0]),
            probs=torch.tensor([[[[1.0]]]]),
        )


def test_histogram_parameter_properties_and_setters():
    leaf = Histogram(
        scope=Scope([0]),
        bin_edges=torch.tensor([0.0, 1.0, 2.0]),
        probs=torch.tensor([[[[0.4, 0.6]]]]),
    )

    assert leaf._torch_distribution_class is None
    assert isinstance(leaf.logits, torch.Tensor)
    params = leaf.params()
    assert "logits" in params
    assert params["logits"].shape == leaf.logits.shape

    new_logits = torch.tensor([[[[2.0, -1.0]]]])
    leaf.logits = new_logits
    torch.testing.assert_close(leaf.logits, new_logits)

    with pytest.raises(InvalidParameterError):
        leaf.probs = torch.tensor([[[[float("inf"), 1.0]]]])
    with pytest.raises(InvalidParameterError):
        leaf.probs = torch.tensor([[[[-1.0, 2.0]]]])


def test_histogram_compute_parameter_estimates_validation_errors():
    leaf = Histogram(
        scope=Scope([0]),
        bin_edges=torch.tensor([0.0, 1.0, 2.0]),
        probs=torch.tensor([[[[0.5, 0.5]]]]),
    )
    weights = torch.ones(3, 1, 1, 1)

    bad_shape_data = torch.zeros(3, 2, 1, 1)
    with pytest.raises(InvalidParameterError):
        leaf._compute_parameter_estimates(data=bad_shape_data, weights=weights, bias_correction=False)

    out_of_support_data = torch.tensor([[[[-1.0]]], [[[0.5]]], [[[1.5]]]])
    with pytest.raises(InvalidParameterError):
        leaf._compute_parameter_estimates(data=out_of_support_data, weights=weights, bias_correction=False)


def test_histogram_marginalize_branches():
    leaf = Histogram(
        scope=Scope([0]),
        bin_edges=torch.tensor([0.0, 1.0, 2.0]),
        probs=torch.tensor([[[[0.5, 0.5]]]]),
    )

    leaf.parameter_fn = lambda evidence: {"logits": leaf.logits}
    with pytest.raises(RuntimeError):
        leaf.marginalize([99])

    leaf.parameter_fn = None
    assert leaf.marginalize([0]) is None

    kept = leaf.marginalize([1])
    assert isinstance(kept, Histogram)
