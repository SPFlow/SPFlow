from __future__ import annotations

from itertools import product

import pytest
import torch
from einops import rearrange, repeat

from spflow.exceptions import InvalidParameterCombinationError, InvalidParameterError
from spflow.meta import Scope
from spflow.modules.leaves import Histogram
from spflow.modules.leaves.histogram import HistogramDist
from spflow.utils.histogram import _get_outer_edges_torch, get_bin_edges_torch


def _make_histogram(*, bin_edges: torch.Tensor, probs: torch.Tensor) -> Histogram:
    scope = Scope([0])
    return Histogram(scope=scope, bin_edges=bin_edges, probs=probs)


def _reference_log_prob(dist: HistogramDist, x: torch.Tensor) -> torch.Tensor:
    x = dist._align_x(x)
    n_samples = x.shape[0]
    target_shape = (n_samples, *dist._logits.shape[:-1])
    x_broadcast = torch.broadcast_to(x, target_shape).contiguous()

    edges = dist.bin_edges.to(device=x_broadcast.device, dtype=x_broadcast.dtype)
    bin_idx = torch.bucketize(x_broadcast, edges, right=True) - 1
    in_support = torch.isfinite(x_broadcast) & (x_broadcast >= edges[0]) & (x_broadcast < edges[-1])

    densities = dist._bin_densities.to(device=x_broadcast.device, dtype=x_broadcast.dtype)
    densities = repeat(rearrange(densities, "f c r b -> 1 f c r b"), "1 f c r b -> n f c r b", n=n_samples)
    gathered = rearrange(
        densities.gather(-1, rearrange(bin_idx.clamp(0, dist.nbins - 1), "n f c r -> n f c r 1")),
        "n f c r 1 -> n f c r",
    )

    min_density = dist._min_prob / dist._bin_widths.max().to(device=gathered.device, dtype=gathered.dtype)
    log_p = torch.log(gathered.clamp_min(min_density))
    return torch.where(in_support, log_p, x_broadcast.new_full((), float("-inf")))


def test_log_likelihood_matches_manual_bin_density():
    bin_edges = torch.tensor([0.0, 1.0, 3.0])
    # Keep axis layout explicit to guard feature/channel/repetition ordering.
    probs = torch.tensor([[[[0.2, 0.8]]]])
    leaf = _make_histogram(bin_edges=bin_edges, probs=probs)

    data = torch.tensor([[0.5], [2.5]])
    # Collapse singleton leaf axes so we can compare per-sample manual densities.
    ll = leaf.log_likelihood(data).squeeze(-1).squeeze(-1).squeeze(1)

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


def test_differentiable_distribution_rsample_in_support_and_produces_grads():
    torch.manual_seed(0)
    leaf = Histogram(
        scope=Scope([0]), bin_edges=torch.tensor([0.0, 1.0, 3.0]), out_channels=2, num_repetitions=1
    )
    dist = leaf.distribution(with_differentiable_sampling=True)
    samples = dist.rsample((7,))

    assert samples.shape == (7, 1, 2, 1)
    assert torch.isfinite(samples).all()
    assert (samples >= 0.0).all()
    assert (samples < 3.0).all()

    samples.mean().backward()
    assert leaf.logits.grad is not None
    assert torch.isfinite(leaf.logits.grad).all()


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


@pytest.mark.parametrize("layout", ["nf", "nfc", "nfcr"])
def test_histogram_log_prob_matches_reference(layout: str):
    torch.manual_seed(0)
    bin_edges = torch.tensor([-1.5, -0.5, 0.25, 1.5, 3.0], dtype=torch.float32)
    logits = torch.randn(2, 3, 2, 4, dtype=torch.float32)
    dist = HistogramDist(bin_edges=bin_edges, logits=logits, min_prob=1e-12)

    if layout == "nf":
        x = torch.tensor(
            [
                [-1.75, 0.00],
                [-0.25, 0.40],
                [0.10, 2.50],
                [float("nan"), 1.40],
                [2.90, 3.10],
            ],
            dtype=torch.float32,
        )
    elif layout == "nfc":
        x = torch.tensor(
            [
                [[-0.25, -0.10, 0.00], [0.40, 0.60, 0.80]],
                [[0.20, 0.35, 0.50], [1.20, 1.80, 2.40]],
                [[float("nan"), 0.10, 0.15], [2.75, 3.05, -2.00]],
            ],
            dtype=torch.float32,
        )
    else:
        x = torch.tensor(
            [
                [
                    [[-0.25, -0.10], [0.00, 0.15], [0.20, 0.35]],
                    [[0.40, 0.60], [0.80, 1.00], [1.20, 1.40]],
                ],
                [
                    [[2.20, 2.40], [2.60, 2.80], [3.10, -2.00]],
                    [[float("nan"), 0.05], [0.10, 0.15], [0.20, 0.25]],
                ],
            ],
            dtype=torch.float32,
        )

    actual = dist.log_prob(x)
    expected = _reference_log_prob(dist, x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_histogram_log_prob_matches_reference_gradients():
    torch.manual_seed(1)
    bin_edges = torch.tensor([-2.0, -0.5, 0.5, 1.5, 2.5], dtype=torch.float32)
    base_logits = torch.randn(2, 2, 3, 4, dtype=torch.float32)
    x = torch.tensor(
        [
            [-1.25, -0.25],
            [0.00, 0.75],
            [1.10, 1.90],
            [2.20, -1.90],
        ],
        dtype=torch.float32,
    )

    actual_logits = base_logits.detach().clone().requires_grad_(True)
    expected_logits = base_logits.detach().clone().requires_grad_(True)
    actual_dist = HistogramDist(bin_edges=bin_edges, logits=actual_logits, min_prob=1e-12)
    expected_dist = HistogramDist(bin_edges=bin_edges, logits=expected_logits, min_prob=1e-12)

    actual_out = actual_dist.log_prob(x)
    expected_out = _reference_log_prob(expected_dist, x)

    torch.testing.assert_close(actual_out, expected_out, rtol=1e-6, atol=1e-6)

    actual_out.sum().backward()
    expected_out.sum().backward()

    assert actual_logits.grad is not None
    assert expected_logits.grad is not None
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad, rtol=1e-6, atol=1e-6)


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
