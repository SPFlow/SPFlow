"""Tests for wrapper base class behavior."""

import torch

from spflow.modules.wrapper.base import Wrapper
from spflow.utils.sampling_context import DifferentiableSamplingContext
from tests.utils.leaves import make_normal_leaf


class _IdentityWrapper(Wrapper):
    def log_likelihood(self, data, cache=None):
        return self.module.log_likelihood(data, cache=cache)

    def sample(self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None):
        return self.module.sample(
            num_samples=num_samples,
            data=data,
            is_mpe=is_mpe,
            cache=cache,
        )

    def _expectation_maximization_step(self, data, bias_correction: bool = True, *, cache):
        return self.module._expectation_maximization_step(data, bias_correction=bias_correction, cache=cache)

    def maximum_likelihood_estimation(
        self, data, weights=None, bias_correction: bool = True, nan_strategy="ignore"
    ):
        return self.module.maximum_likelihood_estimation(
            data=data,
            weights=weights,
            bias_correction=bias_correction,
            nan_strategy=nan_strategy,
        )

    def marginalize(self, marg_rvs, prune: bool = True, cache=None):
        return self.module.marginalize(marg_rvs=marg_rvs, prune=prune, cache=cache)


def test_wrapper_delegates_shape_scope_and_feature_map() -> None:
    leaf = make_normal_leaf(out_features=3, out_channels=2, num_repetitions=1)
    wrapper = _IdentityWrapper(module=leaf)

    assert wrapper.scope == leaf.scope
    assert wrapper.in_shape == leaf.in_shape
    assert wrapper.out_shape == leaf.out_shape
    assert wrapper.feature_to_scope.shape == leaf.feature_to_scope.shape


def test_wrapper_device_and_repr() -> None:
    leaf = make_normal_leaf(out_features=2, out_channels=1, num_repetitions=1)
    wrapper = _IdentityWrapper(module=leaf)
    assert wrapper.device == next(leaf.parameters()).device

    repr_str = wrapper.extra_repr()
    assert "D=" in repr_str
    assert "C=" in repr_str
    assert "R=" in repr_str


def test_wrapper_log_likelihood_passthrough() -> None:
    leaf = make_normal_leaf(out_features=2, out_channels=1, num_repetitions=1)
    wrapper = _IdentityWrapper(module=leaf)
    data = torch.randn(4, 2)
    ll = wrapper.log_likelihood(data)
    assert ll.shape[0] == 4


def test_wrapper_rsample_passthrough() -> None:
    leaf = make_normal_leaf(out_features=2, out_channels=2, num_repetitions=1)
    wrapper = _IdentityWrapper(module=leaf)
    num_samples = 5
    channel_probs = torch.rand((num_samples, wrapper.out_shape.features, wrapper.out_shape.channels))
    channel_probs = channel_probs / channel_probs.sum(dim=-1, keepdim=True)
    sampling_ctx = DifferentiableSamplingContext(
        channel_probs=channel_probs,
        mask=torch.ones((num_samples, wrapper.out_shape.features), dtype=torch.bool),
    )

    samples = wrapper.rsample(num_samples=num_samples, diff_method="gumbel")
    assert samples.shape == (num_samples, 2)
    assert torch.isfinite(samples).all()
