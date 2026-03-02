from itertools import product
import numpy as np

import pytest
import torch

from spflow.exceptions import ShapeError, StructureError
from spflow.learn.expectation_maximization import expectation_maximization
from spflow.meta.data import Scope
from spflow.modules import leaves
from spflow.modules.products import Product
from spflow.modules.sums import Sum
from spflow.modules.wrapper.image_wrapper import ImageWrapper, MarginalizationContext
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot
from tests.utils.leaves import DummyLeaf, make_data, make_normal_leaf
from tests.utils.sampling_context_helpers import assert_nonzero_finite_grad, make_diff_routing_from_logits

num_channel = [1, 3]
num_repetitions = [1, 3]

params = list(product(num_channel, num_repetitions))


def make_wrapper(num_channel, num_reps):
    height = 4
    width = 4
    num_features = height * width * num_channel
    scope = Scope(list(range(num_features)))
    leaf_module = make_normal_leaf(scope=scope, out_channels=2, num_repetitions=num_reps)
    product_layer = Product(inputs=leaf_module)
    root = Sum(inputs=product_layer, out_channels=1, num_repetitions=num_reps)
    wrapper = ImageWrapper(module=root, height=height, width=width, num_channel=num_channel)
    return wrapper


def make_leaf_wrapper(num_channel: int, num_reps: int, out_channels: int = 3) -> ImageWrapper:
    height = 4
    width = 4
    num_features = height * width * num_channel
    scope = Scope(list(range(num_features)))
    leaf_module = make_normal_leaf(scope=scope, out_channels=out_channels, num_repetitions=num_reps)
    return ImageWrapper(module=leaf_module, height=height, width=width, num_channel=num_channel)


@pytest.mark.parametrize("num_channel, num_reps", params)
def test_log_likelihood(num_channel: int, num_reps):
    """Test the log likelihood of a normal distribution."""
    module = make_wrapper(num_channel, num_reps)
    height = 4
    width = 4
    out_features = height * width * num_channel

    data = make_data(cls=DummyLeaf, out_features=out_features, n_samples=5)
    data = data.view(data.shape[0], num_channel, height, width)
    lls = module.log_likelihood(data)
    # ImageWrapper keeps image layout while exposing module-level likelihood axes.
    assert lls.shape == (data.shape[0], module.out_shape.features, module.out_shape.channels, num_reps)
    assert torch.isfinite(lls).all()


@pytest.mark.parametrize(
    " num_channel, is_mpe",
    list(product(num_channel, [True, False])),
)
def test_sample(num_channel: int, is_mpe: bool):
    module = make_wrapper(num_channel, num_reps=1)
    n_samples = 10
    height = 4
    width = 4
    data = torch.full((1, num_channel, height, width), float("nan"))
    samples = module.sample(data=data, is_mpe=is_mpe)

    assert samples.shape == (1, num_channel, height, width)

    assert torch.isfinite(samples).all()


def test_wrapper_sample_differentiable_equals_non_diff_sampling():
    num_channel = 3
    num_reps = 3
    module = make_leaf_wrapper(num_channel=num_channel, num_reps=num_reps, out_channels=3)
    n_samples = 10
    height = 4
    width = 4
    num_features = height * width * num_channel

    channel_index = torch.randint(low=0, high=module.out_shape.channels, size=(n_samples, num_features))
    mask = torch.ones((n_samples, num_features), dtype=torch.bool)
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx_a = SamplingContext(
        channel_index=channel_index.clone(),
        mask=mask.clone(),
        repetition_index=repetition_index.clone(),
    )
    sampling_ctx_b = SamplingContext(
        channel_index=to_one_hot(channel_index, dim=-1, dim_size=module.out_shape.channels),
        mask=mask.clone(),
        repetition_index=to_one_hot(repetition_index, dim=-1, dim_size=num_reps),
        is_differentiable=True,
    )

    # Shared RNG seed makes route/value differences attributable to context mode only.
    torch.manual_seed(1337)
    samples_a = module._sample(
        data=torch.full((n_samples, num_channel, height, width), torch.nan),
        sampling_ctx=sampling_ctx_a,
        cache=Cache(),
    )
    torch.manual_seed(1337)
    samples_b = module._sample(
        data=torch.full((n_samples, num_channel, height, width), torch.nan),
        sampling_ctx=sampling_ctx_b,
        cache=Cache(),
    )

    torch.testing.assert_close(samples_a, samples_b, rtol=1e-6, atol=1e-6)


def test_wrapper_sample_differentiable_gradients_flow():
    num_channel = 3
    num_reps = 3
    module = make_leaf_wrapper(num_channel=num_channel, num_reps=num_reps, out_channels=3)
    n_samples = 8
    height = 4
    width = 4
    num_features = height * width * num_channel

    channel_logits, repetition_logits, channel_index, repetition_index = make_diff_routing_from_logits(
        num_samples=n_samples,
        num_features=num_features,
        num_channels=module.out_shape.channels,
        num_repetitions=num_reps,
    )
    sampling_ctx = SamplingContext(
        channel_index=channel_index,
        mask=torch.ones((n_samples, num_features), dtype=torch.bool),
        repetition_index=repetition_index,
        is_differentiable=True,
    )

    out = module._sample(
        data=torch.full((n_samples, num_channel, height, width), torch.nan),
        sampling_ctx=sampling_ctx,
        cache=Cache(),
    )
    loss = torch.nan_to_num(out).sum()
    loss.backward()

    assert_nonzero_finite_grad(channel_logits, "channel_logits")
    assert_nonzero_finite_grad(repetition_logits, "repetition_logits")


@pytest.mark.parametrize("num_channel, num_reps", params)
def test_expectation_maximization(
    num_channel: int,
    num_reps,
):
    height = 4
    width = 4

    module = make_wrapper(num_channel, num_reps)
    data = make_data(cls=DummyLeaf, out_features=height * width * num_channel, n_samples=20)
    data = data.reshape(20, num_channel, height, width)

    loc_before = module.module.inputs.inputs.loc.detach().clone()
    scale_before = module.module.inputs.inputs.scale.detach().clone()

    max_steps = 2
    ll_history = expectation_maximization(module, data, max_steps=max_steps)
    assert ll_history.ndim == 1
    assert 1 <= ll_history.numel() <= max_steps
    assert ll_history.isfinite().all()

    # Dummy leaf M-step re-centers to standard normal moments after one EM cycle.
    assert not torch.equal(module.module.inputs.inputs.loc, loc_before)
    assert not torch.equal(module.module.inputs.inputs.scale, scale_before)
    torch.testing.assert_close(
        module.module.inputs.inputs.loc,
        torch.zeros_like(module.module.inputs.inputs.loc),
    )
    torch.testing.assert_close(
        module.module.inputs.inputs.scale,
        torch.ones_like(module.module.inputs.inputs.scale),
    )


@pytest.mark.parametrize(
    "height_indices, width_indices, channel_indices",
    list(product([[0], [1, 2], None], [[0], [1, 2], None], [[0], [1, 2], None])),
)
def test_marginalize(height_indices, width_indices, channel_indices):
    """Test marginalization of a normal distribution."""
    height = 4
    width = 4
    num_channel = 3
    num_reps = 3
    num_features = height * width * num_channel

    module = make_wrapper(num_channel, num_reps=num_reps)
    marg_ctx = MarginalizationContext(c=channel_indices, h=height_indices, w=width_indices)
    marginalized_module = module.marginalize(marg_ctx)

    assert (
        marginalized_module.height == height - len(height_indices) if height_indices is not None else height
    )
    assert marginalized_module.width == width - len(width_indices) if width_indices is not None else width
    assert (
        marginalized_module.num_channel == num_channel - len(channel_indices)
        if channel_indices is not None
        else num_channel
    )


# Regression tests for strict shape/structure contracts at wrapper boundaries.
class TestImageWrapperExceptions:
    """Test suite for ImageWrapper exception handling."""

    def test_constructor_shape_mismatch(self):
        """Test that StructureError is raised when module out_features doesn't match height*width*num_channel."""
        height = 4
        width = 4
        num_channel = 3
        num_features = height * width * num_channel

        # Mismatch here should fail fast to prevent silent flattening bugs.
        scope = Scope(list(range(50)))
        leaf_module = make_normal_leaf(scope=scope, out_channels=2, num_repetitions=1)
        product_layer = Product(inputs=leaf_module)
        root = Sum(inputs=product_layer, out_channels=1, num_repetitions=1)

        with pytest.raises(StructureError):
            ImageWrapper(module=root, height=height, width=width, num_channel=num_channel)

    def test_flatten_wrong_dimensions(self):
        """Test that ShapeError is raised when flatten() receives non-4D tensor."""
        wrapper = make_wrapper(num_channel=3, num_reps=1)

        wrong_tensor = torch.randn(10, 3, 16)
        with pytest.raises(ShapeError):
            wrapper.flatten(wrong_tensor)

    def test_flatten_wrong_channel_dimension(self):
        """Test that ShapeError is raised when flatten() receives wrong channel dimension."""
        wrapper = make_wrapper(num_channel=3, num_reps=1)

        wrong_tensor = torch.randn(10, 2, 4, 4)
        with pytest.raises(ShapeError):
            wrapper.flatten(wrong_tensor)

    def test_to_image_format_batch_wrong_dimensions(self):
        """Test that ShapeError is raised when to_image_format() with batch=True receives non-2D tensor."""
        wrapper = make_wrapper(num_channel=3, num_reps=1)

        wrong_tensor = torch.randn(10, 3, 16)
        with pytest.raises(ShapeError):
            wrapper.to_image_format(wrong_tensor, batch=True)

    def test_to_image_format_non_batch_wrong_dimensions(self):
        """Test that ShapeError is raised when to_image_format() with batch=False receives non-1D tensor."""
        wrapper = make_wrapper(num_channel=3, num_reps=1)

        wrong_tensor = torch.randn(3, 16)
        with pytest.raises(ShapeError):
            wrapper.to_image_format(wrong_tensor, batch=False)

    def test_log_likelihood_wrong_shape(self):
        """Test that ShapeError is raised when log_likelihood() receives wrong shaped data."""
        wrapper = make_wrapper(num_channel=3, num_reps=1)

        wrong_data = torch.randn(10, 3, 4, 5)
        with pytest.raises(ShapeError):
            wrapper.log_likelihood(wrong_data)

    def test_log_likelihood_wrong_channels(self):
        """Test that ShapeError is raised when log_likelihood() receives wrong number of channels."""
        wrapper = make_wrapper(num_channel=3, num_reps=1)

        wrong_data = torch.randn(10, 2, 4, 4)
        with pytest.raises(ShapeError):
            wrapper.log_likelihood(wrong_data)

    def test_expectation_maximization_wrong_shape(self):
        """Test that ShapeError is raised when expectation_maximization() receives wrong shaped data."""
        wrapper = make_wrapper(num_channel=3, num_reps=3)

        wrong_data = torch.randn(10, 3, 4, 5)
        with pytest.raises(ShapeError):
            wrapper._expectation_maximization_step(wrong_data, cache=Cache())
