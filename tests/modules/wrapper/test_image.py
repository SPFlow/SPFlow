from itertools import product

import pytest
import torch

from spflow.exceptions import ShapeError, StructureError
from spflow.learn.expectation_maximization import expectation_maximization
from spflow.meta.data import Scope
from spflow.modules import Product, Sum
from spflow.modules import leaf
from spflow.modules.wrapper.image_wrapper import ImageWrapper, MarginalizationContext
from tests.utils.leaves import make_data, make_normal_leaf

num_channel = [1, 3]
num_repetitions = [None, 3]

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


@pytest.mark.parametrize("num_channel, num_reps", params)
def test_log_likelihood(num_channel: int, num_reps):
    """Test the log likelihood of a normal distribution."""

    module = make_wrapper(num_channel, num_reps)
    height = 4
    width = 4
    out_features = height * width * num_channel

    data = make_data(cls=leaf.Normal, out_features=out_features, n_samples=5)
    data = data.view(data.shape[0], num_channel, height, width)
    lls = module.log_likelihood(data)
    if num_reps is not None:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels, num_reps)
    else:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels)
    assert torch.isfinite(lls).all()


@pytest.mark.parametrize(
    " num_channel, is_mpe",
    list(product(num_channel, [True, False])),
)
def test_sample(num_channel: int, is_mpe: bool):
    # cls = leaf.Normal
    module = make_wrapper(num_channel, num_reps=None)
    # Setup sampling context
    n_samples = 10
    height = 4
    width = 4
    # Create dummy data tensor for sampling
    data = torch.full((1, num_channel, height, width), float("nan"))
    samples = module.sample(data=data, is_mpe=is_mpe)

    assert samples.shape == (1, num_channel, height, width)

    # Check finite
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("num_channel, num_reps", params)
def test_expectation_maximization(
    num_channel: int,
    num_reps,
):
    height = 4
    width = 4

    module = make_wrapper(num_channel, num_reps)
    data = make_data(cls=leaf.Normal, out_features=height * width * num_channel, n_samples=20)
    data = data.reshape(20, num_channel, height, width)

    expectation_maximization(module, data, max_steps=10)


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


# Exception tests for ImageWrapper
class TestImageWrapperExceptions:
    """Test suite for ImageWrapper exception handling."""

    def test_constructor_shape_mismatch(self):
        """Test that StructureError is raised when module out_features doesn't match height*width*num_channel."""
        height = 4
        width = 4
        num_channel = 3
        num_features = height * width * num_channel  # 48 features

        # Create a module with different number of features (e.g., 50)
        scope = Scope(list(range(50)))
        leaf_module = make_normal_leaf(scope=scope, out_channels=2, num_repetitions=None)
        product_layer = Product(inputs=leaf_module)
        root = Sum(inputs=product_layer, out_channels=1, num_repetitions=None)

        # This should raise StructureError because 50 != 4*4*3
        with pytest.raises(StructureError):
            ImageWrapper(module=root, height=height, width=width, num_channel=num_channel)

    def test_flatten_wrong_dimensions(self):
        """Test that ShapeError is raised when flatten() receives non-4D tensor."""
        wrapper = make_wrapper(num_channel=3, num_reps=None)

        # 3D tensor instead of 4D
        wrong_tensor = torch.randn(10, 3, 16)
        with pytest.raises(ShapeError):
            wrapper.flatten(wrong_tensor)

    def test_flatten_wrong_channel_dimension(self):
        """Test that ShapeError is raised when flatten() receives wrong channel dimension."""
        wrapper = make_wrapper(num_channel=3, num_reps=None)

        # 4D tensor but with wrong channel dimension (2 instead of 3)
        wrong_tensor = torch.randn(10, 2, 4, 4)
        with pytest.raises(ShapeError):
            wrapper.flatten(wrong_tensor)

    def test_to_image_format_batch_wrong_dimensions(self):
        """Test that ShapeError is raised when to_image_format() with batch=True receives non-2D tensor."""
        wrapper = make_wrapper(num_channel=3, num_reps=None)

        # 3D tensor instead of 2D for batch mode
        wrong_tensor = torch.randn(10, 3, 16)
        with pytest.raises(ShapeError):
            wrapper.to_image_format(wrong_tensor, batch=True)

    def test_to_image_format_non_batch_wrong_dimensions(self):
        """Test that ShapeError is raised when to_image_format() with batch=False receives non-1D tensor."""
        wrapper = make_wrapper(num_channel=3, num_reps=None)

        # 2D tensor instead of 1D for non-batch mode
        wrong_tensor = torch.randn(3, 16)
        with pytest.raises(ShapeError):
            wrapper.to_image_format(wrong_tensor, batch=False)

    def test_log_likelihood_wrong_shape(self):
        """Test that ShapeError is raised when log_likelihood() receives wrong shaped data."""
        wrapper = make_wrapper(num_channel=3, num_reps=None)

        # Wrong shape: (10, 3, 4, 5) instead of (10, 3, 4, 4)
        wrong_data = torch.randn(10, 3, 4, 5)
        with pytest.raises(ShapeError):
            wrapper.log_likelihood(wrong_data)

    def test_log_likelihood_wrong_channels(self):
        """Test that ShapeError is raised when log_likelihood() receives wrong number of channels."""
        wrapper = make_wrapper(num_channel=3, num_reps=None)

        # Wrong channels: (10, 2, 4, 4) instead of (10, 3, 4, 4)
        wrong_data = torch.randn(10, 2, 4, 4)
        with pytest.raises(ShapeError):
            wrapper.log_likelihood(wrong_data)

    def test_expectation_maximization_wrong_shape(self):
        """Test that ShapeError is raised when expectation_maximization() receives wrong shaped data."""
        wrapper = make_wrapper(num_channel=3, num_reps=3)

        # Wrong shape: (10, 3, 4, 5) instead of (10, 3, 4, 4)
        wrong_data = torch.randn(10, 3, 4, 5)
        with pytest.raises(ShapeError):
            wrapper.expectation_maximization(wrong_data)

    def test_maximum_likelihood_estimation_wrong_shape(self):
        """Test that ShapeError is raised when maximum_likelihood_estimation() receives wrong shaped data."""
        wrapper = make_wrapper(num_channel=3, num_reps=3)

        # Wrong shape: (10, 3, 4, 5) instead of (10, 3, 4, 4)
        wrong_data = torch.randn(10, 3, 4, 5)
        with pytest.raises(ShapeError):
            wrapper.maximum_likelihood_estimation(wrong_data)
