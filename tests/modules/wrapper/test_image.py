import unittest
from itertools import product

from spflow.exceptions import InvalidParameterCombinationError
from spflow.learn import train_gradient_descent
from tests.fixtures import auto_set_test_seed, auto_set_test_device

from spflow.meta.dispatch import init_default_sampling_context, SamplingContext
from tests.utils.leaves import evaluate_log_likelihood
import pytest
import torch
from spflow import log_likelihood, sample
from spflow import maximum_likelihood_estimation, sample, marginalize
from spflow.meta.data import Scope
from spflow.modules import leaf
from tests.utils.leaves import make_leaf, make_data, make_leaf_args, make_normal_leaf
from spflow.learn.expectation_maximization import expectation_maximization
from spflow.modules.wrapper.ImageWrapper import ImageWrapper, Pixel, MarginalizationContext
from spflow.modules import Product, Sum

num_channel = [1, 3]
num_repetitions = [None, 3]
implicit_product = [True, False]

params = list(product(num_channel, implicit_product, num_repetitions))

def make_wrapper(num_channel, implicit_product, num_reps):
    height = 4
    width = 4
    num_features = height * width * num_channel
    scope = Scope(list(range(num_features)))
    leaf_module = make_normal_leaf(scope=scope, out_channels=2, num_repetitions=num_reps)
    product_layer = Product(inputs=leaf_module)
    root = Sum(inputs=product_layer, out_channels=1, num_repetitions=num_reps)
    wrapper = ImageWrapper(module=root, height=height, width=width, num_channel=num_channel, implicit_product=implicit_product)
    return wrapper


@pytest.mark.parametrize("num_channel, implicit_product, num_reps", params)
def test_log_likelihood( num_channel: int, implicit_product: bool, num_reps):
    """Test the log likelihood of a normal distribution."""

    module = make_wrapper(num_channel, implicit_product, num_reps)
    height = 4
    width = 4
    out_features = height * width * num_channel

    data = make_data(cls=leaf.Normal, out_features=out_features, n_samples=5)
    data = data.view(data.shape[0], num_channel, height, width)
    lls = log_likelihood(module, data, check_support=True)
    if num_reps is not None:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels, num_reps)
    else:
        assert lls.shape == (data.shape[0], module.out_features, module.out_channels)
    assert torch.isfinite(lls).all()


@pytest.mark.parametrize(
    " num_channel, is_mpe",
    list(product(num_channel, [True, False])),
)
def test_sample(num_channel:int, is_mpe: bool):
    #cls = leaf.Normal
    module = make_wrapper(num_channel, implicit_product=False, num_reps=None)
    # Setup sampling context
    n_samples = 10
    height = 4
    width = 4
    samples = sample(module, is_mpe=is_mpe)

    assert samples.shape == (1, num_channel, height, width)

    # Check finite
    assert torch.isfinite(samples).all()



@pytest.mark.parametrize("num_channel, implicit_product, num_reps", params)
def test_expectation_maximization(
        num_channel: int,
        implicit_product: bool,
        num_reps,

):
    height = 4
    width = 4

    module = make_wrapper(num_channel, implicit_product, num_reps)
    data = make_data(cls=leaf.Normal, out_features=height* width * num_channel, n_samples=20)
    data = data.reshape(20, num_channel, height, width)

    expectation_maximization(module, data, max_steps=10)


@pytest.mark.parametrize(
    "height_indices, width_indices, channel_indices",
    list(
        product(
            [[0],[1,2],None],
            [[0],[1,2],None],
            [[0],[1,2],None]
        )
    ),
)
def test_marginalize(height_indices, width_indices, channel_indices):
    """Test marginalization of a normal distribution."""
    height = 4
    width = 4
    num_channel = 3
    num_reps = 3
    num_features = height * width * num_channel

    module = make_wrapper(num_channel, implicit_product=False, num_reps=num_reps)
    marg_ctx = MarginalizationContext(c=channel_indices, h=height_indices, w=width_indices)
    marginalized_module = marginalize(module, marg_ctx)


    assert marginalized_module.height == height - len(height_indices) if height_indices is not None else height
    assert marginalized_module.width == width - len(width_indices) if width_indices is not None else width
    assert marginalized_module.num_channel == num_channel - len(channel_indices) if channel_indices is not None else num_channel



# @pytest.mark.parametrize(
#     "num_channel, scope",
#     list(product(num_channel,
#                  [
#                      Scope([0]), Scope([10]), Scope([3])
#                  ]
#
#                  ),
# ))
# def test_pixel_scope_conversion(num_channel,scope):
#     """Test conversion between pixel and feature scopes."""
#     out_features = 12 // num_channel
#     leaf_wrapper = make_wrapper(leaf.Normal, out_features, num_channel, False, 2, None)
#     pixel_scope = leaf_wrapper.scope_to_pixel(scope)
#     feature_scope = leaf_wrapper.pixel_to_scope(pixel_scope)
#     assert scope.equal_query(feature_scope)
#
#
# @pytest.mark.parametrize(
#     "targets",
#     [(Pixel(0, 1, 1), 5),
#     (Pixel(0, 2, 2), 9),
#     (Pixel(1, 0, 0), 12),
#     (Pixel(2, 2, 2), 42),
#      ]
#
# )
# def test_pixel_to_index(targets):
#     """Test conversion between pixel and feature scopes."""
#     # ToDo: make module for data = (b, c, h, w)
#     out_features = 16
#     num_channel = 3
#     leaf_wrapper = make_wrapper(leaf.Normal, out_features, num_channel, False, 2, None)
#     pixel_scope = leaf_wrapper.pixel_to_index(targets[0])
#     assert pixel_scope == targets[1]




