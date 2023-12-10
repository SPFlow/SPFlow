import random
import unittest

import numpy as np
import pytest
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.tensorly.inference import log_likelihood
from spflow.torch.learning import (
    em,
    expectation_maximization,
    maximum_likelihood_estimation,
)
from spflow.tensorly.structure.spn import GammaLayer#, ProductNode, SumNode
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.tensorly.utils.helper_functions import tl_toNumpy
from spflow.torch.structure.general.layers.leaves.parametric.gamma import updateBackend

tc = unittest.TestCase()

def test_mle(do_for_all_backends):

    # set seed
    np.random.seed(0)
    random.seed(0)

    layer = GammaLayer(scope=[Scope([0]), Scope([1])])

    # simulate data
    data = np.hstack(
        [
            np.random.gamma(shape=0.3, scale=1.0 / 1.7, size=(50000, 1)),
            np.random.gamma(shape=1.9, scale=1.0 / 0.7, size=(50000, 1)),
        ]
    )

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data), bias_correction=True)

    tc.assertTrue(np.allclose(tl_toNumpy(layer.alpha), tl.tensor([0.3, 1.9]), atol=1e-3, rtol=1e-2))
    tc.assertTrue(np.allclose(tl_toNumpy(layer.beta), tl.tensor([1.7, 0.7]), atol=1e-3, rtol=1e-2))

def test_mle_only_nans(do_for_all_backends):

    layer = GammaLayer(scope=[Scope([0]), Scope([1])])

    # simulate data
    data = tl.tensor([[float("nan"), float("nan")], [float("nan"), 0.5]])

    # check if exception is raised
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        data,
        nan_strategy="ignore",
    )

def test_mle_invalid_support(do_for_all_backends):

    layer = GammaLayer(Scope([0]))

    # perform MLE (should raise exceptions)
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        tl.tensor([[float("inf")]]),
        bias_correction=True,
    )
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        tl.tensor([[-0.1]]),
        bias_correction=True,
    )

def test_mle_nan_strategy_none(do_for_all_backends):

    layer = GammaLayer(Scope([0]))
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        tl.tensor([[float("nan")], [0.1], [1.9], [0.7]]),
        nan_strategy=None,
    )

def test_mle_nan_strategy_ignore(do_for_all_backends):

    layer = GammaLayer(Scope([0]))
    maximum_likelihood_estimation(
        layer,
        tl.tensor([[float("nan")], [0.1], [1.9], [0.7]]),
        nan_strategy="ignore",
        bias_correction=False,
    )
    alpha_ignore, beta_ignore = tl_toNumpy(layer.alpha), tl_toNumpy(layer.beta)

    # since gamma is estimated iteratively by scipy, just make sure it matches the estimate without nan value
    maximum_likelihood_estimation(
        layer,
        tl.tensor([[0.1], [1.9], [0.7]]),
        nan_strategy=None,
        bias_correction=False,
    )
    alpha_none, beta_none = tl_toNumpy(layer.alpha), tl_toNumpy(layer.beta)

    tc.assertTrue(np.allclose(alpha_ignore, alpha_none))
    tc.assertTrue(np.allclose(beta_ignore, beta_none))

def test_mle_nan_strategy_callable(do_for_all_backends):

    layer = GammaLayer(Scope([0]))
    # should not raise an issue
    maximum_likelihood_estimation(layer, tl.tensor([[0.5], [1]]), nan_strategy=lambda x: x)

def test_mle_nan_strategy_invalid(do_for_all_backends):

    layer = GammaLayer(Scope([0]))
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        tl.tensor([[float("nan")], [0.1], [1.9], [0.7]]),
        nan_strategy="invalid_string",
    )
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        tl.tensor([[float("nan")], [1], [0], [1]]),
        nan_strategy=1,
    )

def test_weighted_mle(do_for_all_backends):

    leaf = GammaLayer([Scope([0]), Scope([1])])

    data = tl.tensor(
        np.hstack(
            [
                np.vstack(
                    [
                        np.random.gamma(shape=1.7, scale=1.0 / 0.8, size=(10000, 1)),
                        np.random.gamma(shape=0.5, scale=1.0 / 1.4, size=(10000, 1)),
                    ]
                ),
                np.vstack(
                    [
                        np.random.gamma(shape=0.9, scale=1.0 / 0.3, size=(10000, 1)),
                        np.random.gamma(shape=1.3, scale=1.0 / 1.7, size=(10000, 1)),
                    ]
                ),
            ]
        )
    )
    weights = tl.concatenate([tl.zeros(10000), tl.ones(10000)])

    maximum_likelihood_estimation(leaf, data, weights)

    tc.assertTrue(np.allclose(tl_toNumpy(leaf.alpha), tl.tensor([0.5, 1.3]), atol=1e-3, rtol=1e-2))
    tc.assertTrue(np.allclose(tl_toNumpy(leaf.beta), tl.tensor([1.4, 1.7]), atol=1e-2, rtol=1e-1))

def test_em_step(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = GammaLayer([Scope([0]), Scope([1])])
    data = tl.tensor(
        np.hstack(
            [
                np.random.gamma(shape=0.3, scale=1.0 / 1.7, size=(10000, 1)),
                np.random.gamma(shape=1.4, scale=1.0 / 0.8, size=(10000, 1)),
            ]
        )
    )
    dispatch_ctx = DispatchContext()

    # compute gradients of log-likelihoods w.r.t. module log-likelihoods
    ll = log_likelihood(layer, data, dispatch_ctx=dispatch_ctx)
    ll.retain_grad()
    ll.sum().backward()

    # perform an em step
    em(layer, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(layer.alpha), tl.tensor([0.3, 1.4]), atol=1e-2, rtol=1e-1))
    tc.assertTrue(np.allclose(tl_toNumpy(layer.beta), tl.tensor([1.7, 0.8]), atol=1e-2, rtol=1e-1))

def test_em_product_of_gammas(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = GammaLayer([Scope([0]), Scope([1])])
    prod_node = ProductNode([layer])

    data = tl.tensor(
        np.hstack(
            [
                np.random.gamma(shape=0.3, scale=1.0 / 1.7, size=(15000, 1)),
                np.random.gamma(shape=1.4, scale=1.0 / 0.8, size=(15000, 1)),
            ]
        )
    )

    expectation_maximization(prod_node, data, max_steps=10)

    tc.assertTrue(np.allclose(tl_toNumpy(layer.alpha), tl.tensor([0.3, 1.4]), atol=1e-2, rtol=1e-1))
    tc.assertTrue(np.allclose(tl_toNumpy(layer.beta), tl.tensor([1.7, 0.8]), atol=1e-2, rtol=1e-1))

def test_em_sum_of_gammas(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = GammaLayer([Scope([0]), Scope([0])], alpha=[1.2, 0.6], beta=[0.5, 1.9])
    sum_node = SumNode([layer], weights=[0.5, 0.5])

    data = tl.tensor(
        np.vstack(
            [
                np.random.gamma(shape=0.9, scale=1.0 / 1.9, size=(20000, 1)),
                np.random.gamma(shape=1.4, scale=1.0 / 0.8, size=(20000, 1)),
            ]
        )
    )

    expectation_maximization(sum_node, data, max_steps=10)

    tc.assertTrue(np.allclose(tl_toNumpy(layer.alpha), tl.tensor([1.4, 0.9]), atol=1e-2, rtol=1e-1))
    tc.assertTrue(np.allclose(tl_toNumpy(layer.beta), tl.tensor([0.8, 1.9]), atol=1e-2, rtol=1e-1))

def test_update_backend(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = GammaLayer(scope=[Scope([0]), Scope([1])])

    # simulate data
    data = np.hstack(
        [
            np.random.gamma(shape=0.3, scale=1.0 / 1.7, size=(50000, 1)),
            np.random.gamma(shape=1.9, scale=1.0 / 0.7, size=(50000, 1)),
        ]
    )

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data))

    alpha = tl_toNumpy(tl_toNumpy(layer.alpha))
    beta = tl_toNumpy(tl_toNumpy(layer.beta))

    layer = GammaLayer(scope=[Scope([0]), Scope([1])])
    prod_node = ProductNode([layer])
    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
    alpha_em = tl_toNumpy(layer.alpha)
    beta_em = tl_toNumpy(layer.beta)


    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer = GammaLayer(scope=[Scope([0]), Scope([1])])
            layer_updated = updateBackend(layer)
            maximum_likelihood_estimation(layer_updated, tl.tensor(data))
            # check conversion from torch to python
            alpha_updated = tl_toNumpy(layer_updated.alpha)
            beta_updated = tl_toNumpy(layer_updated.beta)
            tc.assertTrue(np.allclose(alpha, alpha_updated, atol=1e-2, rtol=1e-1))
            tc.assertTrue(np.allclose(beta, beta_updated, atol=1e-2, rtol=1e-1))

            layer = GammaLayer(scope=[Scope([0]), Scope([1])])
            layer_updated = updateBackend(layer)
            prod_node = ProductNode([layer_updated])
            if tl.get_backend() != "pytorch":
                with pytest.raises(NotImplementedError):
                    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
            else:
                expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
                alpha_em_updated = tl_toNumpy(layer_updated.alpha)
                beta_em_updated = tl_toNumpy(layer_updated.beta)
                tc.assertTrue(np.allclose(alpha_em, alpha_em_updated, atol=1e-2, rtol=1e-1))
                tc.assertTrue(np.allclose(beta_em, beta_em_updated, atol=1e-2, rtol=1e-1))

def test_change_dtype(do_for_all_backends):
    np.random.seed(0)
    random.seed(0)

    layer = GammaLayer(scope=[Scope([0]), Scope([1])])
    prod_node = ProductNode([layer])

    # simulate data
    data = np.hstack(
        [
            np.random.gamma(shape=0.3, scale=1.0 / 1.7, size=(50000, 1)),
            np.random.gamma(shape=1.9, scale=1.0 / 0.7, size=(50000, 1)),
        ]
    )

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data, dtype=tl.float32))
    tc.assertTrue(layer.alpha.dtype == tl.float32)
    tc.assertTrue(layer.beta.dtype == tl.float32)

    layer.to_dtype(tl.float64)

    dummy_data = tl.tensor(data, dtype=tl.float64)
    maximum_likelihood_estimation(layer, dummy_data)
    tc.assertTrue(layer.alpha.dtype == tl.float64)
    tc.assertTrue(layer.beta.dtype == tl.float64)

    if do_for_all_backends == "numpy":
        tc.assertRaises(NotImplementedError, expectation_maximization, prod_node, tl.tensor(data, dtype=tl.float64),
                        max_steps=10)
    else:
        # test if em runs without error after dype change
        expectation_maximization(prod_node, tl.tensor(data, dtype=tl.float64), max_steps=10)


def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    np.random.seed(0)
    random.seed(0)

    layer = GammaLayer(scope=[Scope([0]), Scope([1])])
    prod_node = ProductNode([layer])

    # simulate data
    data = np.hstack(
        [
            np.random.gamma(shape=0.3, scale=1.0 / 1.7, size=(50000, 1)),
            np.random.gamma(shape=1.9, scale=1.0 / 0.7, size=(50000, 1)),
        ]
    )

    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer.to_device, cuda)
        return

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data, dtype=tl.float32))

    tc.assertTrue(layer.alpha.device.type == "cpu")
    tc.assertTrue(layer.beta.device.type == "cpu")

    layer.to_device(cuda)

    dummy_data = tl.tensor(data, dtype=tl.float32, device=cuda)

    # perform MLE
    maximum_likelihood_estimation(layer, dummy_data)
    tc.assertTrue(layer.alpha.device.type == "cuda")
    tc.assertTrue(layer.beta.device.type == "cuda")

    # test if em runs without error after device change
    expectation_maximization(prod_node, tl.tensor(data, dtype=tl.float32, device=cuda), max_steps=10)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
