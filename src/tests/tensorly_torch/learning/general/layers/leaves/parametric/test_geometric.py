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
from spflow.tensorly.structure.spn import GeometricLayer#, ProductNode, SumNode
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.tensorly.utils.helper_functions import tl_toNumpy
from spflow.torch.structure.general.layers.leaves.parametric.geometric import updateBackend

tc = unittest.TestCase()

def test_mle(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = GeometricLayer(scope=[Scope([0]), Scope([1])])

    # simulate data
    data = np.hstack(
        [
            np.random.geometric(p=0.3, size=(10000, 1)),
            np.random.geometric(p=0.7, size=(10000, 1)),
        ]
    )

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data), bias_correction=True)
    tc.assertTrue(np.allclose(tl_toNumpy(layer.p), tl.tensor([0.3, 0.7]), atol=1e-2, rtol=1e-2))

def test_mle_bias_correction(do_for_all_backends):

    layer = GeometricLayer(Scope([0]))
    data = tl.tensor([[1.0], [3.0]])

    # perform MLE
    maximum_likelihood_estimation(layer, data, bias_correction=False)
    tc.assertTrue(np.allclose(tl_toNumpy(layer.p), tl.tensor(2.0 / 4.0)))

    # perform MLE
    maximum_likelihood_estimation(layer, data, bias_correction=True)
    tc.assertTrue(np.allclose(tl_toNumpy(layer.p), tl.tensor(1.0 / 4.0)))

def test_mle_edge_1(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = GeometricLayer(Scope([0]))

    # simulate data
    data = tl.ones((100, 1))

    # perform MLE
    maximum_likelihood_estimation(layer, data, bias_correction=True)
    tc.assertTrue(np.all(tl_toNumpy(layer.p) < 1.0))

def test_mle_only_nans(do_for_all_backends):

    layer = GeometricLayer(scope=[Scope([0]), Scope([1])])

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

    layer = GeometricLayer(Scope([0]))

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
        tl.tensor([[0]]),
        bias_correction=True,
    )

def test_mle_nan_strategy_none(do_for_all_backends):

    layer = GeometricLayer(Scope([0]))
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        tl.tensor([[float("nan")], [1], [4], [3]]),
        nan_strategy=None,
    )

def test_mle_nan_strategy_ignore(do_for_all_backends):

    layer = GeometricLayer(Scope([0]))
    maximum_likelihood_estimation(
        layer,
        tl.tensor([[float("nan")], [1], [4], [3]]),
        nan_strategy="ignore",
        bias_correction=False,
    )
    tc.assertTrue(np.allclose(tl_toNumpy(layer.p), tl.tensor(3.0 / 8.0)))

def test_mle_nan_strategy_callable(do_for_all_backends):

    layer = GeometricLayer(Scope([0]))
    # should not raise an issue
    maximum_likelihood_estimation(layer, tl.tensor([[2], [1]]), nan_strategy=lambda x: x)

def test_mle_nan_strategy_invalid(do_for_all_backends):

    layer = GeometricLayer(Scope([0]))
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        tl.tensor([[float("nan")], [1], [4], [3]]),
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

    leaf = GeometricLayer([Scope([0]), Scope([1])], n_nodes=3)

    data = tl.tensor(
        np.hstack(
            [
                np.vstack(
                    [
                        np.random.geometric(p=0.8, size=(10000, 1)),
                        np.random.geometric(p=0.2, size=(10000, 1)),
                    ]
                ),
                np.vstack(
                    [
                        np.random.geometric(p=0.3, size=(10000, 1)),
                        np.random.geometric(p=0.7, size=(10000, 1)),
                    ]
                ),
            ]
        )
    )

    weights = tl.concatenate([tl.zeros(10000), tl.ones(10000)])

    maximum_likelihood_estimation(leaf, data, weights)

    tc.assertTrue(np.allclose(tl_toNumpy(leaf.p), tl.tensor([0.2, 0.7]), atol=1e-3, rtol=1e-2))

def test_em_step(do_for_all_backends):

    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = GeometricLayer([Scope([0]), Scope([1])])
    data = tl.tensor(
        np.hstack(
            [
                np.random.geometric(p=0.2, size=(10000, 1)),
                np.random.geometric(p=0.7, size=(10000, 1)),
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

    tc.assertTrue(np.allclose(tl_toNumpy(layer.p), tl.tensor([0.2, 0.7]), atol=1e-2, rtol=1e-3))

def test_em_product_of_geometrics(do_for_all_backends):

    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = GeometricLayer([Scope([0]), Scope([1])])
    prod_node = ProductNode([layer])

    data = tl.tensor(
        np.hstack(
            [
                np.random.geometric(p=0.8, size=(10000, 1)),
                np.random.geometric(p=0.2, size=(10000, 1)),
            ]
        )
    )

    expectation_maximization(prod_node, data, max_steps=10)

    tc.assertTrue(np.allclose(tl_toNumpy(layer.p), tl.tensor([0.8, 0.2]), atol=1e-3, rtol=1e-2))

def test_em_sum_of_geometrics(do_for_all_backends):

    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    leaf = GeometricLayer(Scope([0]), n_nodes=2, p=[0.4, 0.6])
    sum_node = SumNode([leaf], weights=[0.5, 0.5])

    data = tl.tensor(
        np.vstack(
            [
                np.random.geometric(p=0.8, size=(10000, 1)),
                np.random.geometric(p=0.2, size=(10000, 1)),
            ]
        )
    )

    expectation_maximization(sum_node, data, max_steps=10)

    tc.assertTrue(np.allclose(tl_toNumpy(leaf.p), tl.tensor([0.2, 0.8]), atol=1e-2, rtol=1e-2))

def test_update_backend(do_for_all_backends):

    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = GeometricLayer(scope=[Scope([0]), Scope([1])])

    # simulate data
    data = np.hstack(
        [
            np.random.geometric(p=0.3, size=(10000, 1)),
            np.random.geometric(p=0.7, size=(10000, 1)),
        ]
    )

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data))

    params = tl_toNumpy(layer.p)

    layer = GeometricLayer(scope=[Scope([0]), Scope([1])])
    prod_node = ProductNode([layer])
    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
    params_em = tl_toNumpy(layer.p)


    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer = GeometricLayer(scope=[Scope([0]), Scope([1])])
            layer_updated = updateBackend(layer)
            maximum_likelihood_estimation(layer_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(layer_updated.p), params, atol=1e-2, rtol=1e-3))

            layer = GeometricLayer(scope=[Scope([0]), Scope([1])])
            layer_updated = updateBackend(layer)
            prod_node = ProductNode([layer_updated])
            if tl.get_backend() != "pytorch":
                with pytest.raises(NotImplementedError):
                    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
            else:
                expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
                tc.assertTrue(np.allclose(tl_toNumpy(layer_updated.p), params_em, atol=1e-3, rtol=1e-2))


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    unittest.main()
