import random
import unittest

import numpy as np
import torch
import tensorly as tl
import pytest

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.tensorly.inference import log_likelihood
from spflow.torch.learning import (
    em,
    expectation_maximization,
    maximum_likelihood_estimation,
)
from spflow.tensorly.structure.spn import Gamma
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.torch.structure.general.nodes.leaves.parametric.gamma import updateBackend

from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_mle_1(do_for_all_backends):

    # set seed
    np.random.seed(0)
    random.seed(0)

    leaf = Gamma(Scope([0]))

    # simulate data
    data = np.random.gamma(shape=0.3, scale=1.0 / 1.7, size=(30000, 1))

    # perform MLE
    maximum_likelihood_estimation(leaf, tl.tensor(data), bias_correction=True)

    tc.assertTrue(np.isclose(tl_toNumpy(leaf.alpha), tl.tensor(0.3), atol=1e-3, rtol=1e-2))
    tc.assertTrue(np.isclose(tl_toNumpy(leaf.beta), tl.tensor(1.7), atol=1e-3, rtol=1e-2))

def test_mle_2(do_for_all_backends):

    # set seed
    np.random.seed(0)
    random.seed(0)

    leaf = Gamma(Scope([0]))

    # simulate data
    data = np.random.gamma(shape=1.9, scale=1.0 / 0.7, size=(30000, 1))

    # perform MLE
    maximum_likelihood_estimation(leaf, tl.tensor(data), bias_correction=True)

    tc.assertTrue(np.isclose(tl_toNumpy(leaf.alpha), tl.tensor(1.9), atol=1e-2, rtol=1e-2))
    tc.assertTrue(np.isclose(tl_toNumpy(leaf.beta), tl.tensor(0.7), atol=1e-2, rtol=1e-2))

def test_mle_only_nans(do_for_all_backends):

    leaf = Gamma(Scope([0]))

    # simulate data
    data = tl.tensor([[float("nan")], [float("nan")]])

    # check if exception is raised
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        leaf,
        data,
        nan_strategy="ignore",
    )

def test_mle_invalid_support(do_for_all_backends):

    leaf = Gamma(Scope([0]))

    # perform MLE (should raise exceptions)
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        leaf,
        tl.tensor([[float("inf")]]),
        bias_correction=True,
    )
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        leaf,
        tl.tensor([[-0.1]]),
        bias_correction=True,
    )

def test_mle_nan_strategy_none(do_for_all_backends):

    leaf = Gamma(Scope([0]))
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        leaf,
        tl.tensor([[float("nan")], [0.1], [1.9], [0.7]]),
        nan_strategy=None,
    )

def test_mle_nan_strategy_ignore(do_for_all_backends):

    leaf = Gamma(Scope([0]))
    maximum_likelihood_estimation(
        leaf,
        tl.tensor([[float("nan")], [0.1], [1.9], [0.7]]),
        nan_strategy="ignore",
        bias_correction=False,
    )
    alpha_ignore, beta_ignore = tl_toNumpy(leaf.alpha), tl_toNumpy(leaf.beta)

    # since gamma is estimated iteratively by scipy, just make sure it matches the estimate without nan value
    maximum_likelihood_estimation(
        leaf,
        tl.tensor([[0.1], [1.9], [0.7]]),
        nan_strategy=None,
        bias_correction=False,
    )
    alpha_none, beta_none = tl_toNumpy(leaf.alpha), tl_toNumpy(leaf.beta)

    tc.assertTrue(np.isclose(alpha_ignore, alpha_none))
    tc.assertTrue(np.isclose(beta_ignore, beta_none))

def test_mle_nan_strategy_callable(do_for_all_backends):

    leaf = Gamma(Scope([0]))
    # should not raise an issue
    maximum_likelihood_estimation(leaf, tl.tensor([[0.5], [1]]), nan_strategy=lambda x: x)

def test_mle_nan_strategy_invalid(do_for_all_backends):

    leaf = Gamma(Scope([0]))
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        leaf,
        tl.tensor([[float("nan")], [0.1], [1.9], [0.7]]),
        nan_strategy="invalid_string",
    )
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        leaf,
        tl.tensor([[float("nan")], [1], [0], [1]]),
        nan_strategy=1,
    )

def test_weighted_mle(do_for_all_backends):
    torch.set_default_tensor_type(torch.DoubleTensor)

    leaf = Gamma(Scope([0]))

    data = tl.tensor(
        np.vstack(
            [
                np.random.gamma(shape=1.7, scale=1.0 / 0.8, size=(10000, 1)),
                np.random.gamma(shape=0.5, scale=1.0 / 1.4, size=(10000, 1)),
            ]
        )
    , dtype=tl.float64)
    weights = tl.concatenate([tl.zeros(10000, dtype=tl.float64), tl.ones(10000, dtype=tl.float64)])

    maximum_likelihood_estimation(leaf, data, weights)

    tc.assertTrue(np.isclose(tl_toNumpy(leaf.alpha), tl.tensor(0.5, dtype=tl.float64), atol=1e-1, rtol=1e-2))
    tc.assertTrue(np.isclose(tl_toNumpy(leaf.beta), tl.tensor(1.4, dtype=tl.float64), atol=1e-1, rtol=1e-2))

def test_em_step(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    leaf = Gamma(Scope([0]))
    data = tl.tensor(np.random.gamma(shape=0.3, scale=1.0 / 1.7, size=(30000, 1)))
    dispatch_ctx = DispatchContext()

    # compute gradients of log-likelihoods w.r.t. module log-likelihoods
    ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
    ll.retain_grad()
    ll.sum().backward()

    # perform an em step
    em(leaf, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.isclose(tl_toNumpy(leaf.alpha), tl.tensor(0.3), atol=1e-3, rtol=1e-1))
    tc.assertTrue(np.isclose(tl_toNumpy(leaf.beta), tl.tensor(1.7), atol=1e-3, rtol=1e-1))

def test_em_product_of_gammas(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    l1 = Gamma(Scope([0]))
    l2 = Gamma(Scope([1]))
    prod_node = ProductNode([l1, l2])

    data = tl.tensor(
        np.hstack(
            [
                np.random.gamma(shape=0.3, scale=1.0 / 1.7, size=(15000, 1)),
                np.random.gamma(shape=1.4, scale=1.0 / 0.8, size=(15000, 1)),
            ]
        )
    )

    expectation_maximization(prod_node, data, max_steps=10)

    tc.assertTrue(np.isclose(tl_toNumpy(l1.alpha), tl.tensor(0.3), atol=1e-2, rtol=1e-2))
    tc.assertTrue(np.isclose(tl_toNumpy(l2.alpha), tl.tensor(1.4), atol=1e-2, rtol=1e-2))
    tc.assertTrue(np.isclose(tl_toNumpy(l1.beta), tl.tensor(1.7), atol=1e-2, rtol=1e-2))
    tc.assertTrue(np.isclose(tl_toNumpy(l2.beta), tl.tensor(0.8), atol=1e-2, rtol=1e-2))

def test_em_sum_of_gammas(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    l1 = Gamma(Scope([0]), alpha=1.2, beta=0.5)
    l2 = Gamma(Scope([0]), alpha=0.6, beta=1.9)
    sum_node = SumNode([l1, l2], weights=[0.5, 0.5])

    data = tl.tensor(
        np.vstack(
            [
                np.random.gamma(shape=0.9, scale=1.0 / 1.9, size=(20000, 1)),
                np.random.gamma(shape=1.4, scale=1.0 / 0.8, size=(20000, 1)),
            ]
        )
    )

    expectation_maximization(sum_node, data, max_steps=10)

    tc.assertTrue(np.isclose(tl_toNumpy(l1.alpha), tl.tensor(1.4), atol=1e-2, rtol=1e-1))
    tc.assertTrue(np.isclose(tl_toNumpy(l2.alpha), tl.tensor(0.9), atol=1e-2, rtol=1e-1))
    tc.assertTrue(np.isclose(tl_toNumpy(l1.beta), tl.tensor(0.8), atol=1e-2, rtol=1e-1))
    tc.assertTrue(np.isclose(tl_toNumpy(l2.beta), tl.tensor(1.9), atol=1e-2, rtol=1e-1))

def test_update_backend(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    leaf = Gamma(Scope([0]))

    # simulate data
    data = np.random.gamma(shape=0.3, scale=1.0 / 1.7, size=(30000, 1))

    # perform MLE
    maximum_likelihood_estimation(leaf, tl.tensor(data))

    params = leaf.get_params()[0]
    params2 = leaf.get_params()[1]

    leaf = Gamma(Scope([0]))
    prod_node = ProductNode([leaf])
    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
    params_em = leaf.get_params()[0]
    params_em2 = leaf.get_params()[1]


    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            leaf = Gamma(Scope([0]))
            leaf_updated = updateBackend(leaf)
            maximum_likelihood_estimation(leaf_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.isclose(leaf_updated.get_params()[0], params, atol=1e-2, rtol=1e-3))
            tc.assertTrue(np.isclose(leaf_updated.get_params()[1], params2, atol=1e-2, rtol=1e-3))

            leaf = Gamma(Scope([0]))
            leaf_updated = updateBackend(leaf)
            prod_node = ProductNode([leaf_updated])
            if tl.get_backend() != "pytorch":
                with pytest.raises(NotImplementedError):
                    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
            else:
                expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
                tc.assertTrue(np.isclose(leaf_updated.get_params()[0], params_em, atol=1e-3, rtol=1e-2))
                tc.assertTrue(np.isclose(leaf_updated.get_params()[1], params_em2, atol=1e-3, rtol=1e-2))


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    unittest.main()
