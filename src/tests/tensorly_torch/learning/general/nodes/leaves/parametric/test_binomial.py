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
from spflow.tensorly.structure.spn import Binomial
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.torch.structure.general.nodes.leaves.parametric.binomial import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_mle_1(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    leaf = Binomial(Scope([0]), n=3)

    # simulate data
    data = np.random.binomial(n=3, p=0.3, size=(10000, 1))

    # perform MLE
    maximum_likelihood_estimation(leaf, tl.tensor(data))

    tc.assertTrue(np.isclose(tl_toNumpy(leaf.p), tl.tensor(0.3), atol=1e-2, rtol=1e-3))

def test_mle_2(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    leaf = Binomial(Scope([0]), n=10)

    # simulate data
    data = np.random.binomial(n=10, p=0.7, size=(10000, 1))

    # perform MLE
    maximum_likelihood_estimation(leaf, tl.tensor(data))

    tc.assertTrue(np.isclose(tl_toNumpy(leaf.p), tl.tensor(0.7), atol=1e-2, rtol=1e-3))

def test_mle_edge_0(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    leaf = Binomial(Scope([0]), n=3)

    # simulate data
    data = np.random.binomial(n=3, p=0.0, size=(100, 1))

    # perform MLE
    maximum_likelihood_estimation(leaf, tl.tensor(data))

    tc.assertTrue(tl_toNumpy(leaf.p) > 0.0)

def test_mle_edge_1(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    leaf = Binomial(Scope([0]), n=5)

    # simulate data
    data = np.random.binomial(n=5, p=1.0, size=(100, 1))

    # perform MLE
    maximum_likelihood_estimation(leaf, tl.tensor(data))

    tc.assertTrue(tl_toNumpy(leaf.p) <= 1.0)

def test_mle_only_nans(do_for_all_backends):

    leaf = Binomial(Scope([0]), n=3)

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

    leaf = Binomial(Scope([0]), n=3)

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
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        leaf,
        tl.tensor([[4]]),
        bias_correction=True,
    )

def test_mle_nan_strategy_none(do_for_all_backends):

    leaf = Binomial(Scope([0]), n=2)
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        leaf,
        tl.tensor([[float("nan")], [1], [2], [1]]),
        nan_strategy=None,
    )

def test_mle_nan_strategy_ignore(do_for_all_backends):

    leaf = Binomial(Scope([0]), n=2)
    maximum_likelihood_estimation(
        leaf,
        tl.tensor([[float("nan")], [1], [2], [1]]),
        nan_strategy="ignore",
    )
    tc.assertTrue(np.isclose(tl_toNumpy(leaf.p), tl.tensor(4.0 / 6.0)))

def test_mle_nan_strategy_callable(do_for_all_backends):

    leaf = Binomial(Scope([0]), n=2)
    # should not raise an issue
    maximum_likelihood_estimation(leaf, tl.tensor([[1.0], [0.0], [1.0]]), nan_strategy=lambda x: x)

def test_mle_nan_strategy_invalid(do_for_all_backends):

    leaf = Binomial(Scope([0]), n=2)
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        leaf,
        tl.tensor([[float("nan")], [1], [0], [1]]),
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

    leaf = Binomial(Scope([0]), n=3)

    data = tl.tensor(
        np.vstack(
            [
                np.random.binomial(n=3, p=0.8, size=(10000, 1)),
                np.random.binomial(n=3, p=0.2, size=(10000, 1)),
            ]
        )
    )
    weights = tl.concatenate([tl.zeros(10000), tl.ones(10000)])

    maximum_likelihood_estimation(leaf, data, weights)

    tc.assertTrue(np.isclose(tl_toNumpy(leaf.p), tl.tensor(0.2), atol=1e-3, rtol=1e-2))

def test_em_step(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    leaf = Binomial(Scope([0]), n=3)
    data = tl.tensor(np.random.binomial(n=3, p=0.3, size=(10000, 1)))
    dispatch_ctx = DispatchContext()

    # compute gradients of log-likelihoods w.r.t. module log-likelihoods
    ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
    ll.retain_grad()
    ll.sum().backward()

    # perform an em step
    em(leaf, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.isclose(tl_toNumpy(leaf.p), tl.tensor(0.3), atol=1e-2, rtol=1e-3))

def test_em_product_of_binomials(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    l1 = Binomial(Scope([0]), n=3, p=0.5)
    l2 = Binomial(Scope([1]), n=5, p=0.5)
    prod_node = ProductNode([l1, l2])

    data = tl.tensor(
        np.hstack(
            [
                np.random.binomial(n=3, p=0.8, size=(10000, 1)),
                np.random.binomial(n=5, p=0.2, size=(10000, 1)),
            ]
        )
    )

    expectation_maximization(prod_node, data, max_steps=10)

    tc.assertTrue(np.isclose(tl_toNumpy(l1.p), tl.tensor(0.8), atol=1e-3, rtol=1e-2))
    tc.assertTrue(np.isclose(tl_toNumpy(l2.p), tl.tensor(0.2), atol=1e-3, rtol=1e-2))

def test_em_sum_of_binomials(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    l1 = Binomial(Scope([0]), n=3, p=0.4)
    l2 = Binomial(Scope([0]), n=3, p=0.6)
    sum_node = SumNode([l1, l2], weights=[0.5, 0.5])

    data = tl.tensor(
        np.vstack(
            [
                np.random.binomial(n=3, p=0.8, size=(10000, 1)),
                np.random.binomial(n=3, p=0.3, size=(10000, 1)),
            ]
        )
    )

    expectation_maximization(sum_node, data, max_steps=10)

    # optimal p
    p_opt = data.sum() / (data.shape[0] * 3)
    # total p represented by mixture
    p_em = (sum_node.weights * tl.tensor([l1.p, l2.p])).sum()

    tc.assertTrue(np.isclose(p_opt, tl_toNumpy(p_em)))

def test_update_backend(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    leaf = Binomial(Scope([0]), n=3)

    # simulate data
    data = np.random.binomial(n=3, p=0.3, size=(10000, 1))

    # perform MLE
    maximum_likelihood_estimation(leaf, tl.tensor(data))

    params = leaf.get_params()[0]

    leaf = Binomial(Scope([0]), n=3)
    prod_node = ProductNode([leaf])
    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
    params_em = leaf.get_params()[0]


    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            leaf = Binomial(Scope([0]), n=3)
            leaf_updated = updateBackend(leaf)
            maximum_likelihood_estimation(leaf_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(leaf_updated.get_params()[0], params, atol=1e-2, rtol=1e-3))

            leaf = Binomial(Scope([0]), n=3)
            leaf_updated = updateBackend(leaf)
            prod_node = ProductNode([leaf_updated])
            if tl.get_backend() != "pytorch":
                with pytest.raises(NotImplementedError):
                    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
            else:
                expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
                tc.assertTrue(np.allclose(leaf_updated.get_params()[0], params_em, atol=1e-3, rtol=1e-2))


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    unittest.main()