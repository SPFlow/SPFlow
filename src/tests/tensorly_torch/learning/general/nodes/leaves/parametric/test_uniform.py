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
from spflow.tensorly.structure.spn import Uniform#,ProductNode, SumNode
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.torch.structure.general.nodes.leaves.parametric.uniform import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_mle(do_for_all_backends):

    leaf = Uniform(Scope([0]), start=0.0, end=1.0)

    # simulate data
    data = tl.tensor([[0.5]])

    # perform MLE (should not raise an exception)
    maximum_likelihood_estimation(leaf, data, bias_correction=True)

    tc.assertTrue(tl.all(tl.tensor([leaf.start, leaf.end]) == tl.tensor([0.0, 1.0])))

def test_mle_invalid_support(do_for_all_backends):

    leaf = Uniform(Scope([0]), start=1.0, end=3.0, support_outside=False)

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
        tl.tensor([[0.0]]),
        bias_correction=True,
    )

def test_em_step(do_for_all_backends):

    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)

    leaf = Uniform(Scope([0]), start=-3.0, end=4.5)
    data = torch.rand((100, 1)) * 7.5 - 3.0
    dispatch_ctx = DispatchContext()

    # compute gradients of log-likelihoods w.r.t. module log-likelihoods
    ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
    ll.requires_grad = True
    ll.retain_grad()
    ll.sum().backward()

    # perform an em step
    em(leaf, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(tl.all(tl.tensor([leaf.start, leaf.end]) == tl.tensor([-3.0, 4.5])))

def test_em_product_of_uniforms(do_for_all_backends):

    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    l1 = Uniform(Scope([0]), start=-1.0, end=3.0)
    l2 = Uniform(Scope([1]), start=2.0, end=5.0)
    prod_node = ProductNode([l1, l2])

    data = tl.tensor(
        np.hstack(
            [
                np.random.rand(15000, 1) * 4.0 - 1.0,
                np.random.rand(15000, 1) * 3.0 + 2.0,
            ]
        )
    )

    expectation_maximization(prod_node, data, max_steps=10)

    tc.assertTrue(tl.all(tl.tensor([l1.start, l1.end]) == tl.tensor([-1.0, 3.0])))
    tc.assertTrue(tl.all(tl.tensor([l2.start, l2.end]) == tl.tensor([2.0, 5.0])))

def test_em_sum_of_uniforms(do_for_all_backends):

    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    l1 = Uniform(Scope([0]), start=-1.0, end=3.0)
    l2 = Uniform(Scope([0]), start=-1.0, end=3.0)
    sum_node = SumNode([l1, l2], weights=[0.5, 0.5])

    data = tl.tensor(np.random.rand(15000, 1) * 3.0 + 2.0)

    expectation_maximization(sum_node, data, max_steps=10)

    tc.assertTrue(tl.all(tl.tensor([l1.start, l1.end]) == tl.tensor([-1.0, 3.0])))
    tc.assertTrue(tl.all(tl.tensor([l2.start, l2.end]) == tl.tensor([-1.0, 3.0])))

def test_update_backend(do_for_all_backends):

    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    leaf = Uniform(Scope([0]), start=0.0, end=1.0)

    # simulate data
    data = tl.tensor([[0.5]])

    # perform MLE
    maximum_likelihood_estimation(leaf, tl.tensor(data))

    params = leaf.get_params()[0]
    params2 = leaf.get_params()[1]

    leaf = Uniform(Scope([0]), start=0.0, end=1.0)
    prod_node = ProductNode([leaf])
    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
    params_em = leaf.get_params()[0]
    params_em2 = leaf.get_params()[1]

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            leaf = Uniform(Scope([0]), start=0.0, end=1.0)
            leaf_updated = updateBackend(leaf)
            maximum_likelihood_estimation(leaf_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.isclose(leaf_updated.get_params()[0], params, atol=1e-2, rtol=1e-3))
            tc.assertTrue(np.isclose(leaf_updated.get_params()[1], params2, atol=1e-2, rtol=1e-3))

            leaf = Uniform(Scope([0]), start=0.0, end=1.0)
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
