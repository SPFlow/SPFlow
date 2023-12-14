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
from spflow.tensorly.structure.spn import UniformLayer
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.tensorly.utils.helper_functions import tl_toNumpy
from spflow.torch.structure.general.layer.leaf.uniform import updateBackend

tc = unittest.TestCase()

def test_mle(do_for_all_backends):

    layer = UniformLayer(scope=[Scope([0]), Scope([1])], start=[0.0, -5.0], end=[1.0, -2.0])

    # simulate data
    data = tl.tensor([[0.5, -3.0]])

    # perform MLE (should not raise an exception)
    maximum_likelihood_estimation(layer, data, bias_correction=True)

    tc.assertTrue(tl.all(layer.start == tl.tensor([0.0, -5.0])))
    tc.assertTrue(tl.all(layer.end == tl.tensor([1.0, -2.0])))

def test_mle_invalid_support(do_for_all_backends):

    layer = UniformLayer(Scope([0]), start=1.0, end=3.0, support_outside=False)

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
        tl.tensor([[0.0]]),
        bias_correction=True,
    )

def test_em_step(do_for_all_backends):

    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)

    leaf = UniformLayer([Scope([0]), Scope([1])], start=[-1.0, 2.0], end=[3.0, 5.0])
    data = tl.tensor(
        np.hstack(
            [
                np.random.rand(15000, 1) * 4.0 - 1.0,
                np.random.rand(15000, 1) * 3.0 + 2.0,
            ]
        )
    )
    dispatch_ctx = DispatchContext()

    # compute gradients of log-likelihoods w.r.t. module log-likelihoods
    ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
    ll.requires_grad = True
    ll.retain_grad()
    ll.sum().backward()

    # perform an em step
    em(leaf, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(tl.all(leaf.start == tl.tensor([-1.0, 2.0])))
    tc.assertTrue(tl.all(leaf.end == tl.tensor([3.0, 5.0])))

def test_em_product_of_uniforms(do_for_all_backends):

    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = UniformLayer([Scope([0]), Scope([1])], start=[-1.0, 2.0], end=[3.0, 5.0])
    prod_node = ProductNode([layer])

    data = tl.tensor(
        np.hstack(
            [
                np.random.rand(15000, 1) * 4.0 - 1.0,
                np.random.rand(15000, 1) * 3.0 + 2.0,
            ]
        )
    )

    expectation_maximization(prod_node, data, max_steps=10)

    tc.assertTrue(tl.all(layer.start == tl.tensor([-1.0, 2.0])))
    tc.assertTrue(tl.all(layer.end == tl.tensor([3.0, 5.0])))

def test_em_sum_of_uniforms(do_for_all_backends):

    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = UniformLayer(Scope([0]), n_nodes=2, start=-1.0, end=3.0)
    sum_node = SumNode([layer], weights=[0.5, 0.5])

    data = tl.tensor(np.random.rand(15000, 1) * 3.0 + 2.0)

    expectation_maximization(sum_node, data, max_steps=10)

    tc.assertTrue(tl.all(layer.start == tl.tensor([-1.0, -1.0])))
    tc.assertTrue(tl.all(layer.end == tl.tensor([3.0, 3.0])))

def test_update_backend(do_for_all_backends):

    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = UniformLayer(scope=[Scope([0]), Scope([1])], start=[0.0, -5.0], end=[1.0, -2.0])

    # simulate data
    data = tl.tensor([[0.5, -3.0]])

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data))

    start = tl_toNumpy(layer.start)
    end = tl_toNumpy(layer.end)

    layer = UniformLayer(scope=[Scope([0]), Scope([1])], start=[0.0, -5.0], end=[1.0, -2.0])
    prod_node = ProductNode([layer])
    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
    start_em = tl_toNumpy(layer.start)
    end_em = tl_toNumpy(layer.end)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer = UniformLayer(scope=[Scope([0]), Scope([1])], start=[0.0, -5.0], end=[1.0, -2.0])
            layer_updated = updateBackend(layer)
            maximum_likelihood_estimation(layer_updated, tl.tensor(data))
            # check conversion from torch to python
            start_updated = tl_toNumpy(layer_updated.start)
            end_updated = tl_toNumpy(layer_updated.end)
            tc.assertTrue(np.allclose(start, start_updated, atol=1e-2, rtol=1e-1))
            tc.assertTrue(np.allclose(end, end_updated, atol=1e-2, rtol=1e-1))

            layer = UniformLayer(scope=[Scope([0]), Scope([1])], start=[0.0, -5.0], end=[1.0, -2.0])
            layer_updated = updateBackend(layer)
            prod_node = ProductNode([layer_updated])
            if tl.get_backend() != "pytorch":
                with pytest.raises(NotImplementedError):
                    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
            else:
                expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
                start_em_updated = tl_toNumpy(layer_updated.start)
                end_em_updated = tl_toNumpy(layer_updated.end)
                tc.assertTrue(np.allclose(start_em, start_em_updated, atol=1e-2, rtol=1e-1))
                tc.assertTrue(np.allclose(end_em, end_em_updated, atol=1e-2, rtol=1e-1))


def test_change_dtype(do_for_all_backends):
    np.random.seed(0)
    random.seed(0)

    layer = UniformLayer(scope=[Scope([0]), Scope([1])], start=[0.0, -5.0], end=[1.0, -2.0])
    prod_node = ProductNode([layer])

    # simulate data
    data = tl.tensor([[0.5, -3.0]])

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data, dtype=tl.float32))

    layer.to_dtype(tl.float64)

    dummy_data = tl.tensor(data, dtype=tl.float64)
    maximum_likelihood_estimation(layer, dummy_data)

    if do_for_all_backends == "numpy":
        tc.assertRaises(NotImplementedError, expectation_maximization, prod_node, tl.tensor(data, dtype=tl.float64), max_steps=10)
    else:
        # test if em runs without error after dype change
        expectation_maximization(prod_node, tl.tensor(data, dtype=tl.float64), max_steps=10)


def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    np.random.seed(0)
    random.seed(0)

    layer = UniformLayer(scope=[Scope([0]), Scope([1])], start=[0.0, -5.0], end=[1.0, -2.0])
    prod_node = ProductNode([layer])

    # simulate data
    data = np.hstack(
        [
            np.random.hypergeometric(ngood=7, nbad=10 - 7, nsample=3, size=(1000, 1)),
            np.random.hypergeometric(ngood=2, nbad=4 - 2, nsample=2, size=(1000, 1)),
        ]
    )

    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer.to_device, cuda)
        return

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data, dtype=tl.float32))

    tc.assertTrue(layer.start.device.type == "cpu")
    tc.assertTrue(layer.end.device.type == "cpu")

    layer.to_device(cuda)

    dummy_data = tl.tensor(data, dtype=tl.float32, device=cuda)

    # perform MLE
    maximum_likelihood_estimation(layer, dummy_data)
    tc.assertTrue(layer.start.device.type == "cuda")
    tc.assertTrue(layer.end.device.type == "cuda")

    # test if em runs without error after device change
    expectation_maximization(prod_node, tl.tensor(data, dtype=tl.float32, device=cuda), max_steps=10)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
