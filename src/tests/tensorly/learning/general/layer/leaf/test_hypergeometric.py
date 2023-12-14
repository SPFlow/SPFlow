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
from spflow.tensorly.structure.spn import HypergeometricLayer#, ProductNode, SumNode
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.tensorly.utils.helper_functions import tl_toNumpy
from spflow.torch.structure.general.layer.leaf.hypergeometric import updateBackend

tc = unittest.TestCase()

def test_mle(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=[10, 4], M=[7, 2], n=[3, 2])

    # simulate data
    data = np.hstack(
        [
            np.random.hypergeometric(ngood=7, nbad=10 - 7, nsample=3, size=(1000, 1)),
            np.random.hypergeometric(ngood=2, nbad=4 - 2, nsample=2, size=(1000, 1)),
        ]
    )

    # perform MLE (should not raise an exception)
    maximum_likelihood_estimation(layer, tl.tensor(data), bias_correction=True)

    tc.assertTrue(tl.all(layer.N == tl.tensor([10, 4])))
    tc.assertTrue(tl.all(layer.M == tl.tensor([7, 2])))
    tc.assertTrue(tl.all(layer.n == tl.tensor([3, 2])))

def test_mle_invalid_support(do_for_all_backends):

    layer = HypergeometricLayer(Scope([0]), N=10, M=7, n=3)

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
        tl.tensor([[-1]]),
        bias_correction=True,
    )
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        tl.tensor([[4]]),
        bias_correction=True,
    )

def test_em_step(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    leaf = HypergeometricLayer([Scope([0]), Scope([1])], N=[10, 6], M=[3, 4], n=[5, 2])
    data = tl.tensor(
        np.hstack(
            [
                np.random.hypergeometric(ngood=3, nbad=7, nsample=3, size=(10000, 1)),
                np.random.hypergeometric(ngood=4, nbad=2, nsample=2, size=(10000, 1)),
            ]
        )
    )

    dispatch_ctx = DispatchContext()

    # compute gradients of log-likelihoods w.r.t. module log-likelihoods
    ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
    ll.requires_grad = True
    ll.sum().backward()

    # perform an em step
    em(leaf, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(tl.all(leaf.N == tl.tensor([10, 6])))
    tc.assertTrue(tl.all(leaf.M == tl.tensor([3, 4])))
    tc.assertTrue(tl.all(leaf.n == tl.tensor([5, 2])))

def test_em_product_of_hypergeometrics(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = HypergeometricLayer([Scope([0]), Scope([1])], N=[10, 6], M=[3, 4], n=[5, 2])
    prod_node = ProductNode([layer])

    data = tl.tensor(
        np.hstack(
            [
                np.random.hypergeometric(ngood=3, nbad=7, nsample=3, size=(15000, 1)),
                np.random.hypergeometric(ngood=4, nbad=2, nsample=2, size=(15000, 1)),
            ]
        )
    )

    expectation_maximization(prod_node, data, max_steps=10)

    tc.assertTrue(tl.all(layer.N == tl.tensor([10, 6])))
    tc.assertTrue(tl.all(layer.M == tl.tensor([3, 4])))
    tc.assertTrue(tl.all(layer.n == tl.tensor([5, 2])))

def test_em_sum_of_hypergeometrics(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = HypergeometricLayer(Scope([0]), n_nodes=2, N=10, M=3, n=5)
    sum_node = SumNode([layer], weights=[0.5, 0.5])

    data = tl.tensor(np.random.hypergeometric(ngood=3, nbad=7, nsample=3, size=(15000, 1)))

    expectation_maximization(sum_node, data, max_steps=10)

    tc.assertTrue(tl.all(layer.N == tl.tensor([10, 10])))
    tc.assertTrue(tl.all(layer.M == tl.tensor([3, 3])))
    tc.assertTrue(tl.all(layer.n == tl.tensor([5, 5])))

def test_update_backend(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=[10, 4], M=[7, 2], n=[3, 2])

    # simulate data
    data = np.hstack(
        [
            np.random.hypergeometric(ngood=7, nbad=10 - 7, nsample=3, size=(1000, 1)),
            np.random.hypergeometric(ngood=2, nbad=4 - 2, nsample=2, size=(1000, 1)),
        ]
    )

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data))

    M = tl_toNumpy(layer.M)
    N = tl_toNumpy(layer.N)
    n = tl_toNumpy(layer.n)

    layer = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=[10, 4], M=[7, 2], n=[3, 2])
    prod_node = ProductNode([layer])
    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
    M_em = tl_toNumpy(layer.M)
    N_em = tl_toNumpy(layer.N)
    n_em = tl_toNumpy(layer.n)


    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=[10, 4], M=[7, 2], n=[3, 2])
            layer_updated = updateBackend(layer)
            maximum_likelihood_estimation(layer_updated, tl.tensor(data))
            # check conversion from torch to python
            M_updated = tl_toNumpy(layer_updated.M)
            N_updated = tl_toNumpy(layer_updated.N)
            n_updated = tl_toNumpy(layer_updated.n)
            tc.assertTrue(np.allclose(M, M_updated, atol=1e-2, rtol=1e-1))
            tc.assertTrue(np.allclose(N, N_updated, atol=1e-2, rtol=1e-1))
            tc.assertTrue(np.allclose(n, n_updated, atol=1e-2, rtol=1e-1))

            layer = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=[10, 4], M=[7, 2], n=[3, 2])
            layer_updated = updateBackend(layer)
            prod_node = ProductNode([layer_updated])
            if tl.get_backend() != "pytorch":
                with pytest.raises(NotImplementedError):
                    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
            else:
                expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
                M_em_updated = tl_toNumpy(layer_updated.M)
                N_em_updated = tl_toNumpy(layer_updated.N)
                n_em_updated = tl_toNumpy(layer_updated.n)
                tc.assertTrue(np.allclose(M_em, M_em_updated, atol=1e-2, rtol=1e-1))
                tc.assertTrue(np.allclose(N_em, N_em_updated, atol=1e-2, rtol=1e-1))
                tc.assertTrue(np.allclose(n_em, n_em_updated, atol=1e-2, rtol=1e-1))

def test_change_dtype(do_for_all_backends):
    np.random.seed(0)
    random.seed(0)

    layer = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=[10, 4], M=[7, 2], n=[3, 2])
    prod_node = ProductNode([layer])

    # simulate data
    data = np.hstack(
        [
            np.random.hypergeometric(ngood=7, nbad=10 - 7, nsample=3, size=(1000, 1)),
            np.random.hypergeometric(ngood=2, nbad=4 - 2, nsample=2, size=(1000, 1)),
        ]
    )

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

    layer = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=[10, 4], M=[7, 2], n=[3, 2])
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

    tc.assertTrue(layer.N.device.type == "cpu")
    tc.assertTrue(layer.M.device.type == "cpu")
    tc.assertTrue(layer.n.device.type == "cpu")

    layer.to_device(cuda)

    dummy_data = tl.tensor(data, dtype=tl.float32, device=cuda)

    # perform MLE
    maximum_likelihood_estimation(layer, dummy_data)
    tc.assertTrue(layer.N.device.type == "cuda")
    tc.assertTrue(layer.M.device.type == "cuda")
    tc.assertTrue(layer.n.device.type == "cuda")

    # test if em runs without error after device change
    expectation_maximization(prod_node, tl.tensor(data, dtype=tl.float32, device=cuda), max_steps=10)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
