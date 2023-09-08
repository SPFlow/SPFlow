import random
import unittest

import numpy as np
import pytest
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import log_likelihood
from spflow.torch.learning import (
    em,
    expectation_maximization,
    maximum_likelihood_estimation,
)
from spflow.torch.structure.spn import HypergeometricLayer#, ProductNode, SumNode
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.tensorly.utils.helper_functions import tl_toNumpy
from spflow.torch.structure.general.layers.leaves.parametric.hypergeometric import updateBackend


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_mle(self):

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
        maximum_likelihood_estimation(layer, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.all(layer.N == torch.tensor([10, 4])))
        self.assertTrue(torch.all(layer.M == torch.tensor([7, 2])))
        self.assertTrue(torch.all(layer.n == torch.tensor([3, 2])))

    def test_mle_invalid_support(self):

        layer = HypergeometricLayer(Scope([0]), N=10, M=7, n=3)

        # perform MLE (should raise exceptions)
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[float("inf")]]),
            bias_correction=True,
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[-1]]),
            bias_correction=True,
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            layer,
            torch.tensor([[4]]),
            bias_correction=True,
        )

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = HypergeometricLayer([Scope([0]), Scope([1])], N=[10, 6], M=[3, 4], n=[5, 2])
        data = torch.tensor(
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

        self.assertTrue(torch.all(leaf.N == torch.tensor([10, 6])))
        self.assertTrue(torch.all(leaf.M == torch.tensor([3, 4])))
        self.assertTrue(torch.all(leaf.n == torch.tensor([5, 2])))

    def test_em_product_of_hypergeometrics(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = HypergeometricLayer([Scope([0]), Scope([1])], N=[10, 6], M=[3, 4], n=[5, 2])
        prod_node = ProductNode([layer])

        data = torch.tensor(
            np.hstack(
                [
                    np.random.hypergeometric(ngood=3, nbad=7, nsample=3, size=(15000, 1)),
                    np.random.hypergeometric(ngood=4, nbad=2, nsample=2, size=(15000, 1)),
                ]
            )
        )

        expectation_maximization(prod_node, data, max_steps=10)

        self.assertTrue(torch.all(layer.N == torch.tensor([10, 6])))
        self.assertTrue(torch.all(layer.M == torch.tensor([3, 4])))
        self.assertTrue(torch.all(layer.n == torch.tensor([5, 2])))

    def test_em_sum_of_hypergeometrics(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = HypergeometricLayer(Scope([0]), n_nodes=2, N=10, M=3, n=5)
        sum_node = SumNode([layer], weights=[0.5, 0.5])

        data = torch.tensor(np.random.hypergeometric(ngood=3, nbad=7, nsample=3, size=(15000, 1)))

        expectation_maximization(sum_node, data, max_steps=10)

        self.assertTrue(torch.all(layer.N == torch.tensor([10, 10])))
        self.assertTrue(torch.all(layer.M == torch.tensor([3, 3])))
        self.assertTrue(torch.all(layer.n == torch.tensor([5, 5])))

    def test_update_backend(self):
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
            tl.set_backend(backend)
            layer = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=[10, 4], M=[7, 2], n=[3, 2])
            layer_updated = updateBackend(layer)
            maximum_likelihood_estimation(layer_updated, tl.tensor(data))
            # check conversion from torch to python
            M_updated = tl_toNumpy(layer_updated.M)
            N_updated = tl_toNumpy(layer_updated.N)
            n_updated = tl_toNumpy(layer_updated.n)
            self.assertTrue(np.allclose(M, M_updated, atol=1e-2, rtol=1e-1))
            self.assertTrue(np.allclose(N, N_updated, atol=1e-2, rtol=1e-1))
            self.assertTrue(np.allclose(n, n_updated, atol=1e-2, rtol=1e-1))

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
                self.assertTrue(np.allclose(M_em, M_em_updated, atol=1e-2, rtol=1e-1))
                self.assertTrue(np.allclose(N_em, N_em_updated, atol=1e-2, rtol=1e-1))
                self.assertTrue(np.allclose(n_em, n_em_updated, atol=1e-2, rtol=1e-1))


if __name__ == "__main__":
    unittest.main()
