from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric
from spflow.torch.learning.nodes.leaves.parametric.hypergeometric import maximum_likelihood_estimation, em
from spflow.torch.inference.nodes.leaves.parametric.hypergeometric import log_likelihood

import torch
import numpy as np
import unittest
import random


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
        
        leaf = Hypergeometric(Scope([0]), N=10, M=7, n=3)

        # simulate data
        data = np.random.hypergeometric(ngood=7, nbad=10-7, nsample=3, size=(10000, 1))

        # perform MLE (should not raise an exception)
        maximum_likelihood_estimation(leaf, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.all(torch.tensor([leaf.N, leaf.M, leaf.n]) == torch.tensor([10, 7, 3])))

    def test_mle_invalid_support(self):
        
        leaf = Hypergeometric(Scope([0]), N=10, M=7, n=3)

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("inf")]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[-1]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[4]]), bias_correction=True)

    # TODO: test weighted MLE

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Hypergeometric(Scope([0]), N=10, M=7, n=3)
        data = torch.tensor(np.random.hypergeometric(ngood=7, nbad=10-7, nsample=3, size=(10000, 1)))
        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
        ll.requires_grad = True
        ll.retain_grad()
        ll.sum().backward()

        # perform an em step
        em(leaf, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.all(torch.tensor([leaf.N, leaf.M, leaf.n]) == torch.tensor([10, 7, 3])))

    def test_em_mixture_of_hypergeometrics(self):
        pass


if __name__ == "__main__":
    unittest.main()