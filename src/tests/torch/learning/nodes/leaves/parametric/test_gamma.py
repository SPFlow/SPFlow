from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.leaves.parametric.gamma import Gamma
from spflow.torch.learning.nodes.leaves.parametric.gamma import maximum_likelihood_estimation, em
from spflow.torch.inference.nodes.leaves.parametric.gamma import log_likelihood

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

    def test_mle_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        leaf = Gamma(Scope([0]))

        # simulate data
        data = np.random.gamma(shape=0.3, scale=1.0/1.7, size=(30000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.isclose(leaf.alpha, torch.tensor(0.3), atol=1e-3, rtol=1e-2))
        self.assertTrue(torch.isclose(leaf.beta, torch.tensor(1.7), atol=1e-3, rtol=1e-2))
    
    def test_mle_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        leaf = Gamma(Scope([0]))

        # simulate data
        data = np.random.gamma(shape=1.9, scale=1.0/0.7, size=(30000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.isclose(leaf.alpha, torch.tensor(1.9), atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.isclose(leaf.beta, torch.tensor(0.7), atol=1e-2, rtol=1e-2))

    def test_mle_only_nans(self):
        
        leaf = Gamma(Scope([0]))

        # simulate data
        data = torch.tensor([[float("nan")], [float("nan")]])

        # check if exception is raised
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, data, nan_strategy='ignore')
    
    def test_mle_invalid_support(self):

        leaf = Gamma(Scope([0]))

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("inf")]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[-0.1]]), bias_correction=True)
    
    def test_mle_nan_strategy_none(self):

        leaf = Gamma(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]), nan_strategy=None)
    
    def test_mle_nan_strategy_ignore(self):

        leaf = Gamma(Scope([0]))
        maximum_likelihood_estimation(leaf, torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]), nan_strategy='ignore', bias_correction=False)
        alpha_ignore, beta_ignore = leaf.alpha, leaf.beta

        # since gamma is estimated iteratively by scipy, just make sure it matches the estimate without nan value
        maximum_likelihood_estimation(leaf, torch.tensor([[0.1], [1.9], [0.7]]), nan_strategy=None, bias_correction=False)
        alpha_none, beta_none = leaf.alpha, leaf.beta

        self.assertTrue(torch.isclose(alpha_ignore, alpha_none))
        self.assertTrue(torch.isclose(beta_ignore, beta_none))

    def test_mle_nan_strategy_callable(self):

        leaf = Gamma(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(leaf, torch.tensor([[0.5], [1]]), nan_strategy=lambda x: x)
    
    def test_mle_nan_strategy_invalid(self):

        leaf = Gamma(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]), nan_strategy='invalid_string')
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan")], [1], [0], [1]]), nan_strategy=1)

    # TODO: test weighted MLE

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Gamma(Scope([0]))
        data = torch.tensor(np.random.gamma(shape=0.3, scale=1.0/1.7, size=(30000, 1)))
        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
        ll.retain_grad()
        ll.sum().backward()

        # perform an em step
        em(leaf, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.isclose(leaf.alpha, torch.tensor(0.3), atol=1e-3, rtol=1e-2))
        self.assertTrue(torch.isclose(leaf.beta, torch.tensor(1.7), atol=1e-3, rtol=1e-2))

    def test_em_mixture_of_gammas(self):
        pass


if __name__ == "__main__":
    unittest.main()