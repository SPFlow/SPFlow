from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.leaves.parametric.geometric import Geometric
from spflow.torch.learning.nodes.leaves.parametric.geometric import maximum_likelihood_estimation, em
from spflow.torch.inference.nodes.leaves.parametric.geometric import log_likelihood

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
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        
        leaf = Geometric(Scope([0]))

        # simulate data
        data = np.random.geometric(p=0.3, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.isclose(leaf.p, torch.tensor(0.3), atol=1e-2, rtol=1e-2))

    def test_mle_2(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        
        leaf = Geometric(Scope([0]))

        # simulate data
        data = np.random.geometric(p=0.7, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.isclose(leaf.p, torch.tensor(0.7), atol=1e-2, rtol=1e-2))
    
    def test_mle_bias_correction(self):

        leaf = Geometric(Scope([0]))
        data = torch.tensor([[1.0], [3.0]])

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=False)
        self.assertTrue(torch.isclose(leaf.p, torch.tensor(2.0/4.0)))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)
        self.assertTrue(torch.isclose(leaf.p, torch.tensor(1.0/4.0)))

    def test_mle_edge_1(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        
        leaf = Geometric(Scope([0]))

        # simulate data
        data = torch.ones(100, 1)

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(leaf.p < 1.0)
    
    def test_mle_only_nans(self):
        
        leaf = Geometric(Scope([0]))

        # simulate data
        data = torch.tensor([[float("nan")], [float("nan")]])

        # check if exception is raised
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, data, nan_strategy='ignore')

    def test_mle_invalid_support(self):
        
        leaf = Geometric(Scope([0]))

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("inf")]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[0]]), bias_correction=True)

    def test_mle_nan_strategy_none(self):

        leaf = Geometric(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan")], [1], [4], [3]]), nan_strategy=None)
    
    def test_mle_nan_strategy_ignore(self):

        leaf = Geometric(Scope([0]))
        maximum_likelihood_estimation(leaf, torch.tensor([[float("nan")], [1], [4], [3]]), nan_strategy='ignore', bias_correction=False)
        self.assertTrue(torch.isclose(leaf.p, torch.tensor(3.0/8.0)))

    def test_mle_nan_strategy_callable(self):

        leaf = Geometric(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(leaf, torch.tensor([[2], [1]]), nan_strategy=lambda x: x)

    def test_mle_nan_strategy_invalid(self):

        leaf = Geometric(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan")], [1], [4], [3]]), nan_strategy='invalid_string')
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan")], [1], [0], [1]]), nan_strategy=1)

    # TODO: test weighted MLE

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = Geometric(Scope([0]))
        data = torch.tensor(np.random.geometric(p=0.3, size=(10000, 1)))
        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
        ll.retain_grad()
        ll.sum().backward()

        # perform an em step
        em(leaf, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.isclose(leaf.p, torch.tensor(0.3), atol=1e-2, rtol=1e-3))

    def test_em_mixture_of_geometrics(self):
        pass


if __name__ == "__main__":
    unittest.main()