from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.leaves.parametric.log_normal import LogNormal
from spflow.torch.learning.nodes.leaves.parametric.log_normal import maximum_likelihood_estimation, em
from spflow.torch.inference.nodes.leaves.parametric.log_normal import log_likelihood

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
        
        leaf = LogNormal(Scope([0]))

        # simulate data
        data = np.random.lognormal(mean=-1.7, sigma=0.2, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.isclose(leaf.mean, torch.tensor(-1.7), atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.isclose(leaf.std, torch.tensor(0.2), atol=1e-2, rtol=1e-2))
    
    def test_mle_2(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        
        leaf = LogNormal(Scope([0]))

        # simulate data
        data = np.random.lognormal(mean=0.5, sigma=1.3, size=(30000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.isclose(leaf.mean, torch.tensor(0.5), atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.isclose(leaf.std, torch.tensor(1.3), atol=1e-2, rtol=1e-2))
    
    def test_mle_bias_correction(self):

        leaf = LogNormal(Scope([0]))
        data = torch.exp(torch.tensor([[-1.0], [1.0]]))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=False)
        self.assertTrue(torch.isclose(leaf.std, torch.sqrt(torch.tensor(1.0))))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)
        self.assertTrue(torch.isclose(leaf.std, torch.sqrt(torch.tensor(2.0))))

    def test_mle_edge_std_0(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        
        leaf = LogNormal(Scope([0]))

        # simulate data
        data = torch.exp(torch.randn(1, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=False)

        self.assertTrue(torch.isclose(leaf.mean, torch.log(data[0])))
        self.assertTrue(leaf.std > 0)

    def test_mle_edge_std_nan(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        leaf = LogNormal(Scope([0]))

        # simulate data
        data = torch.exp(torch.randn(1, 1))

        # perform MLE (Torch does not throw a warning different to NumPy)
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(torch.isclose(leaf.mean, torch.log(data[0])))
        self.assertFalse(torch.isnan(leaf.std))
        self.assertTrue(leaf.std > 0)

    def test_mle_only_nans(self):
        
        leaf = LogNormal(Scope([0]))

        # simulate data
        data = torch.tensor([[float("nan")], [float("nan")]])

        # check if exception is raised
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, data, nan_strategy='ignore')

    def test_mle_invalid_support(self):

        leaf = LogNormal(Scope([0]))

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("inf")]]), bias_correction=True)

    def test_mle_nan_strategy_none(self):

        leaf = LogNormal(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan")], [0.1], [-1.8], [0.7]]), nan_strategy=None)
    
    def test_mle_nan_strategy_ignore(self):

        leaf = LogNormal(Scope([0]))
        maximum_likelihood_estimation(leaf, torch.exp(torch.tensor([[float("nan")], [0.1], [-1.8], [0.7]])), nan_strategy='ignore', bias_correction=False)
        self.assertTrue(torch.isclose(leaf.mean, torch.tensor(-1.0/3.0)))
        self.assertTrue(torch.isclose(leaf.std, torch.sqrt(1/3*torch.sum((torch.tensor([[0.1], [-1.8], [0.7]])+1.0/3.0)**2))))

    def test_mle_nan_strategy_callable(self):

        leaf = LogNormal(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(leaf, torch.tensor([[0.5], [1]]), nan_strategy=lambda x: x)

    def test_mle_nan_strategy_invalid(self):

        leaf = LogNormal(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]), nan_strategy='invalid_string')
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan")], [1], [0], [1]]), nan_strategy=1)

    # TODO: test weighted MLE

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = LogNormal(Scope([0]))
        data = torch.tensor(np.random.lognormal(mean=-1.7, sigma=0.2, size=(10000, 1)))
        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
        ll.retain_grad()
        ll.sum().backward()

        # perform an em step
        em(leaf, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.isclose(leaf.mean, torch.tensor(-1.7), atol=1e-2, rtol=1e-3))
        self.assertTrue(torch.isclose(leaf.std, torch.tensor(0.2), atol=1e-2, rtol=1e-3))

    def test_em_mixture_of_log_normals(self):
        pass


if __name__ == "__main__":
    unittest.main()