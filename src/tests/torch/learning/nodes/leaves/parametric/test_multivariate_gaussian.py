from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.torch.structure.nodes.node import SPNSumNode, SPNProductNode
from spflow.torch.inference.nodes.node import log_likelihood
from spflow.torch.learning.nodes.node import em
from spflow.torch.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian
from spflow.torch.learning.nodes.leaves.parametric.multivariate_gaussian import maximum_likelihood_estimation, em
from spflow.torch.inference.nodes.leaves.parametric.multivariate_gaussian import log_likelihood
from spflow.torch.learning.expectation_maximization.expectation_maximization import expectation_maximization

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
        
        leaf = MultivariateGaussian(Scope([0,1]))

        # simulate data
        data = np.random.multivariate_normal(mean=np.array([-1.7, 0.3]), cov=np.array([[1.0, 0.25], [0.25, 0.5]]), size=(10000,))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.allclose(leaf.mean, torch.tensor([-1.7, 0.3]), atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(leaf.cov, torch.tensor([[1.0, 0.25], [0.25, 0.5]]), atol=1e-2, rtol=1e-2))
    
    def test_mle_2(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        
        leaf = MultivariateGaussian(Scope([0,1]))

        # simulate data
        data = np.random.multivariate_normal(mean=np.array([0.5, 0.2]), cov=np.array([[1.3, -0.7], [-0.7, 1.0]]), size=(10000,))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.allclose(leaf.mean, torch.tensor([0.5, 0.2]), atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(leaf.cov, torch.tensor([[1.3, -0.7], [-0.7, 1.0]]), atol=1e-2, rtol=1e-2))

    def test_mle_bias_correction(self):

        leaf = MultivariateGaussian(Scope([0,1]))
        data = torch.tensor([[-1.0, 1.0], [1.0, 0.5]])

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=False)
        self.assertTrue(torch.allclose(leaf.cov, torch.tensor([[1.0, -0.25], [-0.25, 0.0625]])))

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=True)
        self.assertTrue(torch.allclose(leaf.cov, 2*torch.tensor([[1.0, -0.25], [-0.25, 0.0625]])))

    def test_mle_edge_cov_zero(self):
        
        leaf = MultivariateGaussian(Scope([0,1]))

        # simulate data
        data = torch.tensor([[-1.0, 1.0]])

        # perform MLE
        maximum_likelihood_estimation(leaf, data, bias_correction=False)
        # without bias correction diagonal values are zero and should be set to larger value
        self.assertTrue(torch.all(torch.diag(leaf.cov) > 0))

    def test_mle_only_nans(self):
        
        leaf = MultivariateGaussian(Scope([0,1]))

        # simulate data
        data = torch.tensor([[float("nan"), float("nan")], [float("nan"), float("nan")]])

        # check if exception is raised
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, data, nan_strategy='ignore')

    def test_mle_invalid_support(self):

        leaf = MultivariateGaussian(Scope([0,1]))

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("inf"), 0.0]]), bias_correction=True)

    def test_mle_nan_strategy_none(self):

        leaf = MultivariateGaussian(Scope([0,1]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan"), 0.0], [-2.3, 0.1], [-1.8, 1.9], [0.9, 0.7]]), nan_strategy=None)
    
    def test_mle_nan_strategy_ignore(self):

        leaf = MultivariateGaussian(Scope([0,1]))
        # row of NaN values since partially missing rows are not taken into account by numpy.ma.cov and therefore results in different result
        maximum_likelihood_estimation(leaf, torch.exp(torch.tensor([[float("nan"), float("nan")], [-2.3, 0.1], [-1.8, 1.9], [0.9, 0.7]])), nan_strategy='ignore', bias_correction=False)
        mean_ignore, cov_ignore = leaf.mean, leaf.cov

        maximum_likelihood_estimation(leaf, torch.exp(torch.tensor([[-2.3, 0.1], [-1.8, 1.9], [0.9, 0.7]])), nan_strategy=None, bias_correction=False)
        mean_none, cov_none = leaf.mean, leaf.cov

        self.assertTrue(torch.allclose(mean_ignore, mean_none))
        self.assertTrue(torch.allclose(cov_ignore, cov_none))

    def test_mle_nan_strategy_callable(self):

        leaf = MultivariateGaussian(Scope([0,1]))
        # should not raise an issue
        maximum_likelihood_estimation(leaf, torch.tensor([[0.5, 1.0], [-1.0, 0.0]]), nan_strategy=lambda x: x)

    def test_mle_nan_strategy_invalid(self):

        leaf = MultivariateGaussian(Scope([0,1]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan"), 0.0], [1, 0.1], [1.9, -0.2]]), nan_strategy='invalid_string')
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan"), 0.0], [1, 0.1], [1.9, -0.2]]), nan_strategy=1)

    def test_weighted_mle(self):

        leaf = MultivariateGaussian(Scope([0,1]))

        data = torch.tensor(np.vstack([
            np.random.multivariate_normal([1.7, 2.1], np.eye(2), size=(10000,)),
            np.random.multivariate_normal([0.5, -0.3], np.eye(2), size=(10000,))
        ]))
        weights = torch.concat([
            torch.zeros(10000),
            torch.ones(10000)
        ])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(torch.allclose(leaf.mean, torch.tensor([0.5, -0.3]), atol=1e-2, rtol=1e-1))
        self.assertTrue(torch.allclose(leaf.cov, torch.eye(2), atol=1e-2, rtol=1e-2))

    def test_em_step(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        leaf = MultivariateGaussian(Scope([0, 1]))
        data = torch.tensor(np.random.multivariate_normal(mean=torch.tensor([-1.7, 0.6]), cov=np.array([[0.75, 0.0], [0.0, 1.3]]), size=(10000,)))
        dispatch_ctx = DispatchContext()

        # compute gradients of log-likelihoods w.r.t. module log-likelihoods
        ll = log_likelihood(leaf, data, dispatch_ctx=dispatch_ctx)
        ll.retain_grad()
        ll.sum().backward()

        # perform an em step
        em(leaf, data, dispatch_ctx=dispatch_ctx)

        self.assertTrue(torch.allclose(leaf.mean, torch.tensor([-1.7, 0.6]), atol=1e-2, rtol=1e-1))
        self.assertTrue(torch.allclose(leaf.cov, torch.tensor([[0.75, 0.0], [0.0, 1.3]]), atol=1e-2, rtol=1e-1))

    def test_em_product_of_multivariate_gaussians(self):
        
        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = MultivariateGaussian(Scope([0,1]), mean=[1.5, 0.5], cov=[[0.75, 0], [0, 1.3]])
        l2 = MultivariateGaussian(Scope([2]), mean=[2.5], cov=[[1.5]])
        prod_node = SPNProductNode([l1, l2])

        data = torch.tensor(np.hstack([
            np.random.multivariate_normal(mean=np.ones(2), cov=np.eye(2), size=(20000,)),
            np.random.normal(-2.0, 1.0, size=(20000, 1))
        ]))

        expectation_maximization(prod_node, data, max_steps=10)

        self.assertTrue(torch.allclose(l1.mean, torch.tensor([1.0, 1.0]), atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(l1.cov, torch.eye(2), atol=1e-1, rtol=1e-1))
        self.assertTrue(torch.allclose(l2.mean, torch.tensor([-2.0]), atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(l2.cov, torch.tensor([[1.0]]), atol=1e-2, rtol=1e-1))

    def test_em_mixture_of_multivariate_gaussians(self):
        
        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        l1 = MultivariateGaussian(Scope([0,1]), mean=[1.5, 1.5], cov=[[0.7, 0.0], [0.0, 1.3]])
        l2 = MultivariateGaussian(Scope([0,1]), mean=[-1.5, -1.5], cov=[[1.3, 0.0], [0.0, 0.7]])
        sum_node = SPNSumNode([l1, l2], weights=[0.5, 0.5])

        data = torch.tensor(np.vstack([
            np.random.multivariate_normal(mean=[2.0, 2.0], cov=np.eye(2), size=(15000,)),
            np.random.multivariate_normal(mean=[-2.0, -2.0], cov=np.eye(2), size=(15000,))
        ]))

        expectation_maximization(sum_node, data, max_steps=10)

        self.assertTrue(torch.allclose(l1.mean, torch.tensor([2.0, 2.0]), atol=1e-2, rtol=1e-1))
        self.assertTrue(torch.allclose(l1.cov, torch.eye(2), atol=1e-2, rtol=1e-1))
        self.assertTrue(torch.allclose(l2.mean, torch.tensor([-2.0, -2.0]), atol=1e-2, rtol=1e-1))
        self.assertTrue(torch.allclose(l2.cov, torch.eye(2), atol=1e-2, rtol=1e-1))
        self.assertTrue(torch.allclose(sum_node.weights, torch.tensor([0.5, 0.5]), atol=1e-3, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()