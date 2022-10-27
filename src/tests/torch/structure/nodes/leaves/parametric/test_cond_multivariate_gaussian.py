from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_multivariate_gaussian import CondMultivariateGaussian as BaseCondMultivariateGaussian
from spflow.torch.structure.nodes.node import SPNProductNode
from spflow.torch.structure.nodes.leaves.parametric.cond_multivariate_gaussian import CondMultivariateGaussian, toBase, toTorch, marginalize
from spflow.torch.structure.nodes.leaves.parametric.cond_gaussian import CondGaussian
from typing import Callable

import torch
import numpy as np

import math

import unittest


class TestMultivariateGaussian(unittest.TestCase):
    def test_initialization(self):

        multivariate_gaussian = CondMultivariateGaussian(Scope([0]))
        self.assertTrue(multivariate_gaussian.cond_f is None)
        multivariate_gaussian = CondMultivariateGaussian(Scope([0]), lambda x: {'mean': torch.zeros(2), 'cov': torch.eye(2)})
        self.assertTrue(isinstance(multivariate_gaussian.cond_f, Callable))
        
        # invalid scopes
        self.assertRaises(Exception, CondMultivariateGaussian, Scope([]))

    def test_retrieve_params(self):

        # Valid parameters for Multivariate Gaussian distribution: mean vector in R^k, covariance matrix in R^(k x k) symmetric positive semi-definite

        multivariate_gaussian = CondMultivariateGaussian(Scope([0,1]))

        # mean contains inf and mean contains nan
        multivariate_gaussian.set_cond_f(lambda data: {'mean': torch.tensor([0.0, float('inf')]), 'cov': torch.eye(2)})
        self.assertRaises(Exception, multivariate_gaussian.retrieve_params, torch.tensor([[1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': torch.tensor([-float('inf'), 0.0]), 'cov': torch.eye(2)})
        self.assertRaises(Exception, multivariate_gaussian.retrieve_params, torch.tensor([[1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': torch.tensor([0.0, float('na')]), 'cov': torch.eye(2)})
        self.assertRaises(Exception, multivariate_gaussian.retrieve_params, torch.tensor([[1.0]]), DispatchContext())

        # mean vector of wrong shape
        multivariate_gaussian.set_cond_f(lambda data: {'mean': torch.zeros(3), 'cov': torch.eye(2)})
        self.assertRaises(Exception, multivariate_gaussian.retrieve_params, torch.tensor([[1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': torch.zeros(1,1,2), 'cov': torch.eye(2)})
        self.assertRaises(Exception, multivariate_gaussian.retrieve_params, torch.tensor([[1.0]]), DispatchContext())

        # covariance matrix of wrong shape
        M = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        multivariate_gaussian.set_cond_f(lambda data: {'mean': torch.zeros(2), 'cov': M})
        self.assertRaises(Exception, multivariate_gaussian.retrieve_params, torch.tensor([[1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': torch.zeros(2), 'cov': M.T})
        self.assertRaises(Exception, multivariate_gaussian.retrieve_params, torch.tensor([[1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': torch.zeros(2), 'cov': torch.zeros(3)})
        self.assertRaises(Exception, multivariate_gaussian.retrieve_params, torch.tensor([[1.0]]), DispatchContext())

        # covariance matrix not symmetric positive semi-definite
        multivariate_gaussian.set_cond_f(lambda data: {'mean': torch.zeros(2), 'cov': torch.tensor([[1.0, 0.0], [1.0, 0.0]])})
        self.assertRaises(Exception, multivariate_gaussian.retrieve_params, torch.tensor([[1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': torch.zeros(2), 'cov': -torch.eye(2)})
        self.assertRaises(Exception, multivariate_gaussian.retrieve_params, torch.tensor([[1.0]]), DispatchContext())

        # covariance matrix containing inf or nan
        multivariate_gaussian.set_cond_f(lambda data: {'mean': torch.zeros(2), 'cov': torch.tensor([[float("inf"), 0], [0, float("inf")]])})
        self.assertRaises(Exception, multivariate_gaussian.retrieve_params, torch.tensor([[1.0]]), DispatchContext())
        multivariate_gaussian.set_cond_f(lambda data: {'mean': torch.zeros(2), 'cov': torch.tensor([[float("nan"), 0], [0, float("nan")]])})
        self.assertRaises(Exception, multivariate_gaussian.retrieve_params, torch.tensor([[1.0]]), DispatchContext())
        
        # initialize using lists
        multivariate_gaussian.set_cond_f(lambda data: {'mean': [0.0, 0.0], 'cov': [[1.0, 0.0], [0.0, 1.0]]})
        mean, cov, cov_tril = multivariate_gaussian.retrieve_params(torch.tensor([[1.0]]), DispatchContext())
        self.assertTrue(cov_tril is None)
        self.assertTrue(torch.all(mean == torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.all(cov == torch.tensor([[1.0, 0.0], [0.0, 1.0]])))

        # initialize using numpy arrays
        multivariate_gaussian.set_cond_f(lambda data: {'mean': np.zeros(2), 'cov': np.eye(2)})
        mean, cov, cov_tril = multivariate_gaussian.retrieve_params(torch.tensor([[1.0]]), DispatchContext())
        self.assertTrue(cov_tril is None)
        self.assertTrue(torch.all(mean == torch.zeros(2)))
        self.assertTrue(torch.all(cov == torch.eye(2)))

    def test_structural_marginalization(self):
    
        multivariate_gaussian = CondMultivariateGaussian(Scope([0,1]))

        self.assertTrue(isinstance(marginalize(multivariate_gaussian, [2]), CondMultivariateGaussian))
        self.assertTrue(isinstance(marginalize(multivariate_gaussian, [1]), CondGaussian))
        self.assertTrue(marginalize(multivariate_gaussian, [0,1]) is None)

    def test_base_backend_conversion(self):

        torch_multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1, 2]))
        node_multivariate_gaussian = BaseCondMultivariateGaussian(Scope([0, 1, 2]))
        
        # check conversion from torch to python
        self.assertTrue(
            np.all(torch_multivariate_gaussian.scopes_out == toBase(torch_multivariate_gaussian).scopes_out)
        )
        # check conversion from python to torch
        self.assertTrue(
            np.all(node_multivariate_gaussian.scopes_out == toTorch(node_multivariate_gaussian).scopes_out)
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
