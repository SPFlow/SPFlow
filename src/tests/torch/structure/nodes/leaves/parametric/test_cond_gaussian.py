from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_gaussian import CondGaussian as BaseCondGaussian
from spflow.torch.structure.nodes.leaves.parametric.cond_gaussian import CondGaussian, toBase, toTorch
from spflow.torch.structure.nodes.node import marginalize
from typing import Callable

import torch
import numpy as np

import random
import unittest


class TestGaussian(unittest.TestCase):
    def test_initialization(self):

        gaussian = CondGaussian(Scope([0]))
        self.assertTrue(gaussian.cond_f is None)
        gaussian = CondGaussian(Scope([0]), lambda x: {'mean': 0.0, 'std': 1.0})
        self.assertTrue(isinstance(gaussian.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondGaussian, Scope([]))
        self.assertRaises(Exception, CondGaussian, Scope([0, 1]))
        self.assertRaises(Exception, CondGaussian, Scope([0],[1]))

    def test_retrieve_params(self):

        # Valid parameters for Gaussian distribution: mean in R, std > 0

        gaussian = CondGaussian(Scope([0]))

        # mean = inf and mean = nan
        gaussian.set_cond_f(lambda data: {'mean': torch.tensor(float('inf')), 'std': torch.tensor(1.0)})
        self.assertRaises(Exception, gaussian.retrieve_params, np.array([[1.0]]), DispatchContext())
        gaussian.set_cond_f(lambda data: {'mean': -torch.tensor(float('inf')), 'std': torch.tensor(1.0)})
        self.assertRaises(Exception, gaussian.retrieve_params, np.array([[1.0]]), DispatchContext())
        gaussian.set_cond_f(lambda data: {'mean': torch.tensor(float('nan')), 'std': torch.tensor(1.0)})
        self.assertRaises(Exception, gaussian.retrieve_params, np.array([[1.0]]), DispatchContext())

        # std = 0 and std < 0
        gaussian.set_cond_f(lambda data: {'mean': torch.tensor(0.0), 'std': torch.tensor(0.0)})
        self.assertRaises(Exception, gaussian.retrieve_params, np.array([[1.0]]), DispatchContext())
        gaussian.set_cond_f(lambda data: {'mean': torch.tensor(0.0), 'std': torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))})
        self.assertRaises(Exception, gaussian.retrieve_params, np.array([[1.0]]), DispatchContext())

        # std = inf and std = nan
        gaussian.set_cond_f(lambda data: {'mean': torch.tensor(0.0), 'std': -torch.tensor(float('inf'))})
        self.assertRaises(Exception, gaussian.retrieve_params, np.array([[1.0]]), DispatchContext())
        gaussian.set_cond_f(lambda data: {'mean': torch.tensor(0.0), 'std': torch.tensor(float('nan'))})
        self.assertRaises(Exception, gaussian.retrieve_params, np.array([[1.0]]), DispatchContext())

    def test_structural_marginalization(self):
        
        gaussian = CondGaussian(Scope([0]))

        self.assertTrue(marginalize(gaussian, [1]) is not None)
        self.assertTrue(marginalize(gaussian, [0]) is None)
    
    def test_base_backend_conversion(self):

        mean = random.random()
        std = random.random() + 1e-7  # offset by small number to avoid zero

        torch_gaussian = CondGaussian(Scope([0]))
        node_gaussian = BaseCondGaussian(Scope([0]))

        # check conversion from torch to python
        self.assertTrue(
            np.all(torch_gaussian.scopes_out == toBase(torch_gaussian).scopes_out)
        )
        # check conversion from python to torch
        self.assertTrue(
            np.all(node_gaussian.scopes_out == toTorch(node_gaussian).scopes_out)
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
