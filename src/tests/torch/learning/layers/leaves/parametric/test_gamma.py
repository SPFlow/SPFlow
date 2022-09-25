from spflow.meta.scope.scope import Scope
from spflow.torch.structure.layers.leaves.parametric.gamma import GammaLayer
from spflow.torch.learning.layers.leaves.parametric.gamma import maximum_likelihood_estimation

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
        np.random.seed(0)
        random.seed(0)
        
        layer = GammaLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = np.hstack([np.random.gamma(shape=0.3, scale=1.0/1.7, size=(50000, 1)), np.random.gamma(shape=1.9, scale=1.0/0.7, size=(50000, 1))])

        # perform MLE
        maximum_likelihood_estimation(layer, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.allclose(layer.alpha, torch.tensor([0.3, 1.9]), atol=1e-3, rtol=1e-2))
        self.assertTrue(torch.allclose(layer.beta, torch.tensor([1.7, 0.7]), atol=1e-3, rtol=1e-2))
    
    def test_mle_only_nans(self):
        
        layer = GammaLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = torch.tensor([[float("nan"), float("nan")], [float("nan"), 0.5]])

        # check if exception is raised
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, data, nan_strategy='ignore')
    
    def test_mle_invalid_support(self):

        layer = GammaLayer(Scope([0]))

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("inf")]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[-0.1]]), bias_correction=True)
    
    def test_mle_nan_strategy_none(self):

        layer = GammaLayer(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]), nan_strategy=None)
    
    def test_mle_nan_strategy_ignore(self):

        layer = GammaLayer(Scope([0]))
        maximum_likelihood_estimation(layer, torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]), nan_strategy='ignore', bias_correction=False)
        alpha_ignore, beta_ignore = layer.alpha, layer.beta

        # since gamma is estimated iteratively by scipy, just make sure it matches the estimate without nan value
        maximum_likelihood_estimation(layer, torch.tensor([[0.1], [1.9], [0.7]]), nan_strategy=None, bias_correction=False)
        alpha_none, beta_none = layer.alpha, layer.beta

        self.assertTrue(torch.allclose(alpha_ignore, alpha_none))
        self.assertTrue(torch.allclose(beta_ignore, beta_none))
    
    def test_mle_nan_strategy_callable(self):

        layer = GammaLayer(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(layer, torch.tensor([[0.5], [1]]), nan_strategy=lambda x: x)

    def test_mle_nan_strategy_invalid(self):

        layer = GammaLayer(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]), nan_strategy='invalid_string')
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("nan")], [1], [0], [1]]), nan_strategy=1)


if __name__ == "__main__":
    unittest.main()