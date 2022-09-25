from spflow.meta.scope.scope import Scope
from spflow.torch.structure.layers.leaves.parametric.exponential import ExponentialLayer
from spflow.torch.learning.layers.leaves.parametric.exponential import maximum_likelihood_estimation

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
        
        layer = ExponentialLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = np.hstack([np.random.exponential(scale=1.0/0.3, size=(20000, 1)), np.random.exponential(scale=1.0/2.7, size=(20000, 1))])

        # perform MLE
        maximum_likelihood_estimation(layer, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.allclose(layer.l, torch.tensor([0.3, 2.7]), atol=1e-2, rtol=1e-3))
    
    def test_mle_bias_correction(self):

        layer = ExponentialLayer(Scope([0]))
        data = torch.tensor([[0.3], [2.7]])

        # perform MLE
        maximum_likelihood_estimation(layer, data, bias_correction=False)
        self.assertTrue(torch.isclose(layer.l, torch.tensor(2.0/3.0)))

        # perform MLE
        maximum_likelihood_estimation(layer, data, bias_correction=True)
        self.assertTrue(torch.isclose(layer.l, torch.tensor(1.0/3.0)))
    
    def test_mle_edge_0(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = ExponentialLayer(Scope([0]))

        # simulate data
        data = np.random.exponential(scale=1.0, size=(1, 1))

        # perform MLE (bias correction leads to zero result)
        maximum_likelihood_estimation(layer, torch.tensor(data), bias_correction=True)

        self.assertFalse(torch.isnan(layer.l))
        self.assertTrue(torch.all(layer.l > 0.0))
    
    def test_mle_only_nans(self):
        
        layer = ExponentialLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = torch.tensor([[float("nan"), float("nan")], [float("nan"), 0.5]])

        # check if exception is raised
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, data, nan_strategy='ignore')
    
    def test_mle_invalid_support(self):

        layer = ExponentialLayer(Scope([0]))

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("nan")]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[-0.1]]), bias_correction=True)
    
    def test_mle_nan_strategy_none(self):

        layer = ExponentialLayer(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]), nan_strategy=None)
    
    def test_mle_nan_strategy_ignore(self):

        layer = ExponentialLayer(Scope([0]))
        maximum_likelihood_estimation(layer, torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]), nan_strategy='ignore', bias_correction=False)
        self.assertTrue(torch.isclose(layer.l, torch.tensor(3.0/2.7)))

    def test_mle_nan_strategy_callable(self):

        layer = ExponentialLayer(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(layer, torch.tensor([[0.5], [1]]), nan_strategy=lambda x: x)

    def test_mle_nan_strategy_invalid(self):

        layer = ExponentialLayer(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("nan")], [0.1], [1.9], [0.7]]), nan_strategy='invalid_string')
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("nan")], [1], [0], [1]]), nan_strategy=1)


if __name__ == "__main__":
    unittest.main()