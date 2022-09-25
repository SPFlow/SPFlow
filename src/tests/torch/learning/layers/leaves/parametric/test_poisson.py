from spflow.meta.scope.scope import Scope
from spflow.torch.structure.layers.leaves.parametric.poisson import PoissonLayer
from spflow.torch.learning.layers.leaves.parametric.poisson import maximum_likelihood_estimation

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
        
        layer = PoissonLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = np.hstack([np.random.poisson(lam=0.3, size=(20000, 1)), np.random.poisson(lam=2.7, size=(20000, 1))])

        # perform MLE
        maximum_likelihood_estimation(layer, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.allclose(layer.l, torch.tensor([0.3, 2.7]), atol=1e-2, rtol=1e-3))

    def test_mle_edge_0(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        
        layer = PoissonLayer(Scope([0]))

        # simulate data
        data = np.random.poisson(lam=1.0, size=(1, 1))

        # perform MLE
        maximum_likelihood_estimation(layer, torch.tensor(data), bias_correction=True)

        self.assertFalse(torch.isnan(layer.l))
        self.assertTrue(torch.all(layer.l > 0.0))

    def test_mle_only_nans(self):
        
        layer = PoissonLayer(Scope([0]))

        # simulate data
        data = torch.tensor([[float("nan"), float("nan")], [float("nan"), 2.0]])

        # check if exception is raised
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, data, nan_strategy='ignore')

    def test_mle_invalid_support(self):
        
        layer = PoissonLayer(Scope([0]))

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("inf")]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[-0.1]]), bias_correction=True)

    def test_mle_nan_strategy_none(self):

        layer = PoissonLayer(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("nan")], [1], [0], [2]]), nan_strategy=None)

    def test_mle_nan_strategy_ignore(self):

        layer = PoissonLayer(Scope([0]))
        maximum_likelihood_estimation(layer, torch.tensor([[float("nan")], [1], [0], [2]]), nan_strategy='ignore', bias_correction=False)
        l_ignore = layer.l

        maximum_likelihood_estimation(layer, torch.tensor([[1], [0], [2]]), nan_strategy='ignore', bias_correction=False)
        l_none = layer.l

        self.assertTrue(torch.isclose(l_ignore, l_none))

    def test_mle_nan_strategy_callable(self):

        layer = PoissonLayer(Scope([0]))
        # should not raise an issue
        maximum_likelihood_estimation(layer, torch.tensor([[2], [1]]), nan_strategy=lambda x: x)

    def test_mle_nan_strategy_invalid(self):

        layer = PoissonLayer(Scope([0]))
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("nan")], [1], [0], [2]]), nan_strategy='invalid_string')
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("nan")], [1], [0], [2]]), nan_strategy=1)


if __name__ == "__main__":
    unittest.main()