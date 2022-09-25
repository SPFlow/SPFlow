from spflow.meta.scope.scope import Scope
from spflow.torch.structure.nodes.leaves.parametric.binomial import Binomial
from spflow.torch.learning.nodes.leaves.parametric.binomial import maximum_likelihood_estimation

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
        
        leaf = Binomial(Scope([0]), n=3)

        # simulate data
        data = np.random.binomial(n=3, p=0.3, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data))

        self.assertTrue(torch.isclose(leaf.p, torch.tensor(0.3), atol=1e-2, rtol=1e-3))

    def test_mle_2(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        
        leaf = Binomial(Scope([0]), n=10)

        # simulate data
        data = np.random.binomial(n=10, p=0.7, size=(10000, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data))

        self.assertTrue(torch.isclose(leaf.p, torch.tensor(0.7), atol=1e-2, rtol=1e-3))

    def test_mle_edge_0(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        
        leaf = Binomial(Scope([0]), n=3)

        # simulate data
        data = np.random.binomial(n=3, p=0.0, size=(100, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data))

        self.assertTrue(leaf.p > 0.0)

    def test_mle_edge_1(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        
        leaf = Binomial(Scope([0]), n=5)

        # simulate data
        data = np.random.binomial(n=5, p=1.0, size=(100, 1))

        # perform MLE
        maximum_likelihood_estimation(leaf, torch.tensor(data))

        self.assertTrue(leaf.p < 1.0)
    
    def test_mle_only_nans(self):

        leaf = Binomial(Scope([0]), n=3)

        # simulate data
        data = torch.tensor([[float("nan")], [float("nan")]])

        # check if exception is raised
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, data, nan_strategy='ignore')

    def test_mle_invalid_support(self):
        
        leaf = Binomial(Scope([0]), n=3)

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("inf")]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[-0.1]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[4]]), bias_correction=True)

    def test_mle_nan_strategy_none(self):

        leaf = Binomial(Scope([0]), n=2)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan")], [1], [2], [1]]), nan_strategy=None)
    
    def test_mle_nan_strategy_ignore(self):

        leaf = Binomial(Scope([0]), n=2)
        maximum_likelihood_estimation(leaf, torch.tensor([[float("nan")], [1], [2], [1]]), nan_strategy='ignore')
        self.assertTrue(torch.isclose(leaf.p, torch.tensor(4.0/6.0)))

    def test_mle_nan_strategy_callable(self):

        leaf = Binomial(Scope([0]), n=2)
        # should not raise an issue
        maximum_likelihood_estimation(leaf, torch.tensor([[1.0], [0.0], [1.0]]), nan_strategy=lambda x: x)

    def test_mle_nan_strategy_invalid(self):

        leaf = Binomial(Scope([0]), n=2)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan")], [1], [0], [1]]), nan_strategy='invalid_string')
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("nan")], [1], [0], [1]]), nan_strategy=1)


if __name__ == "__main__":
    unittest.main()