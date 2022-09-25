from spflow.meta.scope.scope import Scope
from spflow.torch.structure.layers.leaves.parametric.hypergeometric import HypergeometricLayer
from spflow.torch.learning.layers.leaves.parametric.hypergeometric import maximum_likelihood_estimation

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
        
        layer = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=[10, 4], M=[7, 2], n=[3, 2])

        # simulate data
        data = np.hstack([np.random.hypergeometric(ngood=7, nbad=10-7, nsample=3, size=(1000, 1)), np.random.hypergeometric(ngood=2, nbad=4-2, nsample=2, size=(1000, 1))])

        # perform MLE (should not raise an exception)
        maximum_likelihood_estimation(layer, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.all(layer.N == torch.tensor([10, 4])))
        self.assertTrue(torch.all(layer.M == torch.tensor([7, 2])))
        self.assertTrue(torch.all(layer.n == torch.tensor([3, 2])))

    def test_mle_invalid_support(self):
        
        layer = HypergeometricLayer(Scope([0]), N=10, M=7, n=3)

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("inf")]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[-1]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[4]]), bias_correction=True)


if __name__ == "__main__":
    unittest.main()