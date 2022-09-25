from spflow.meta.scope.scope import Scope
from spflow.torch.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric
from spflow.torch.learning.nodes.leaves.parametric.hypergeometric import maximum_likelihood_estimation

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
        
        leaf = Hypergeometric(Scope([0]), N=10, M=7, n=3)

        # simulate data
        data = np.random.hypergeometric(ngood=7, nbad=10-7, nsample=3, size=(10000, 1))

        # perform MLE (should not raise an exception)
        maximum_likelihood_estimation(leaf, torch.tensor(data), bias_correction=True)

        self.assertTrue(torch.all(torch.tensor([leaf.N, leaf.M, leaf.n]) == torch.tensor([10, 7, 3])))

    def test_mle_invalid_support(self):
        
        leaf = Hypergeometric(Scope([0]), N=10, M=7, n=3)

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("inf")]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[-1]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[4]]), bias_correction=True)


if __name__ == "__main__":
    unittest.main()