from spflow.meta.scope.scope import Scope
from spflow.torch.structure.nodes.leaves.parametric.uniform import Uniform
from spflow.torch.learning.nodes.leaves.parametric.uniform import maximum_likelihood_estimation

import torch
import unittest


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_mle(self):
        
        leaf = Uniform(Scope([0]), start=0.0, end=1.0)

        # simulate data
        data = torch.tensor([[0.5]])

        # perform MLE (should not raise an exception)
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(torch.all(torch.tensor([leaf.start, leaf.end]) == torch.tensor([0.0, 1.0])))

    def test_mle_invalid_support(self):
        
        leaf = Uniform(Scope([0]), start=1.0, end=3.0, support_outside=False)

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[float("inf")]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, leaf, torch.tensor([[0.0]]), bias_correction=True)


if __name__ == "__main__":
    unittest.main()