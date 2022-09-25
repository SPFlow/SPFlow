from spflow.meta.scope.scope import Scope
from spflow.torch.structure.layers.leaves.parametric.uniform import UniformLayer
from spflow.torch.learning.layers.leaves.parametric.uniform import maximum_likelihood_estimation

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
        
        layer = UniformLayer(scope=[Scope([0]), Scope([1])], start=[0.0, -5.0], end=[1.0, -2.0])

        # simulate data
        data = torch.tensor([[0.5, -3.0]])

        # perform MLE (should not raise an exception)
        maximum_likelihood_estimation(layer, data, bias_correction=True)

        self.assertTrue(torch.all(layer.start == torch.tensor([0.0, -5.0])))
        self.assertTrue(torch.all(layer.end == torch.tensor([1.0, -2.0])))

    def test_mle_invalid_support(self):
        
        layer = UniformLayer(Scope([0]), start=1.0, end=3.0, support_outside=False)

        # perform MLE (should raise exceptions)
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[float("inf")]]), bias_correction=True)
        self.assertRaises(ValueError, maximum_likelihood_estimation, layer, torch.tensor([[0.0]]), bias_correction=True)


if __name__ == "__main__":
    unittest.main()