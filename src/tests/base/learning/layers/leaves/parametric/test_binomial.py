from spflow.meta.scope.scope import Scope
from spflow.base.structure.layers.leaves.parametric.binomial import BinomialLayer
from spflow.base.learning.layers.leaves.parametric.binomial import maximum_likelihood_estimation

import numpy as np
import unittest
import random


class TestNode(unittest.TestCase):
    def test_mle(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        layer = BinomialLayer(scope=[Scope([0]), Scope([1])], n=[3, 10])

        # simulate data
        data = np.hstack([np.random.binomial(n=3, p=0.3, size=(10000, 1)), np.random.binomial(n=10, p=0.7, size=(10000, 1))])

        # perform MLE
        maximum_likelihood_estimation(layer, data)

        self.assertTrue(np.allclose(layer.p, np.array([0.3, 0.7]), atol=1e-2, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()