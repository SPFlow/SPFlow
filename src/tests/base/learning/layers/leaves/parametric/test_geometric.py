from spflow.meta.scope.scope import Scope
from spflow.base.structure.layers.leaves.parametric.geometric import GeometricLayer
from spflow.base.learning.layers.leaves.parametric.geometric import maximum_likelihood_estimation

import numpy as np
import unittest
import random


class TestNode(unittest.TestCase):
    def test_mle(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        layer = GeometricLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = np.hstack([np.random.geometric(p=0.3, size=(10000, 1)), np.random.geometric(p=0.7, size=(10000, 1))])

        # perform MLE
        maximum_likelihood_estimation(layer, data)

        self.assertTrue(np.allclose(layer.p, np.array([0.3, 0.7]), atol=1e-2, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()