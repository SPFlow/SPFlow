from spflow.meta.scope.scope import Scope
from spflow.base.structure.layers.leaves.parametric.log_normal import LogNormalLayer
from spflow.base.learning.layers.leaves.parametric.log_normal import maximum_likelihood_estimation

import numpy as np
import unittest
import random


class TestNode(unittest.TestCase):
    def test_mle(self):

        # set seed
        np.random.seed(0)
        random.seed(0)
        
        layer = LogNormalLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = np.hstack([np.random.lognormal(mean=-1.7, sigma=0.2, size=(20000, 1)), np.random.lognormal(mean=0.5, sigma=1.3, size=(20000, 1))])

        # perform MLE
        maximum_likelihood_estimation(layer, data)

        self.assertTrue(np.allclose(layer.mean, np.array([-1.7, 0.5]), atol=1e-2, rtol=1e-2))
        self.assertTrue(np.allclose(layer.std, np.array([0.2, 1.3]), atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()