import random
import unittest

import numpy as np

from spflow.base.learning import maximum_likelihood_estimation
from spflow.base.structure.spn import LogNormalLayer
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_mle(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        layer = LogNormalLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = np.hstack(
            [
                np.random.lognormal(mean=-1.7, sigma=0.2, size=(20000, 1)),
                np.random.lognormal(mean=0.5, sigma=1.3, size=(20000, 1)),
            ]
        )

        # perform MLE
        maximum_likelihood_estimation(layer, data)

        self.assertTrue(np.allclose(layer.mean, np.array([-1.7, 0.5]), atol=1e-2, rtol=1e-2))
        self.assertTrue(np.allclose(layer.std, np.array([0.2, 1.3]), atol=1e-2, rtol=1e-2))

    def test_weighted_mle(self):

        leaf = LogNormalLayer([Scope([0]), Scope([1])])

        data = np.hstack(
            [
                np.vstack(
                    [
                        np.random.lognormal(1.7, 0.8, size=(10000, 1)),
                        np.random.lognormal(0.5, 1.4, size=(10000, 1)),
                    ]
                ),
                np.vstack(
                    [
                        np.random.lognormal(0.9, 0.3, size=(10000, 1)),
                        np.random.lognormal(1.3, 1.7, size=(10000, 1)),
                    ]
                ),
            ]
        )
        weights = np.concatenate([np.zeros(10000), np.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(np.allclose(leaf.mean, np.array([0.5, 1.3]), atol=1e-2, rtol=1e-1))
        self.assertTrue(np.allclose(leaf.std, np.array([1.4, 1.7]), atol=1e-2, rtol=1e-1))


if __name__ == "__main__":
    unittest.main()
