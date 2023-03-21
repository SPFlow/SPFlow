import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose, tl_vstack

from spflow.tensorly.learning import maximum_likelihood_estimation
from spflow.tensorly.structure.spn import ExponentialLayer
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_mle(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        layer = ExponentialLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = np.hstack(
            [
                np.random.exponential(scale=1.0 / 0.3, size=(20000, 1)),
                np.random.exponential(scale=1.0 / 2.7, size=(20000, 1)),
            ]
        )

        # perform MLE
        maximum_likelihood_estimation(layer, data)

        self.assertTrue(tl_allclose(layer.l, tl.tensor([0.3, 2.7]), atol=1e-2, rtol=1e-2))

    def test_weighted_mle(self):

        leaf = ExponentialLayer([Scope([0]), Scope([1])])

        data = np.hstack(
            [
                tl_vstack(
                    [
                        np.random.exponential(scale=1.0 / 1.8, size=(10000, 1)),
                        np.random.exponential(scale=1.0 / 0.2, size=(10000, 1)),
                    ]
                ),
                tl_vstack(
                    [
                        np.random.exponential(scale=1.0 / 0.3, size=(10000, 1)),
                        np.random.exponential(scale=1.0 / 1.7, size=(10000, 1)),
                    ]
                ),
            ]
        )
        weights = tl.concatenate([tl.zeros(10000), tl.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)
        self.assertTrue(tl_allclose(leaf.l, tl.tensor([0.2, 1.7]), atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
