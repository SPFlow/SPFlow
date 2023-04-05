import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose, tl_vstack

from spflow.tensorly.learning import maximum_likelihood_estimation
from spflow.tensorly.structure.spn import BinomialLayer
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_mle(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        layer = BinomialLayer(scope=[Scope([0]), Scope([1])], n=[3, 10])

        # simulate data
        data = np.hstack(
            [
                np.random.binomial(n=3, p=0.3, size=(10000, 1)),
                np.random.binomial(n=10, p=0.7, size=(10000, 1)),
            ]
        )

        # perform MLE
        maximum_likelihood_estimation(layer, data)

        self.assertTrue(tl_allclose(layer.p, tl.tensor([0.3, 0.7]), atol=1e-2, rtol=1e-3))

    def test_weighted_mle(self):

        leaf = BinomialLayer([Scope([0]), Scope([1])], n=[3, 5])

        data = np.hstack(
            [
                np.vstack(
                    [
                        np.random.binomial(n=3, p=0.8, size=(10000, 1)),
                        np.random.binomial(n=3, p=0.2, size=(10000, 1)),
                    ]
                ),
                np.vstack(
                    [
                        np.random.binomial(n=5, p=0.3, size=(10000, 1)),
                        np.random.binomial(n=5, p=0.7, size=(10000, 1)),
                    ]
                ),
            ]
        )

        weights = tl.concatenate([tl.zeros(10000), tl.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(tl.all(leaf.n == tl.tensor([3, 5])))
        self.assertTrue(tl_allclose(leaf.p, tl.tensor([0.2, 0.7]), atol=1e-3, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
