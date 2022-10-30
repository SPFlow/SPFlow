from spflow.meta.data.scope import Scope
from spflow.base.structure.layers.leaves.parametric.binomial import (
    BinomialLayer,
)
from spflow.base.learning.layers.leaves.parametric.binomial import (
    maximum_likelihood_estimation,
)

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
        data = np.hstack(
            [
                np.random.binomial(n=3, p=0.3, size=(10000, 1)),
                np.random.binomial(n=10, p=0.7, size=(10000, 1)),
            ]
        )

        # perform MLE
        maximum_likelihood_estimation(layer, data)

        self.assertTrue(
            np.allclose(layer.p, np.array([0.3, 0.7]), atol=1e-2, rtol=1e-3)
        )

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

        weights = np.concatenate([np.zeros(10000), np.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(np.all(leaf.n == np.array([3, 5])))
        self.assertTrue(
            np.allclose(leaf.p, np.array([0.2, 0.7]), atol=1e-3, rtol=1e-2)
        )


if __name__ == "__main__":
    unittest.main()
