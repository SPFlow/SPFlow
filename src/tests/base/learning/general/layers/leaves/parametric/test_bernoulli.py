from spflow.meta.data import Scope
from spflow.base.structure.spn import (
    BernoulliLayer,
)
from spflow.base.learning import (
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

        layer = BernoulliLayer(scope=[Scope([0]), Scope([1])])

        # simulate data
        data = np.hstack(
            [
                np.random.binomial(n=1, p=0.3, size=(10000, 1)),
                np.random.binomial(n=1, p=0.7, size=(10000, 1)),
            ]
        )

        # perform MLE
        maximum_likelihood_estimation(layer, data)

        self.assertTrue(
            np.allclose(layer.p, np.array([0.3, 0.7]), atol=1e-2, rtol=1e-3)
        )

    def test_weighted_mle(self):

        leaf = BernoulliLayer([Scope([0]), Scope([1])], n_nodes=3)

        data = np.hstack(
            [
                np.vstack(
                    [
                        np.random.binomial(n=1, p=0.8, size=(10000, 1)),
                        np.random.binomial(n=1, p=0.2, size=(10000, 1)),
                    ]
                ),
                np.vstack(
                    [
                        np.random.binomial(n=1, p=0.3, size=(10000, 1)),
                        np.random.binomial(n=1, p=0.7, size=(10000, 1)),
                    ]
                ),
            ]
        )

        weights = np.concatenate([np.zeros(10000), np.ones(10000)])

        maximum_likelihood_estimation(leaf, data, weights)

        self.assertTrue(
            np.allclose(leaf.p, np.array([0.2, 0.7]), atol=1e-3, rtol=1e-2)
        )


if __name__ == "__main__":
    unittest.main()