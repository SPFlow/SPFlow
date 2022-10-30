from spflow.meta.data.scope import Scope
from spflow.base.structure.layers.leaves.parametric.hypergeometric import (
    HypergeometricLayer,
)
from spflow.base.learning.layers.leaves.parametric.hypergeometric import (
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

        layer = HypergeometricLayer(
            scope=[Scope([0]), Scope([1])], N=[10, 4], M=[7, 1], n=[5, 3]
        )

        # simulate data
        data = np.hstack(
            [
                np.random.hypergeometric(
                    ngood=7, nbad=10 - 7, nsample=5, size=(10000, 1)
                ),
                np.random.hypergeometric(
                    ngood=1, nbad=4 - 1, nsample=3, size=(10000, 1)
                ),
            ]
        )

        # perform MLE
        maximum_likelihood_estimation(layer, data)

        self.assertTrue(np.all(layer.N == np.array([10, 4])))
        self.assertTrue(np.all(layer.M == np.array([7, 1])))
        self.assertTrue(np.all(layer.n == np.array([5, 3])))


if __name__ == "__main__":
    unittest.main()
