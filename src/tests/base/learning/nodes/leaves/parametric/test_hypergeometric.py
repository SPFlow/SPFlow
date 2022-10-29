from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.hypergeometric import (
    Hypergeometric,
)
from spflow.base.learning.nodes.leaves.parametric.hypergeometric import (
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

        leaf = Hypergeometric(Scope([0]), N=10, M=7, n=3)

        # simulate data
        data = np.random.hypergeometric(
            ngood=7, nbad=10 - 7, nsample=3, size=(10000, 1)
        )

        # perform MLE (should not raise an exception)
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(np.all([leaf.N, leaf.M, leaf.n] == [10, 7, 3]))

    def test_mle_invalid_support(self):

        leaf = Hypergeometric(Scope([0]), N=10, M=7, n=3)

        # perform MLE (should raise exceptions)
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            np.array([[np.inf]]),
            bias_correction=True,
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            np.array([[-1]]),
            bias_correction=True,
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            np.array([[4]]),
            bias_correction=True,
        )


if __name__ == "__main__":
    unittest.main()
