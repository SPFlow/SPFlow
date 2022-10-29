from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.uniform import Uniform
from spflow.base.learning.nodes.leaves.parametric.uniform import (
    maximum_likelihood_estimation,
)

import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_mle(self):

        leaf = Uniform(Scope([0]), start=0.0, end=1.0)

        # simulate data
        data = np.array([[0.5]])

        # perform MLE (should not raise an exception)
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(np.all([leaf.start, leaf.end] == [0.0, 1.0]))

    def test_mle_invalid_support(self):

        leaf = Uniform(Scope([0]), start=1.0, end=3.0, support_outside=False)

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
            np.array([[0.0]]),
            bias_correction=True,
        )


if __name__ == "__main__":
    unittest.main()
