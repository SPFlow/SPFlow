import unittest

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_isclose, tl_vstack, tl_isnan

from spflow.tensorly.learning import maximum_likelihood_estimation
from spflow.tensorly.structure.spn import Uniform
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_mle(self):

        leaf = Uniform(Scope([0]), start=0.0, end=1.0)

        # simulate data
        data = tl.tensor([[0.5]])

        # perform MLE (should not raise an exception)
        maximum_likelihood_estimation(leaf, data, bias_correction=True)

        self.assertTrue(tl.all([leaf.start, leaf.end] == [0.0, 1.0]))

    def test_mle_invalid_support(self):

        leaf = Uniform(Scope([0]), start=1.0, end=3.0, support_outside=False)

        # perform MLE (should raise exceptions)
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            tl.tensor([[tl.inf]]),
            bias_correction=True,
        )
        self.assertRaises(
            ValueError,
            maximum_likelihood_estimation,
            leaf,
            tl.tensor([[0.0]]),
            bias_correction=True,
        )


if __name__ == "__main__":
    unittest.main()
