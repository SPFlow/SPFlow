import unittest

import numpy as np

from spn.algorithms.Validity import is_valid
from spn.algorithms.stats.Expectations import Expectation
from spn.algorithms.stats.Moments import get_mean
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear


class TestGradient(unittest.TestCase):
    def test_piecewise_linear_simple(self):
        piecewise_spn = 0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[0]) + 0.5 * PiecewiseLinear(
            [-2, -1, 0], [0, 1, 0], [], scope=[0]
        )
        self.assertTrue(is_valid(piecewise_spn))

        mean = get_mean(piecewise_spn)
        self.assertAlmostEqual(np.array([[0]]), mean, 5)

    def test_piecewise_linear_combined(self):
        piecewise_spn = (
            0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[0])
            + 0.5 * PiecewiseLinear([-2, -1, 0], [0, 1, 0], [], scope=[0])
        ) * (
            0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[1])
            + 0.5 * PiecewiseLinear([-1, 0, 1], [0, 1, 0], [], scope=[1])
        )

        self.assertTrue(is_valid(piecewise_spn))

        mean = get_mean(piecewise_spn)
        self.assertAlmostEqual(0.0, mean[0, 0], 5)
        self.assertAlmostEqual(0.5, mean[0, 1], 5)

    def test_histogram_combined(self):
        piecewise_spn = (
            0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[0])
            + 0.5 * PiecewiseLinear([-2, -1, 0], [0, 1, 0], [], scope=[0])
        ) * (
            0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[1])
            + 0.5 * PiecewiseLinear([-1, 0, 1], [0, 1, 0], [], scope=[1])
        )

        self.assertTrue(is_valid(piecewise_spn))

        mean = get_mean(piecewise_spn)
        self.assertAlmostEqual(0.0, mean[0, 0], 5)
        self.assertAlmostEqual(0.5, mean[0, 1], 5)


if __name__ == "__main__":
    unittest.main()
