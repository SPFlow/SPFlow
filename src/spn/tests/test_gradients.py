import unittest

import numpy as np

from spn.algorithms.Validity import is_valid

from spn.algorithms.Inference import likelihood
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear


class TestGradient(unittest.TestCase):
    def redo_test_piecewise_linear_simple(self):
        piecewise_spn = 0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[0]) + 0.5 * PiecewiseLinear(
            [-2, -1, 0], [0, 1, 0], [], scope=[0]
        )
        self.assertTrue(is_valid(piecewise_spn))

        evidence = np.array([[0.5], [1.5], [-0.5], [-1.5]])

        results = gradient_forward(piecewise_spn, evidence)
        expected_results = np.array([[0.5], [-0.5], [-0.5], [0.5]])

        for i, _ in enumerate(evidence):
            self.assertTrue(
                results[i] == expected_results[i],
                "Expected result was {}, but computed result was {}".format(expected_results[i], results[i]),
            )

    def redo_test_piecewise_linear_combined(self):
        piecewise_spn = (
            0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[0])
            + 0.5 * PiecewiseLinear([-2, -1, 0], [0, 1, 0], [], scope=[0])
        ) * (
            0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[1])
            + 0.5 * PiecewiseLinear([-1, 0, 1], [0, 1, 0], [], scope=[1])
        )

        self.assertTrue(is_valid(piecewise_spn))

        evidence = np.array([[0.5, 0], [100, 36], [-0.5, -0.5], [-1.5, 0.5]])
        results = gradient_forward(piecewise_spn, evidence)
        expected_results = np.array([[0.25, 0.125], [0, 0], [-0.125, 0.125], [0.25, 0]])

        for i, _ in enumerate(evidence):
            self.assertTrue(
                np.all(np.equal(results[i], expected_results[i])),
                "Expected result was {}, but computed result was {}".format(expected_results[i], results[i]),
            )


if __name__ == "__main__":
    unittest.main()
