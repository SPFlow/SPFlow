import unittest

import numpy as np

from spn.algorithms.Validity import is_valid

from spn.algorithms.Inference import likelihood
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear
from spn.algorithms.Gradient import feature_gradient


class TestGradient(unittest.TestCase):
    def test_piecewise_linear_simple(self):
        piecewise_spn = 0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[0]) + 0.5 * PiecewiseLinear(
            [-2, -1, 0], [0, 1, 0], [], scope=[0]
        )
        self.assertTrue(is_valid(piecewise_spn))

        evidence = np.array([[0.5], [1.5], [-0.5], [-1.5]])

        results = feature_gradient(piecewise_spn, evidence)
        expected_results = np.array([[0.5], [-0.5], [-0.5], [0.5]])

        for i, _ in enumerate(evidence):
            self.assertTrue(
                results[i] == expected_results[i],
                "Expected result was {}, but computed result was {}".format(expected_results[i], results[i]),
            )

    def test_piecewise_linear_combined(self):
        piecewise_spn = (
            0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[0])
            + 0.5 * PiecewiseLinear([-2, -1, 0], [0, 1, 0], [], scope=[0])
        ) * (
            0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[1])
            + 0.5 * PiecewiseLinear([-1, 0, 1], [0, 1, 0], [], scope=[1])
        )

        self.assertTrue(is_valid(piecewise_spn))

        evidence = np.array([[0.5, 0], [-0.5, -0.5], [-1.5, 0.5]])
        results = feature_gradient(piecewise_spn, evidence)
        expected_results = np.array([[0.25, 0.125], [-0.125, 0.125], [0.25, 0]])

        self.assertTrue(
            np.all(np.isclose(results, expected_results, atol=0.000001)),
            "Expected result was {}, but computed result was {}".format(expected_results, results),
        )


if __name__ == "__main__":
    unittest.main()
