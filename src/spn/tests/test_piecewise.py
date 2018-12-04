import unittest

import numpy as np

from spn.algorithms.Inference import likelihood
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear


class TestGradient(unittest.TestCase):
    def test_piecewise_linear_simple(self):
        piecewise_spn = 0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[0]) + 0.5 * PiecewiseLinear(
            [-2, -1, 0], [0, 1, 0], [], scope=[0]
        )

        evidence = np.array([[-2], [-1.5], [-1], [-0.5], [0], [0.5], [1], [1.5], [2], [3], [-3]])
        results = likelihood(piecewise_spn, evidence)
        expected_results = np.array([[0], [0.25], [0.5], [0.25], [0], [0.25], [0.5], [0.25], [0], [0], [0]])
        self.assertTrue(np.all(np.equal(results, expected_results)))

    def test_piecewise_linear_multiplied(self):
        piecewise_spn = (
            0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[0])
            + 0.5 * PiecewiseLinear([-2, -1, 0], [0, 1, 0], [], scope=[0])
        ) * (
            0.5 * PiecewiseLinear([0, 1, 2], [0, 1, 0], [], scope=[1])
            + 0.5 * PiecewiseLinear([-2, -1, 0], [0, 1, 0], [], scope=[1])
        )

        evidence = np.array(
            [
                [-2, -2],
                [-1.5, -1.5],
                [-1, -1],
                [-0.5, -0.5],
                [0, 0],
                [0.5, 0.5],
                [1, 1],
                [1.5, 1.5],
                [2, 2],
                [3, 3],
                [-3, -3],
                [0, 100],
            ]
        )
        results = likelihood(piecewise_spn, evidence)
        expected_results = np.array([[0], [0.25], [0.5], [0.25], [0], [0.25], [0.5], [0.25], [0], [0], [0], [0]]) ** 2
        self.assertTrue(np.all(np.equal(results, expected_results)))

    def test_piecewise_linear_constant(self):
        piecewise_spn = 0.5 * PiecewiseLinear([1, 2], [1, 1], [], scope=[0]) + 0.5 * PiecewiseLinear(
            [-2, -1], [1, 1], [], scope=[0]
        )

        evidence = np.array([[-3000]])
        results = likelihood(piecewise_spn, evidence)
        expected_results = np.array([[1]])
        self.assertTrue(np.all(np.equal(results, expected_results)))


if __name__ == "__main__":
    unittest.main()
