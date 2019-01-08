"""
Created on December 12, 2018

@author: Alejandro Molina
"""
import unittest
import numpy as np

from spn.algorithms.splitting.PoissonStabilityTest import get_split_cols_poisson_py
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Poisson


class TestIndependence(unittest.TestCase):
    def test_poisson(self):
        np.random.seed(17)
        y = np.concatenate(
            (
                np.random.poisson(5, 1000).reshape(-1, 1),
                np.random.poisson(10, 1000).reshape(-1, 1),
                np.random.poisson(25, 1000).reshape(-1, 1),
                np.random.poisson(35, 1000).reshape(-1, 1),
            ),
            axis=1,
        )

        y = np.concatenate((y, (y[:, 0] + np.random.poisson(0.001, 1000)).reshape(-1, 1)), axis=1)

        test = get_split_cols_poisson_py(alpha=0.3, n_jobs=20)

        result = test(y, Context(parametric_types=[Poisson] * y.shape[1]), list(range(y.shape[1])))

        self.assertEqual(len(result), 4)
        self.assertListEqual(result[0][1], [0, 4])
        self.assertListEqual(result[1][1], [1])
        self.assertListEqual(result[2][1], [2])
        self.assertListEqual(result[3][1], [3])


if __name__ == "__main__":
    unittest.main()
