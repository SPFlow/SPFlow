import unittest

import numpy as np

from spflow.base.inference import likelihood, log_likelihood
from spflow.base.structure.spn import CondSumNode, Gaussian, ProductNode
from spflow.meta.data import Scope


def create_example_spn():
    spn = CondSumNode(
        children=[
            ProductNode(
                children=[
                    Gaussian(Scope([0])),
                    CondSumNode(
                        children=[
                            ProductNode(
                                children=[
                                    Gaussian(Scope([1])),
                                    Gaussian(Scope([2])),
                                ]
                            ),
                            ProductNode(
                                children=[
                                    Gaussian(Scope([1])),
                                    Gaussian(Scope([2])),
                                ]
                            ),
                        ],
                        cond_f=lambda data: {"weights": np.array([0.3, 0.7])},
                    ),
                ],
            ),
            ProductNode(
                children=[
                    ProductNode(
                        children=[
                            Gaussian(Scope([0])),
                            Gaussian(Scope([1])),
                        ]
                    ),
                    Gaussian(Scope([2])),
                ]
            ),
        ],
        cond_f=lambda data: {"weights": np.array([0.4, 0.6])},
    )
    return spn


class TestNode(unittest.TestCase):
    def test_likelihood(self):
        dummy_spn = create_example_spn()
        dummy_data = np.array([[1.0, 0.0, 1.0]])

        l_result = likelihood(dummy_spn, dummy_data)
        ll_result = log_likelihood(dummy_spn, dummy_data)
        self.assertAlmostEqual(l_result[0][0], 0.023358)
        self.assertAlmostEqual(ll_result[0][0], -3.7568156)

    def test_likelihood_marginalization(self):
        spn = create_example_spn()
        dummy_data = np.array([[np.nan, 0.0, 1.0]])

        l_result = likelihood(spn, dummy_data)
        ll_result = log_likelihood(spn, dummy_data)
        self.assertAlmostEqual(l_result[0][0], 0.09653235)
        self.assertAlmostEqual(ll_result[0][0], -2.33787707)


if __name__ == "__main__":
    unittest.main()
