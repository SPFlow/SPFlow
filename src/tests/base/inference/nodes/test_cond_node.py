from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.nodes.cond_node import SPNCondSumNode
from spflow.base.structure.nodes.node import SPNProductNode
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.inference.nodes.cond_node import log_likelihood
from spflow.base.inference.nodes.leaves.parametric.gaussian import (
    log_likelihood,
)
from spflow.base.inference.module import likelihood, log_likelihood
from ...structure.nodes.dummy_node import DummyNode
import numpy as np
import unittest
import random


def create_example_spn():
    spn = SPNCondSumNode(
        children=[
            SPNProductNode(
                children=[
                    Gaussian(Scope([0])),
                    SPNCondSumNode(
                        children=[
                            SPNProductNode(
                                children=[
                                    Gaussian(Scope([1])),
                                    Gaussian(Scope([2])),
                                ]
                            ),
                            SPNProductNode(
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
            SPNProductNode(
                children=[
                    SPNProductNode(
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
