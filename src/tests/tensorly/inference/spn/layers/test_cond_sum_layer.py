import unittest

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.inference.spn.layers.cond_sum_layer import log_likelihood
from spflow.tensorly.structure.spn import CondSumLayer, Gaussian, SumNode
from spflow.meta.data import Scope


class TestNode(unittest.TestCase):
    def test_sum_layer_likelihood(self):

        input_nodes = [
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
        ]

        layer_spn = SumNode(
            children=[
                CondSumLayer(
                    n_nodes=3,
                    children=input_nodes,
                    cond_f=lambda data: {
                        "weights": [
                            [0.8, 0.1, 0.1],
                            [0.2, 0.3, 0.5],
                            [0.2, 0.7, 0.1],
                        ]
                    },
                ),
            ],
            weights=[0.3, 0.4, 0.3],
        )

        nodes_spn = SumNode(
            children=[
                SumNode(children=input_nodes, weights=[0.8, 0.1, 0.1]),
                SumNode(children=input_nodes, weights=[0.2, 0.3, 0.5]),
                SumNode(children=input_nodes, weights=[0.2, 0.7, 0.1]),
            ],
            weights=[0.3, 0.4, 0.3],
        )

        dummy_data = tl.tensor(
            [
                [1.0],
                [
                    0.0,
                ],
                [0.25],
            ]
        )

        layer_ll = log_likelihood(layer_spn, dummy_data)
        nodes_ll = log_likelihood(nodes_spn, dummy_data)

        self.assertTrue(tl_allclose(layer_ll, nodes_ll))


if __name__ == "__main__":
    unittest.main()
