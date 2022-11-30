import itertools
import unittest

import numpy as np

from spflow.base.inference import log_likelihood
from spflow.base.structure.spn import (
    Gaussian,
    HadamardLayer,
    PartitionLayer,
    ProductLayer,
    ProductNode,
    SumLayer,
    SumNode,
)
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
                SumLayer(
                    n_nodes=3,
                    children=input_nodes,
                    weights=[[0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.2, 0.7, 0.1]],
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

        dummy_data = np.array(
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

        self.assertTrue(np.allclose(layer_ll, nodes_ll))

    def test_product_layer_likelihood(self):

        input_nodes = [
            Gaussian(Scope([0])),
            Gaussian(Scope([1])),
            Gaussian(Scope([2])),
        ]

        layer_spn = SumNode(
            children=[ProductLayer(n_nodes=3, children=input_nodes)],
            weights=[0.3, 0.4, 0.3],
        )

        nodes_spn = SumNode(
            children=[
                ProductNode(children=input_nodes),
                ProductNode(children=input_nodes),
                ProductNode(children=input_nodes),
            ],
            weights=[0.3, 0.4, 0.3],
        )

        dummy_data = np.array(
            [[1.0, 0.25, 0.0], [0.0, 1.0, 0.25], [0.25, 0.0, 1.0]]
        )

        layer_ll = log_likelihood(layer_spn, dummy_data)
        nodes_ll = log_likelihood(nodes_spn, dummy_data)

        self.assertTrue(np.allclose(layer_ll, nodes_ll))

    def test_partition_layer_likelihood(self):

        input_partitions = [
            [Gaussian(Scope([0])), Gaussian(Scope([0]))],
            [Gaussian(Scope([1])), Gaussian(Scope([1])), Gaussian(Scope([1]))],
            [Gaussian(Scope([2]))],
        ]

        layer_spn = SumNode(
            children=[PartitionLayer(child_partitions=input_partitions)],
            weights=[0.2, 0.1, 0.2, 0.2, 0.2, 0.1],
        )

        nodes_spn = SumNode(
            children=[
                ProductNode(
                    children=[
                        input_partitions[0][i],
                        input_partitions[1][j],
                        input_partitions[2][k],
                    ]
                )
                for (i, j, k) in itertools.product([0, 1], [0, 1, 2], [0])
            ],
            weights=[0.2, 0.1, 0.2, 0.2, 0.2, 0.1],
        )

        dummy_data = np.array(
            [[1.0, 0.25, 0.0], [0.0, 1.0, 0.25], [0.25, 0.0, 1.0]]
        )

        layer_ll = log_likelihood(layer_spn, dummy_data)
        nodes_ll = log_likelihood(nodes_spn, dummy_data)

        self.assertTrue(np.allclose(layer_ll, nodes_ll))

    def test_hadamard_layer_likelihood(self):

        input_partitions = [
            [Gaussian(Scope([0]))],
            [Gaussian(Scope([1])), Gaussian(Scope([1])), Gaussian(Scope([1]))],
            [Gaussian(Scope([2]))],
            [Gaussian(Scope([3])), Gaussian(Scope([3])), Gaussian(Scope([3]))],
        ]

        layer_spn = SumNode(
            children=[HadamardLayer(child_partitions=input_partitions)],
            weights=[0.3, 0.2, 0.5],
        )

        nodes_spn = SumNode(
            children=[
                ProductNode(
                    children=[
                        input_partitions[0][i],
                        input_partitions[1][j],
                        input_partitions[2][k],
                        input_partitions[3][l],
                    ]
                )
                for (i, j, k, l) in [[0, 0, 0, 0], [0, 1, 0, 1], [0, 2, 0, 2]]
            ],
            weights=[0.3, 0.2, 0.5],
        )

        dummy_data = np.array(
            [
                [1.0, 0.25, 0.0, -0.7],
                [0.0, 1.0, 0.25, 0.12],
                [0.25, 0.0, 1.0, 0.0],
            ]
        )

        layer_ll = log_likelihood(layer_spn, dummy_data)
        nodes_ll = log_likelihood(nodes_spn, dummy_data)

        self.assertTrue(np.allclose(layer_ll, nodes_ll))


if __name__ == "__main__":
    unittest.main()
