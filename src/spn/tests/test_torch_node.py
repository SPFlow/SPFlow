from spn.base.nodes.node import ProductNode, SumNode, LeafNode
from spn.backend.pytorch.nodes.node import (
    TorchProductNode,
    TorchSumNode,
    TorchLeafNode,
    toTorch,
    toNodes,
)
from spn.base.nodes.validity_checks import _isvalid_spn
import unittest
import numpy as np


class TestTorchNode(unittest.TestCase):
    def test_spn_fail_weights(self):

        with self.assertRaises(ValueError):
            # creat SPN with (invalid) negative weights
            spn: TorchNode = TorchSumNode(
                [TorchLeafNode(scope=[1]), TorchLeafNode(scope=[2])],
                scope=[1, 2],
                weights=np.array([-0.5, 0.5]),
            )

        with self.assertRaises(ValueError):
            # creat SPN with not enough weights
            spn: TorchNode = TorchSumNode(
                [TorchLeafNode(scope=[1]), TorchLeafNode(scope=[2])],
                scope=[1, 2],
                weights=np.array([0.5]),
            )

        with self.assertRaises(ValueError):
            # creat SPN with too many weights
            spn: TorchNode = TorchSumNode(
                [TorchLeafNode(scope=[1]), TorchLeafNode(scope=[2])],
                scope=[1, 2],
                weights=np.array([0.25, 0.25, 0.5]),
            )

    def test_spn_conversion(self):

        # generate random weights for a sum node with two children
        weights: np.array = np.random.rand(2)
        weights /= weights.sum()

        # Node graph
        graph: TorchNode = SumNode(
            [
                ProductNode([LeafNode(scope=[1]), LeafNode(scope=[2])], scope=[1, 2]),
                ProductNode([LeafNode(scope=[3])], scope=[3]),
            ],
            scope=[1, 2, 3],
            weights=weights,
        )

        # conversion to PyTorch graph
        graph_torch = toTorch(graph)

        # conversion back to Node representation
        graph_nodes = toNodes(graph_torch)

        # check whether converted graph matches original graph
        self.assertTrue(graph.equals(graph_nodes))

    def test_spn_fail_scope(self):
        # based on the corresponding test for Nodes
        invalid_spn: TorchNode = TorchProductNode(
            children=[TorchLeafNode(scope=[1]), TorchLeafNode(scope=[1])], scope=[1, 2]
        )

        # make sure that invalid spn fails the test
        with self.assertRaises(AssertionError):
            _isvalid_spn(toNodes(invalid_spn))

    def test_spn_pass_scope(self):
        # based on the corresponding test for Nodes
        valid_spn: TorchNode = TorchProductNode(
            children=[TorchLeafNode(scope=[1]), TorchLeafNode(scope=[2])], scope=[1, 2]
        )

        # make sure that valid spn passes the test
        _isvalid_spn(toNodes(valid_spn))

    def test_spn_fail_no_children(self):

        with self.assertRaises(ValueError):
            spn: TorchNode = TorchProductNode(
                children=None,
                scope=[1, 2],
            )


if __name__ == "__main__":
    unittest.main()
