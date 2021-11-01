from spflow.base.structure.nodes import IProductNode, ISumNode, ILeafNode
from spflow.torch.structure.nodes import (
    TorchProductNode,
    TorchSumNode,
    TorchLeafNode,
    TorchGaussian,
    toTorch,
    toNodes,
)
from spflow.torch.inference import log_likelihood
import torch
from spflow.base.structure.nodes.validity_checks import _isvalid_spn
import unittest
import numpy as np


class TestTorchNode(unittest.TestCase):
    def test_spn_fail_weights(self):

        with self.assertRaises(ValueError):
            # creat SPN with (invalid) negative weights
            spn = TorchSumNode(
                [TorchLeafNode(scope=[1]), TorchLeafNode(scope=[2])],
                scope=[1, 2],
                weights=np.array([-0.5, 0.5]),
            )

        with self.assertRaises(ValueError):
            # creat SPN with not enough weights
            spn = TorchSumNode(
                [TorchLeafNode(scope=[1]), TorchLeafNode(scope=[2])],
                scope=[1, 2],
                weights=np.array([0.5]),
            )

        with self.assertRaises(ValueError):
            # creat SPN with too many weights
            spn = TorchSumNode(
                [TorchLeafNode(scope=[1]), TorchLeafNode(scope=[2])],
                scope=[1, 2],
                weights=np.array([0.25, 0.25, 0.5]),
            )

    def test_spn_conversion(self):

        # generate random weights for a sum node with two children
        weights: np.array = np.random.rand(2)
        weights /= weights.sum()

        # Node graph
        graph = ISumNode(
            [
                IProductNode([ILeafNode(scope=[1]), ILeafNode(scope=[2])], scope=[1, 2]),
                IProductNode([ILeafNode(scope=[3])], scope=[3]),
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
    
    def test_sum_node_gradient_update(self):

        # create dummy data
        data = torch.cat([
            torch.normal(mean= 1.0, std=1.0, size=(90,1)),
            torch.normal(mean=-1.0, std=1.0, size=(10,1))
        ])

        # generate random weights for a sum node with two children
        weights: torch.Tensor = torch.rand(2)
        weights /= weights.sum()

        # Node graph
        sum_node = TorchSumNode(
            [
                TorchGaussian(scope=[0], mean=0.0, stdev=1.0),
                TorchGaussian(scope=[0], mean=0.0, stdev=1.0),
            ],
            scope=[0],
            weights=weights,
        )

        # make sure that weights are correctly projected
        self.assertTrue(torch.allclose(weights, sum_node.weights))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(sum_node.parameters(), lr=0.1)

        nll = -log_likelihood(sum_node, data).sum()
        nll.backward()

        # check if gradients are computed
        self.assertTrue(sum_node.weights_aux.grad is not None)

        # update parameters
        optimizer.step()

        # verify that sum node weights are still valid after update
        self.assertTrue(torch.isclose(sum_node.weights.sum(), torch.tensor(1.0)))

    def test_spn_fail_scope(self):
        # based on the corresponding test for Nodes
        invalid_spn = TorchProductNode(
            children=[TorchLeafNode(scope=[1]), TorchLeafNode(scope=[1])], scope=[1, 2]
        )

        # make sure that invalid spn fails the test
        with self.assertRaises(AssertionError):
            _isvalid_spn(toNodes(invalid_spn))

    def test_spn_pass_scope(self):
        # based on the corresponding test for Nodes
        valid_spn = TorchProductNode(
            children=[TorchLeafNode(scope=[1]), TorchLeafNode(scope=[2])], scope=[1, 2]
        )

        # make sure that valid spn passes the test
        _isvalid_spn(toNodes(valid_spn))

    def test_spn_fail_no_children(self):

        with self.assertRaises(ValueError):
            spn = TorchProductNode(
                children=None,
                scope=[1, 2],
            )


if __name__ == "__main__":
    unittest.main()
