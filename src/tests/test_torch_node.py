from spflow.base.structure.nodes import IProductNode, ISumNode, ILeafNode
from spflow.torch.structure.nodes import (
    TorchProductNode,
    TorchSumNode,
    TorchLeafNode,
    TorchGaussian,
    toTorch,
    toNodes,
    proj_convex_to_real,
    proj_real_to_convex,
)
from spflow.torch.inference.nodes.node import log_likelihood
from spflow.torch.sampling import sample
from spflow.base.structure.nodes.validity_checks import _isvalid_spn
import torch
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

        # INode graph
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

        # conversion back to INode representation
        graph_nodes = toNodes(graph_torch)

        # check whether converted graph matches original graph
        self.assertTrue(graph.equals(graph_nodes))

    def test_projection(self):

        self.assertTrue(
            torch.allclose(proj_real_to_convex(torch.randn(5)).sum(), torch.tensor(1.0))
        )

        weights = torch.rand(5)
        weights /= weights.sum()

        self.assertTrue(torch.allclose(proj_convex_to_real(weights), torch.log(weights)))

    def test_sum_node_initialization(self):

        self.assertRaises(ValueError, TorchSumNode, [], [0, 1], torch.tensor([0.5, 0.5]))

        leaf_1 = TorchLeafNode([0])
        leaf_2 = TorchLeafNode([0])
        # infer weights automatically

        sum_node = TorchSumNode([leaf_1, leaf_2], [0])
        self.assertTrue(torch.allclose(sum_node.weights.sum(), torch.tensor(1.0)))

        weights = torch.rand(2)
        weights /= weights.sum()

        sum_node.weights = weights
        self.assertTrue(torch.allclose(sum_node.weights, weights))

    def test_sum_gradient_optimization(self):

        torch.manual_seed(0)

        # generate random weights for a sum node with two children
        weights = torch.tensor([0.3, 0.7])

        data_1 = torch.randn((70000, 1))
        data_1 = (data_1 - data_1.mean()) / data_1.std() + 5.0
        data_2 = torch.randn((30000, 1))
        data_2 = (data_2 - data_2.mean()) / data_2.std() - 5.0

        data = torch.cat([data_1, data_2])

        # initialize Gaussians
        gaussian_1 = TorchGaussian(scope=[0], mean=5.0, stdev=1.0)
        gaussian_2 = TorchGaussian(scope=[0], mean=-5.0, stdev=1.0)

        # freeze Gaussians
        gaussian_1.requires_grad = False
        gaussian_2.requires_grad = False

        # sum node to be optimized
        sum_node = TorchSumNode(
            [gaussian_1, gaussian_2],
            scope=[0],
            weights=weights,
        )

        # make sure that weights are correctly projected
        self.assertTrue(torch.allclose(weights, sum_node.weights))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(sum_node.parameters(), lr=0.5)

        for i in range(50):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log likelihood
            nll = -log_likelihood(sum_node, data).mean()
            nll.backward()

            if i == 0:
                # check a few general things (just for the first update)

                # check if gradients are computed
                self.assertTrue(sum_node.weights_aux.grad is not None)

                # update parameters
                optimizer.step()

                # verify that sum node weights are still valid after update
                self.assertTrue(torch.isclose(sum_node.weights.sum(), torch.tensor(1.0)))
            else:
                # update parameters
                optimizer.step()

        self.assertTrue(
            torch.allclose(sum_node.weights, torch.tensor([0.7, 0.3]), atol=1e-3, rtol=1e-3)
        )

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

    def test_sum_node_sampling(self):

        l1 = TorchGaussian([0], -5.0, 1.0)
        l2 = TorchGaussian([0], 5.0, 1.0)

        # ----- weights 0, 1 -----

        s = TorchSumNode([l1, l2], [0], weights=[0.0, 1.0])

        samples = sample(s, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(5.0), rtol=0.1))

        # ----- weights 1, 0 -----

        s = TorchSumNode([l1, l2], [0], weights=[1.0, 0.0])

        samples = sample(s, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(-5.0), rtol=0.1))

        # ----- weights 0.2, 0.8 -----

        s = TorchSumNode([l1, l2], [0], weights=[0.2, 0.8])

        samples = sample(s, 1000)
        self.assertTrue(torch.isclose(samples.mean(), torch.tensor(3.0), rtol=0.1))

    def test_product_node_sampling(self):

        l1 = TorchGaussian([0], -5.0, 1.0)
        l2 = TorchGaussian([1], 5.0, 1.0)

        p = TorchProductNode([l1, l2], [0, 1])

        samples = sample(p, 1000)
        self.assertTrue(torch.allclose(samples.mean(dim=0), torch.tensor([-5.0, 5.0]), rtol=0.1))

    def test_sampling(self):

        s = TorchSumNode(
            weights=[0.7, 0.3],
            scope=[0, 1],
            children=[
                TorchSumNode(
                    weights=[0.2, 0.8],
                    scope=[0, 1],
                    children=[
                        TorchProductNode(
                            scope=[0, 1],
                            children=[TorchGaussian([0], -7.0, 1.0), TorchGaussian([1], 7.0, 1.0)],
                        ),
                        TorchProductNode(
                            scope=[0, 1],
                            children=[TorchGaussian([0], -5.0, 1.0), TorchGaussian([1], 5.0, 1.0)],
                        ),
                    ],
                ),
                TorchSumNode(
                    weights=[0.6, 0.4],
                    scope=[0, 1],
                    children=[
                        TorchProductNode(
                            scope=[0, 1],
                            children=[TorchGaussian([0], -3.0, 1.0), TorchGaussian([1], 3.0, 1.0)],
                        ),
                        TorchProductNode(
                            scope=[0, 1],
                            children=[TorchGaussian([0], -1.0, 1.0), TorchGaussian([1], 1.0, 1.0)],
                        ),
                    ],
                ),
            ],
        )

        samples = sample(s, 1000)
        expected_mean = 0.7 * (0.2 * torch.tensor([-7, 7]) + 0.8 * torch.tensor([-5, 5])) + 0.3 * (
            0.6 * torch.tensor([-3, 3]) + 0.4 * torch.tensor([-1, 1])
        )

        self.assertTrue(torch.allclose(samples.mean(dim=0), expected_mean, rtol=0.1))


if __name__ == "__main__":
    unittest.main()
