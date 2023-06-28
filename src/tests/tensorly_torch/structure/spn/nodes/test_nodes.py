import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.spn import Gaussian as BaseGaussian
from spflow.base.structure.spn import ProductNode as BaseProductNode
from spflow.base.structure.spn import SumNode as BaseSumNode
from spflow.meta.data import Scope
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.torch.structure.spn import Gaussian

from ...general.nodes.dummy_node import DummyNode


class TestTorchNode(unittest.TestCase):
    def test_sum_node_initialization(self):

        # empty children
        self.assertRaises(ValueError, SumNode, [], [])
        # non-Module children
        self.assertRaises(ValueError, SumNode, [DummyNode(Scope([0])), 0], [0.5, 0.5])
        # children with different scopes
        self.assertRaises(
            ValueError,
            SumNode,
            [DummyNode(Scope([0])), DummyNode(Scope([1]))],
            [0.5, 0.5],
        )
        # number of child outputs not matching number of weights
        self.assertRaises(
            ValueError,
            SumNode,
            [DummyNode(Scope([0])), DummyNode(Scope([0]))],
            [1.0],
        )
        # non-positive weights
        self.assertRaises(ValueError, SumNode, [DummyNode(Scope([0]))], [0.0])
        # weights not summing up to one
        self.assertRaises(
            ValueError,
            SumNode,
            [DummyNode(Scope([0])), DummyNode(Scope([0]))],
            [0.3, 0.5],
        )
        # weights of invalid shape
        self.assertRaises(ValueError, SumNode, [DummyNode(Scope([0]))], [[1.0]])

        # weights as list of floats
        SumNode([DummyNode(Scope([0])), DummyNode(Scope([0]))], tl.tensor([0.5, 0.5]))
        # weights as numpy array
        SumNode([DummyNode(Scope([0])), DummyNode(Scope([0]))], tl.tensor(np.array([0.5, 0.5])))
        # weights as torch tensor
        SumNode(
            [DummyNode(Scope([0])), DummyNode(Scope([0]))],
            tl.tensor(torch.tensor([0.5, 0.5])),
        )
        # no weights
        SumNode([DummyNode(Scope([0])), DummyNode(Scope([0]))])

    def test_sum_node_marginalization_1(self):

        s = SumNode([DummyNode(Scope([0])), DummyNode(Scope([0]))])

        s_marg = marginalize(s, [1])
        self.assertEqual(s_marg.scopes_out, s.scopes_out)

        s_marg = marginalize(s, [0])
        self.assertEqual(s_marg, None)

    def test_sum_node_marginalization_2(self):

        s = SumNode(
            [
                ProductNode([DummyNode(Scope([0])), DummyNode(Scope([1]))]),
                ProductNode([DummyNode(Scope([0])), DummyNode(Scope([1]))]),
            ]
        )

        s_marg = marginalize(s, [0])
        self.assertEqual(s_marg.scopes_out, [Scope([1])])

        s_marg = marginalize(s, [1])
        self.assertEqual(s_marg.scopes_out, [Scope([0])])

        s_marg = marginalize(s, [0, 1])
        self.assertEqual(s_marg, None)

    def test_product_node_initialization(self):

        # empty children
        self.assertRaises(ValueError, ProductNode, [])
        # non-Module children
        self.assertRaises(ValueError, ProductNode, [DummyNode(Scope([0])), 0])
        # children with non-disjoint scopes
        self.assertRaises(
            ValueError,
            ProductNode,
            [DummyNode(Scope([0])), DummyNode(Scope([0]))],
        )

        ProductNode([DummyNode(Scope([0])), DummyNode(Scope([1]))])

    def test_product_node_marginalization_1(self):

        p = ProductNode([DummyNode(Scope([0])), DummyNode(Scope([1]))])

        p_marg = marginalize(p, [2])
        self.assertEqual(p_marg.scopes_out, p.scopes_out)

        p_marg = marginalize(p, [1], prune=False)
        self.assertEqual(p_marg.scopes_out, [Scope([0])])

        p_marg = marginalize(p, [1], prune=True)
        # pruning should return single child directly
        self.assertTrue(isinstance(p_marg, DummyNode))

    def test_product_node_marginalization_2(self):

        p = ProductNode(
            [
                ProductNode([DummyNode(Scope([0])), DummyNode(Scope([1]))]),
                ProductNode([DummyNode(Scope([2])), DummyNode(Scope([3]))]),
            ]
        )

        p_marg = marginalize(p, [2])
        self.assertEqual(p_marg.scopes_out, [Scope([0, 1, 3])])


    def test_backend_conversion(self):

        # generate random weights for a sum nodes
        weights_1: np.array = np.random.rand(2)
        weights_1 /= weights_1.sum()

        weights_2: np.array = np.random.rand(1)
        weights_2 /= weights_2.sum()

        # INode graph
        graph = BaseProductNode(
            [
                BaseSumNode(
                    [BaseGaussian(Scope([0])), BaseGaussian(Scope([0]))],
                    weights=weights_1,
                ),
                BaseSumNode([BaseGaussian(Scope([1]))], weights=weights_2),
            ]
        )

        # conversion to PyTorch graph
        graph_torch = toTorch(graph)

        # check first sum node
        self.assertTrue(
            np.allclose(
                list(graph_torch.children())[0].weights.detach().numpy(),
                graph.children[0].weights,
            )
        )
        # check first gaussian
        self.assertTrue(
            np.allclose(
                list(list(graph_torch.children())[0].children())[0].mean.detach().numpy(),
                graph.children[0].children[0].mean,
            )
        )
        self.assertTrue(
            np.allclose(
                list(list(graph_torch.children())[0].children())[0].std.detach().numpy(),
                graph.children[0].children[0].std,
            )
        )
        # check second gaussian
        self.assertTrue(
            np.allclose(
                list(list(graph_torch.children())[0].children())[0].mean.detach().numpy(),
                graph.children[0].children[0].mean,
            )
        )
        self.assertTrue(
            np.allclose(
                list(list(graph_torch.children())[0].children())[1].std.detach().numpy(),
                graph.children[0].children[1].std,
            )
        )
        # check second sum node
        self.assertTrue(
            np.allclose(
                list(graph_torch.children())[1].weights.detach().numpy(),
                graph.children[1].weights,
            )
        )
        # check third gaussian
        self.assertTrue(
            np.allclose(
                list(list(graph_torch.children())[1].children())[0].mean.detach().numpy(),
                graph.children[1].children[0].mean,
            )
        )
        self.assertTrue(
            np.allclose(
                list(list(graph_torch.children())[1].children())[0].std.detach().numpy(),
                graph.children[1].children[0].std,
            )
        )

        # conversion back to INode representation
        graph_nodes = toBase(graph_torch)

        # check first sum node
        self.assertTrue(np.allclose(graph_nodes.children[0].weights, graph.children[0].weights))
        # check first gaussian
        self.assertTrue(
            np.allclose(
                graph_nodes.children[0].children[0].mean,
                graph.children[0].children[0].mean,
            )
        )
        self.assertTrue(
            np.allclose(
                graph_nodes.children[0].children[0].std,
                graph.children[0].children[0].std,
            )
        )
        # check second gaussian
        self.assertTrue(
            np.allclose(
                graph_nodes.children[0].children[0].mean,
                graph.children[0].children[0].mean,
            )
        )
        self.assertTrue(
            np.allclose(
                graph_nodes.children[0].children[1].std,
                graph.children[0].children[1].std,
            )
        )
        # check second sum node
        self.assertTrue(np.allclose(graph_nodes.children[1].weights, graph.children[1].weights))
        # check third gaussian
        self.assertTrue(
            np.allclose(
                graph_nodes.children[1].children[0].mean,
                graph.children[1].children[0].mean,
            )
        )
        self.assertTrue(
            np.allclose(
                graph_nodes.children[1].children[0].std,
                graph.children[1].children[0].std,
            )
        )


if __name__ == "__main__":
    unittest.main()
