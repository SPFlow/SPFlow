import unittest

import numpy as np
import torch

from spflow.base.structure.spn import CondSumNode as BaseCondSumNode
from spflow.base.structure.spn import Gaussian as BaseGaussian
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.torch.structure.spn import CondSumNode, Gaussian, ProductNode

from ...general.nodes.dummy_node import DummyNode


class TestTorchNode(unittest.TestCase):
    def test_sum_node_initialization(self):

        # empty children
        self.assertRaises(ValueError, CondSumNode, [], [])
        # non-Module children
        self.assertRaises(ValueError, CondSumNode, [DummyNode(Scope([0])), 0])
        # children with different scopes
        self.assertRaises(
            ValueError,
            CondSumNode,
            [DummyNode(Scope([0])), DummyNode(Scope([1]))],
        )

    def test_retrieve_params(self):

        node = CondSumNode([DummyNode(Scope([0])), DummyNode(Scope([0]))])

        # number of child outputs not matching number of weights
        node.set_cond_f(lambda data: {"weights": [1.0]})
        self.assertRaises(
            ValueError,
            node.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )
        # non-positive weights
        node.set_cond_f(lambda data: {"weights": [0.0, 1.0]})
        self.assertRaises(
            ValueError,
            node.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )
        # weights not summing up to one
        node.set_cond_f(lambda data: {"weights": [0.3, 0.5]})
        self.assertRaises(
            ValueError,
            node.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )
        # weights of invalid shape
        node.set_cond_f(lambda data: {"weights": [[0.5, 0.5]]})
        self.assertRaises(
            ValueError,
            node.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        # weights as list of floats
        node.set_cond_f(lambda data: {"weights": [0.5, 0.5]})
        node.retrieve_params(torch.tensor([[1]]), DispatchContext())
        # weights as numpy array
        node.set_cond_f(lambda data: {"weights": np.array([0.5, 0.5])})
        node.retrieve_params(torch.tensor([[1]]), DispatchContext())
        # weights as torch tensor
        node.set_cond_f(lambda data: {"weights": torch.tensor([0.5, 0.5])})
        node.retrieve_params(torch.tensor([[1]]), DispatchContext())

    def test_sum_node_marginalization_1(self):

        s = CondSumNode([DummyNode(Scope([0])), DummyNode(Scope([0]))])

        s_marg = marginalize(s, [1])
        self.assertEqual(s_marg.scopes_out, s.scopes_out)

        s_marg = marginalize(s, [0])
        self.assertEqual(s_marg, None)

    def test_sum_node_marginalization_2(self):

        s = CondSumNode(
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

    def test_backend_conversion_1(self):

        s_torch = CondSumNode([Gaussian(Scope([0])), Gaussian(Scope([0]))])
        s_base = toBase(s_torch)

        self.assertTrue(np.all(s_torch.scopes_out == s_base.scopes_out))

    def test_backend_conversion_2(self):

        s_base = BaseCondSumNode([BaseGaussian(Scope([0])), BaseGaussian(Scope([0]))])
        s_torch = toTorch(s_base)

        self.assertTrue(np.all(s_torch.scopes_out == s_base.scopes_out))


if __name__ == "__main__":
    unittest.main()
