from spflow.torch.structure.nodes.cond_node import (
    SPNCondSumNode,
    marginalize,
    toBase,
    torch,
)
from spflow.torch.structure.nodes.node import (
    SPNProductNode,
    marginalize,
    toBase,
    toTorch,
)
from spflow.base.structure.nodes.cond_node import (
    SPNCondSumNode as BaseSPNCondSumNode,
)
from spflow.torch.structure.nodes.leaves.parametric.gaussian import (
    Gaussian,
    toBase,
    toTorch,
)
from spflow.base.structure.nodes.leaves.parametric.gaussian import (
    Gaussian as BaseGaussian,
)
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.meta.data.scope import Scope
from .dummy_node import DummyNode
import numpy as np
import torch
import unittest
import random


class TestTorchNode(unittest.TestCase):
    def test_sum_node_initialization(self):

        # empty children
        self.assertRaises(ValueError, SPNCondSumNode, [], [])
        # non-Module children
        self.assertRaises(
            ValueError, SPNCondSumNode, [DummyNode(Scope([0])), 0]
        )
        # children with different scopes
        self.assertRaises(
            ValueError,
            SPNCondSumNode,
            [DummyNode(Scope([0])), DummyNode(Scope([1]))],
        )

    def test_retrieve_params(self):

        node = SPNCondSumNode([DummyNode(Scope([0])), DummyNode(Scope([0]))])

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

        s = SPNCondSumNode([DummyNode(Scope([0])), DummyNode(Scope([0]))])

        s_marg = marginalize(s, [1])
        self.assertEqual(s_marg.scopes_out, s.scopes_out)

        s_marg = marginalize(s, [0])
        self.assertEqual(s_marg, None)

    def test_sum_node_marginalization_2(self):

        s = SPNCondSumNode(
            [
                SPNProductNode([DummyNode(Scope([0])), DummyNode(Scope([1]))]),
                SPNProductNode([DummyNode(Scope([0])), DummyNode(Scope([1]))]),
            ]
        )

        s_marg = marginalize(s, [0])
        self.assertEqual(s_marg.scopes_out, [Scope([1])])

        s_marg = marginalize(s, [1])
        self.assertEqual(s_marg.scopes_out, [Scope([0])])

        s_marg = marginalize(s, [0, 1])
        self.assertEqual(s_marg, None)

    def test_backend_conversion_1(self):

        s_torch = SPNCondSumNode([Gaussian(Scope([0])), Gaussian(Scope([0]))])
        s_base = toBase(s_torch)

        self.assertTrue(np.all(s_torch.scopes_out == s_base.scopes_out))

    def test_backend_conversion_2(self):

        s_base = BaseSPNCondSumNode(
            [BaseGaussian(Scope([0])), BaseGaussian(Scope([0]))]
        )
        s_torch = toTorch(s_base)

        self.assertTrue(np.all(s_torch.scopes_out == s_base.scopes_out))


if __name__ == "__main__":
    unittest.main()
