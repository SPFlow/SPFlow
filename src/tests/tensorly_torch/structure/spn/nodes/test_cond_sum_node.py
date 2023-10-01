import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.tensorly.structure.spn import CondSumNode, ProductNode
from spflow.tensorly.structure import marginalize
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gaussian import Gaussian
from spflow.tensorly.structure.spn.nodes.cond_sum_node import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

from ...general.nodes.dummy_node import DummyNode

tc = unittest.TestCase()

def test_sum_node_initialization(do_for_all_backends):

    # empty children
    tc.assertRaises(ValueError, CondSumNode, [], [])
    # non-Module children
    tc.assertRaises(ValueError, CondSumNode, [DummyNode(Scope([0])), 0])
    # children with different scopes
    tc.assertRaises(
        ValueError,
        CondSumNode,
        [DummyNode(Scope([0])), DummyNode(Scope([1]))],
    )

def test_retrieve_params(do_for_all_backends):

    node = CondSumNode([DummyNode(Scope([0])), DummyNode(Scope([0]))])

    # number of child outputs not matching number of weights
    node.set_cond_f(lambda data: {"weights": [1.0]})
    tc.assertRaises(
        ValueError,
        node.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )
    # non-positive weights
    node.set_cond_f(lambda data: {"weights": [0.0, 1.0]})
    tc.assertRaises(
        ValueError,
        node.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )
    # weights not summing up to one
    node.set_cond_f(lambda data: {"weights": [0.3, 0.5]})
    tc.assertRaises(
        ValueError,
        node.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )
    # weights of invalid shape
    node.set_cond_f(lambda data: {"weights": [[0.5, 0.5]]})
    tc.assertRaises(
        ValueError,
        node.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    # weights as list of floats
    node.set_cond_f(lambda data: {"weights": [0.5, 0.5]})
    node.retrieve_params(tl.tensor([[1]]), DispatchContext())
    # weights as numpy array
    node.set_cond_f(lambda data: {"weights": np.array([0.5, 0.5])})
    node.retrieve_params(tl.tensor([[1]]), DispatchContext())
    # weights as torch tensor
    node.set_cond_f(lambda data: {"weights": tl.tensor([0.5, 0.5])})
    node.retrieve_params(tl.tensor([[1]]), DispatchContext())

def test_sum_node_marginalization_1(do_for_all_backends):

    s = CondSumNode([DummyNode(Scope([0])), DummyNode(Scope([0]))])

    s_marg = marginalize(s, [1])
    tc.assertEqual(s_marg.scopes_out, s.scopes_out)

    s_marg = marginalize(s, [0])
    tc.assertEqual(s_marg, None)

def test_sum_node_marginalization_2(do_for_all_backends):

    s = CondSumNode(
        [
            ProductNode([DummyNode(Scope([0])), DummyNode(Scope([1]))]),
            ProductNode([DummyNode(Scope([0])), DummyNode(Scope([1]))]),
        ]
    )

    s_marg = marginalize(s, [0])
    tc.assertEqual(s_marg.scopes_out, [Scope([1])])

    s_marg = marginalize(s, [1])
    tc.assertEqual(s_marg.scopes_out, [Scope([0])])

    s_marg = marginalize(s, [0, 1])
    tc.assertEqual(s_marg, None)

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    sum_node = CondSumNode([Gaussian(Scope([0])), Gaussian(Scope([0]))])
    for backend in backends:
        with tl.backend_context(backend):
            sum_node_updated = updateBackend(sum_node)
            tc.assertTrue(sum_node.scopes_out == sum_node_updated.scopes_out)


if __name__ == "__main__":
    unittest.main()
