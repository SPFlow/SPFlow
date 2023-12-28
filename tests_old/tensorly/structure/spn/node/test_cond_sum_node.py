import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.structure.spn import CondSumNode, ProductNode
from spflow.structure import marginalize
from spflow.modules.node import Gaussian
from spflow.modules.node import updateBackend

from ...general.node.dummy_node import DummyNode

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


def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)

    input_nodes = [
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
    ]
    weights = tl.tensor([0.3, 0.3, 0.4])

    # two dimensional weight array
    model_default = CondSumNode(
        children=input_nodes,
        cond_f=lambda data: {"weights": weights},
    )
    for m in model_default.modules():
        tc.assertTrue(m.dtype == tl.float32)

    weights = model_default.retrieve_params(tl.tensor([[1]]), DispatchContext())

    tc.assertTrue(weights.dtype == tl.float32)

    # change to float64 model

    model_updated = CondSumNode(
        children=input_nodes,
        cond_f=lambda data: {"weights": weights},
    )
    model_updated.to_dtype(tl.float64)
    for m in model_updated.modules():
        tc.assertTrue(m.dtype == tl.float64)

    weights_up = model_updated.retrieve_params(tl.tensor([[1]]), DispatchContext())

    tc.assertTrue(weights_up.dtype == tl.float64)


def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    torch.set_default_dtype(torch.float32)
    input_nodes = [
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
    ]
    weights = tl.tensor([0.3, 0.3, 0.4])

    # two dimensional weight array
    model_default = CondSumNode(
        children=input_nodes,
        cond_f=lambda data: {"weights": weights},
    )
    input_nodes = [
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
    ]
    weights = tl.tensor([0.3, 0.3, 0.4])
    model_updated = CondSumNode(
        children=input_nodes,
        cond_f=lambda data: {"weights": weights},
    )
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, model_updated.to_device, cuda)
        return

    # put model on gpu
    model_updated.to_device(cuda)

    weights = model_default.retrieve_params(tl.tensor([[1]]), DispatchContext())
    weights_up = model_updated.retrieve_params(tl.tensor([[1]]), DispatchContext())
    tc.assertTrue(weights.device.type == "cpu")
    tc.assertTrue(weights_up.device.type == "cuda")

    for m in model_default.modules():
        tc.assertTrue(m.device.type == "cpu")
    for m in model_updated.modules():
        tc.assertTrue(m.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
