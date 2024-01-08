import unittest

import numpy as np
import torch
import tensorly as tl


from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext

# from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.structure import marginalize
from spflow.modules.node import Gaussian
from spflow.structure.spn import CondSumLayer
from spflow.structure.spn.layer_layerbased import toLayerBased, toNodeBased, updateBackend

from ...general.node.dummy_node import DummyNode

tc = unittest.TestCase()


def test_sum_layer_initialization(do_for_all_backends):
    # dummy children over same scope
    input_nodes = [
        DummyNode(Scope([0, 1])),
        DummyNode(Scope([0, 1])),
        DummyNode(Scope([0, 1])),
    ]

    # ----- check attributes after correct initialization -----

    l = CondSumLayer(n_nodes=3, children=input_nodes)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([0, 1]), Scope([0, 1]), Scope([0, 1])]))

    # ----- children of different scopes -----
    tc.assertRaises(
        ValueError,
        CondSumLayer,
        n_nodes=3,
        children=[
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0])),
        ],
    )

    # ----- no children -----
    tc.assertRaises(ValueError, CondSumLayer, n_nodes=3, children=[])

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, CondSumLayer, n_nodes=0, children=input_nodes)

    # -----number of cond_f functions -----
    CondSumLayer(
        children=input_nodes,
        n_nodes=2,
        cond_f=[
            lambda data: {"weights": [0.2, 0.2, 0.6]},
            lambda data: {"weights": [0.3, 0.5, 0.2]},
        ],
    )
    tc.assertRaises(
        ValueError,
        CondSumLayer,
        children=input_nodes,
        n_nodes=2,
        cond_f=[lambda data: {"weights": [0.5, 0.3, 0.2]}],
    )


def test_retrieve_params(do_for_all_backends):
    # dummy children over same scope
    input_nodes = [
        DummyNode(Scope([0, 1])),
        DummyNode(Scope([0, 1])),
        DummyNode(Scope([0, 1])),
    ]

    # ----- same weights for all nodes -----
    weights = tl.tensor([[0.3, 0.3, 0.4]])

    # two dimensional weight array
    l = CondSumLayer(
        n_nodes=3,
        children=input_nodes,
        cond_f=lambda data: {"weights": weights},
    )

    for node_weights in l.retrieve_params(tl.tensor([[1]]), DispatchContext()):
        tc.assertTrue(np.allclose(node_weights, weights))

    # one dimensional weight array
    l.set_cond_f(lambda data: {"weights": weights.squeeze(0)})

    for node_weights in l.retrieve_params(tl.tensor([[1]]), DispatchContext()):
        tc.assertTrue(np.allclose(node_weights, weights))

    # ----- different weights for all nodes -----
    weights = tl.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3], [0.1, 0.7, 0.2]])

    l.set_cond_f(lambda data: {"weights": weights})

    for weights_actual, node_weights in zip(weights, l.retrieve_params(tl.tensor([[1]]), DispatchContext())):
        tc.assertTrue(np.allclose(node_weights, weights_actual))

    # ----- two dimensional weight array of wrong shape -----
    weights = tl.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3]])

    l.set_cond_f(lambda data: {"weights": weights})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    l.set_cond_f(lambda data: {"weights": weights.T})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    l.set_cond_f(lambda data: {"weights": np.expand_dims(weights, 0)})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    l.set_cond_f(lambda data: {"weights": np.expand_dims(weights, -1)})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    # ----- incorrect number of weights -----
    l.set_cond_f(lambda data: {"weights": np.array([[0.3, 0.3, 0.3, 0.1], [0.5, 0.2, 0.2, 0.1]])})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    l.set_cond_f(lambda data: {"weights": np.array([[0.3, 0.7], [0.5, 0.5]])})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    # ----- weights not summing up to one per row -----
    l.set_cond_f(lambda data: {"weights": np.array([[0.3, 0.3, 0.4], [0.5, 0.7, 0.3], [0.1, 0.7, 0.2]])})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    # ----- non-positive weights -----
    l.set_cond_f(lambda data: {"weights": np.array([[0.3, 0.3, 0.4], [0.5, 0.0, 0.5], [0.1, 0.7, 0.2]])})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )


def test_sum_layer_structural_marginalization(do_for_all_backends):
    # dummy children over same scope
    input_nodes = [
        DummyNode(Scope([0, 1])),
        DummyNode(Scope([0, 1])),
        DummyNode(Scope([0, 1])),
    ]
    l = CondSumLayer(n_nodes=3, children=input_nodes)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- marginalize over partial scope -----
    l_marg = marginalize(l, [0])
    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1]), Scope([1])])

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])
    tc.assertTrue(l_marg.scopes_out == [Scope([0, 1]), Scope([0, 1]), Scope([0, 1])])


def test_sum_layer_layerbased_conversion(do_for_all_backends):
    sum_layer = CondSumLayer(
        n_nodes=3,
        children=[
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
        ],
    )

    layer_based_sum_layer = toLayerBased(sum_layer)
    tc.assertEqual(layer_based_sum_layer.n_out, sum_layer.n_out)
    node_based_sum_layer = toNodeBased(layer_based_sum_layer)
    tc.assertEqual(node_based_sum_layer.n_out, sum_layer.n_out)

    node_based_sum_layer2 = toNodeBased(sum_layer)
    tc.assertEqual(node_based_sum_layer2.n_out, sum_layer.n_out)
    layer_based_sum_layer2 = toLayerBased(layer_based_sum_layer)
    tc.assertEqual(layer_based_sum_layer2.n_out, sum_layer.n_out)


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    sum_layer = CondSumLayer(
        n_nodes=3,
        children=[
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
        ],
    )
    n_out = sum_layer.n_out
    for backend in backends:
        with tl.backend_context(backend):
            sum_layer_updated = updateBackend(sum_layer)
            tc.assertTrue(n_out == sum_layer_updated.n_out)


def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)

    input_nodes = [
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
    ]
    weights = tl.tensor([[0.3, 0.3, 0.4]])

    # two dimensional weight array
    model_default = CondSumLayer(
        n_nodes=3,
        children=input_nodes,
        cond_f=lambda data: {"weights": weights},
    )
    for m in model_default.modules():
        tc.assertTrue(m.dtype == tl.float32)

    weights = model_default.retrieve_params(tl.tensor([[1]]), DispatchContext())

    tc.assertTrue(weights.dtype == tl.float32)

    # change to float64 model

    model_updated = CondSumLayer(
        n_nodes=3,
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
    weights = tl.tensor([[0.3, 0.3, 0.4]])

    # two dimensional weight array
    model_default = CondSumLayer(
        n_nodes=3,
        children=input_nodes,
        cond_f=lambda data: {"weights": weights},
    )
    input_nodes = [
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
    ]
    weights = tl.tensor([[0.3, 0.3, 0.4]])
    model_updated = CondSumLayer(
        n_nodes=3,
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
