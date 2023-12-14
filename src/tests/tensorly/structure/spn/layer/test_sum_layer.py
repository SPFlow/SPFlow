import unittest

import numpy as np
import torch
import tensorly as tl


from spflow.meta.data import Scope
from spflow.tensorly.structure.general.node.leaf.general_gaussian import Gaussian
from spflow.tensorly.structure.spn import SumLayer
from spflow.tensorly.structure import marginalize
from spflow.tensorly.structure.spn.layer.sum_layer import toLayerBased, toNodeBased
from spflow.tensorly.structure.spn.layer_layerbased.sum_layer import toLayerBased, toNodeBased, updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy


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

    l = SumLayer(n_nodes=3, children=input_nodes)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([0, 1]), Scope([0, 1]), Scope([0, 1])]))
    # make sure weight property works correctly
    tc.assertTrue(l.weights.shape == (3, 3))

    # ----- same weights for all nodes -----
    weights = tl.tensor([[0.3, 0.3, 0.4]])

    # two dimensional weight array
    l = SumLayer(n_nodes=3, children=input_nodes, weights=weights)

    for i in range(3):
        tc.assertTrue(np.allclose(tl_toNumpy(l.weights[i]), weights))

    # one dimensional weight array
    l = SumLayer(n_nodes=3, children=input_nodes, weights=weights.squeeze(0))

    for i in range(3):
        tc.assertTrue(np.allclose(tl_toNumpy(l.weights[i]), weights))

    # ----- different weights for all nodes -----
    weights = tl.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3], [0.1, 0.7, 0.2]])

    l = SumLayer(n_nodes=3, children=input_nodes, weights=weights)
    for i in range(3):
        tc.assertTrue(np.allclose(tl_toNumpy(l.weights[i]), weights[i]))

    # ----- two dimensional weight array of wrong shape -----
    weights = tl.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3]])

    tc.assertRaises(ValueError, SumLayer, 3, input_nodes, weights)
    tc.assertRaises(ValueError, SumLayer, 3, input_nodes, weights.T)

    # ----- weights not summing up to one per row -----
    weights = tl.tensor([[0.3, 0.3, 0.4], [0.5, 0.7, 0.3], [0.1, 0.7, 0.2]])
    tc.assertRaises(ValueError, SumLayer, 3, input_nodes, weights)

    # ----- non-positive weights -----
    weights = tl.tensor([[0.3, 0.3, 0.4], [0.5, 0.0, 0.5], [0.1, 0.7, 0.2]])
    tc.assertRaises(ValueError, SumLayer, 3, input_nodes, weights)

    # ----- children of different scopes -----
    input_nodes = [
        DummyNode(Scope([0, 1])),
        DummyNode(Scope([0, 1])),
        DummyNode(Scope([0])),
    ]
    tc.assertRaises(ValueError, SumLayer, 3, input_nodes)

    # ----- no children -----
    tc.assertRaises(ValueError, SumLayer, 3, [])

def test_sum_layer_structural_marginalization(do_for_all_backends):

    # dummy children over same scope
    input_nodes = [
        DummyNode(Scope([0, 1])),
        DummyNode(Scope([0, 1])),
        DummyNode(Scope([0, 1])),
    ]
    l = SumLayer(n_nodes=3, children=input_nodes)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- marginalize over partial scope -----
    l_marg = marginalize(l, [0])
    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1]), Scope([1])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.weights), tl_toNumpy(l_marg.weights)))

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])
    tc.assertTrue(l_marg.scopes_out == [Scope([0, 1]), Scope([0, 1]), Scope([0, 1])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.weights), tl_toNumpy(l_marg.weights)))

def test_sum_layer_layerbased_conversion(do_for_all_backends):

    sum_layer = SumLayer(
        n_nodes=3,
        children=[
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
        ],
    )

    layer_based_sum_layer = toLayerBased(sum_layer)
    tc.assertTrue(np.allclose(tl_toNumpy(layer_based_sum_layer.weights), tl_toNumpy(sum_layer.weights)))
    tc.assertEqual(layer_based_sum_layer.n_out, sum_layer.n_out)
    node_based_sum_layer = toNodeBased(layer_based_sum_layer)
    tc.assertTrue(np.allclose(tl_toNumpy(node_based_sum_layer.weights), tl_toNumpy(sum_layer.weights)))
    tc.assertEqual(node_based_sum_layer.n_out, sum_layer.n_out)

    node_based_sum_layer2 = toNodeBased(sum_layer)
    tc.assertTrue(np.allclose(tl_toNumpy(node_based_sum_layer2.weights), tl_toNumpy(sum_layer.weights)))
    tc.assertEqual(node_based_sum_layer2.n_out, sum_layer.n_out)
    layer_based_sum_layer2 = toLayerBased(layer_based_sum_layer)
    tc.assertTrue(np.allclose(tl_toNumpy(layer_based_sum_layer2.weights), tl_toNumpy(sum_layer.weights)))
    tc.assertEqual(layer_based_sum_layer2.n_out, sum_layer.n_out)

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    sum_layer = SumLayer(
        n_nodes=3,
        children=[
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
        ],
    )
    weights = tl_toNumpy(sum_layer.weights)
    n_out = sum_layer.n_out
    for backend in backends:
        with tl.backend_context(backend):
            sum_layer_updated = updateBackend(sum_layer)
            tc.assertTrue(n_out == sum_layer_updated.n_out)
            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    weights,
                    tl_toNumpy(sum_layer_updated.weights)
                )
            )

def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    model_default = SumLayer(
        n_nodes=3,
        children=[
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
        ],
    )
    for m in model_default.modules():
        tc.assertTrue(m.dtype == tl.float32)

    tc.assertTrue(model_default.weights.dtype == tl.float32)

    # change to float64 model
    model_updated = SumLayer(
        n_nodes=3,
        children=[
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
        ],
    )
    model_updated.to_dtype(tl.float64)
    for m in model_updated.modules():
        tc.assertTrue(m.dtype == tl.float64)

    tc.assertTrue(model_updated.weights.dtype == tl.float64)


def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    torch.set_default_dtype(torch.float32)
    model_default = SumLayer(
        n_nodes=3,
        children=[
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
        ],
    )
    model_updated = SumLayer(
        n_nodes=3,
        children=[
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
            Gaussian(Scope([0])),
        ],
    )
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, model_updated.to_device, cuda)
        return

    # put model on gpu
    model_updated.to_device(cuda)

    tc.assertTrue(model_default.weights.device.type == "cpu")
    tc.assertTrue(model_updated.weights.device.type == "cuda")

    for m in model_default.modules():
        tc.assertTrue(m.device.type == "cpu")
    for m in model_updated.modules():
        tc.assertTrue(m.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
