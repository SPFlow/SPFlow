import unittest

import numpy as np
import tensorly as tl

from spflow.meta.data import Scope
from spflow.tensorly.structure import marginalize
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gaussian import Gaussian
from spflow.tensorly.structure.spn import ProductLayer
from spflow.tensorly.structure.spn.layers.product_layer import toLayerBased, toNodeBased
from spflow.tensorly.structure.spn.layers_layerbased.product_layer import toLayerBased, toNodeBased, updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

from ...general.nodes.dummy_node import DummyNode

tc = unittest.TestCase()

def test_product_layer_initialization(do_for_all_backends):

    # dummy children pair-wise disjoint scopes
    input_nodes = [
        DummyNode(Scope([0, 1])),
        DummyNode(Scope([3])),
        DummyNode(Scope([2])),
    ]

    # ----- check attributes after correct initialization -----

    l = ProductLayer(n_nodes=3, children=input_nodes)
    # make sure scopes are correct
    tc.assertTrue(
        np.all(
            l.scopes_out
            == [
                Scope([0, 1, 2, 3]),
                Scope([0, 1, 2, 3]),
                Scope([0, 1, 2, 3]),
            ]
        )
    )

    # ----- children of non-pair-wise disjoint scopes -----
    input_nodes = [
        DummyNode(Scope([0, 1])),
        DummyNode(Scope([3])),
        DummyNode(Scope([1])),
    ]
    tc.assertRaises(ValueError, ProductLayer, 3, input_nodes)

    # ----- no children -----
    tc.assertRaises(ValueError, ProductLayer, 3, [])

def test_product_layer_structural_marginalization(do_for_all_backends):

    # dummy children over pair-wise disjoint scopes
    input_nodes = [
        DummyNode(Scope([0, 1])),
        DummyNode(Scope([3])),
        DummyNode(Scope([2])),
    ]
    l = ProductLayer(n_nodes=3, children=input_nodes)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1, 2, 3]) == None)

    # ----- marginalize over partial scope -----
    l_marg = marginalize(l, [3])
    tc.assertTrue(l_marg.scopes_out == [Scope([0, 1, 2]), Scope([0, 1, 2]), Scope([0, 1, 2])])
    # number of children should be reduced by one (i.e., marginalized over)
    tc.assertTrue(len(l_marg.children) == 2)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [4])
    tc.assertTrue(l_marg.scopes_out == [Scope([0, 1, 2, 3]), Scope([0, 1, 2, 3]), Scope([0, 1, 2, 3])])

    # ----- pruning -----
    l = ProductLayer(n_nodes=3, children=input_nodes[:2])

    l_marg = marginalize(l, [0, 1], prune=True)
    tc.assertTrue(isinstance(l_marg, DummyNode))

def test_sum_layer_layerbased_conversion(do_for_all_backends):

    product_layer = ProductLayer(
        n_nodes=3,
        children=[
            Gaussian(Scope([0])),
            Gaussian(Scope([1])),
            Gaussian(Scope([2])),
        ],
    )

    layer_based_product_layer = toLayerBased(product_layer)
    tc.assertEqual(layer_based_product_layer.n_out, product_layer.n_out)
    node_based_product_layer = toNodeBased(layer_based_product_layer)
    tc.assertEqual(node_based_product_layer.n_out, product_layer.n_out)

    node_based_product_layer2 = toNodeBased(product_layer)
    tc.assertEqual(node_based_product_layer2.n_out, product_layer.n_out)
    layer_based_product_layer2 = toLayerBased(layer_based_product_layer)
    tc.assertEqual(layer_based_product_layer2.n_out, product_layer.n_out)

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    product_layer = ProductLayer(
        n_nodes=3,
        children=[
            Gaussian(Scope([0])),
            Gaussian(Scope([1])),
            Gaussian(Scope([2])),
        ],
    )
    n_out = product_layer.n_out
    for backend in backends:
        with tl.backend_context(backend):
            product_layer_updated = updateBackend(product_layer)
            tc.assertTrue(n_out == product_layer_updated.n_out)


if __name__ == "__main__":
    unittest.main()
