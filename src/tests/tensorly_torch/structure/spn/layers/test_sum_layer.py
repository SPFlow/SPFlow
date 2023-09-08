import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.spn import Gaussian as BaseGaussian
from spflow.base.structure.spn import SumLayer as BaseSumLayer
from spflow.meta.data import Scope
from spflow.torch.structure import toBase, toTorch
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gaussian import Gaussian
from spflow.tensorly.structure.spn import SumLayer
from spflow.tensorly.structure import marginalize
from spflow.tensorly.structure.spn.layers.sum_layer import toLayerBased, toNodeBased
from spflow.tensorly.structure.spn.layers_layerbased.sum_layer import toLayerBased, toNodeBased, updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy


from ...general.nodes.dummy_node import DummyNode


class TestNode(unittest.TestCase):
    def test_sum_layer_initialization(self):

        # dummy children over same scope
        input_nodes = [
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
        ]

        # ----- check attributes after correct initialization -----

        l = SumLayer(n_nodes=3, children=input_nodes)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([0, 1]), Scope([0, 1]), Scope([0, 1])]))
        # make sure weight property works correctly
        self.assertTrue(l.weights.shape == (3, 3))

        # ----- same weights for all nodes -----
        weights = torch.tensor([[0.3, 0.3, 0.4]])

        # two dimensional weight array
        l = SumLayer(n_nodes=3, children=input_nodes, weights=weights)

        for i in range(3):
            self.assertTrue(torch.allclose(l.weights[i], weights))

        # one dimensional weight array
        l = SumLayer(n_nodes=3, children=input_nodes, weights=weights.squeeze(0))

        for i in range(3):
            self.assertTrue(torch.allclose(l.weights[i], weights))

        # ----- different weights for all nodes -----
        weights = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3], [0.1, 0.7, 0.2]])

        l = SumLayer(n_nodes=3, children=input_nodes, weights=weights)
        for i in range(3):
            self.assertTrue(torch.allclose(l.weights[i], weights[i]))

        # ----- two dimensional weight array of wrong shape -----
        weights = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3]])

        self.assertRaises(ValueError, SumLayer, 3, input_nodes, weights)
        self.assertRaises(ValueError, SumLayer, 3, input_nodes, weights.T)

        # ----- weights not summing up to one per row -----
        weights = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.7, 0.3], [0.1, 0.7, 0.2]])
        self.assertRaises(ValueError, SumLayer, 3, input_nodes, weights)

        # ----- non-positive weights -----
        weights = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.0, 0.5], [0.1, 0.7, 0.2]])
        self.assertRaises(ValueError, SumLayer, 3, input_nodes, weights)

        # ----- children of different scopes -----
        input_nodes = [
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0])),
        ]
        self.assertRaises(ValueError, SumLayer, 3, input_nodes)

        # ----- no children -----
        self.assertRaises(ValueError, SumLayer, 3, [])

    def test_sum_layer_structural_marginalization(self):

        # dummy children over same scope
        input_nodes = [
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
        ]
        l = SumLayer(n_nodes=3, children=input_nodes)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- marginalize over partial scope -----
        l_marg = marginalize(l, [0])
        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1]), Scope([1])])
        self.assertTrue(torch.allclose(l.weights, l_marg.weights))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])
        self.assertTrue(l_marg.scopes_out == [Scope([0, 1]), Scope([0, 1]), Scope([0, 1])])
        self.assertTrue(torch.allclose(l.weights, l_marg.weights))
    """
    def test_sum_layer_backend_conversion_1(self):

        torch_sum_layer = SumLayer(
            n_nodes=3,
            children=[
                Gaussian(Scope([0])),
                Gaussian(Scope([0])),
                Gaussian(Scope([0])),
            ],
        )

        base_sum_layer = toBase(torch_sum_layer)
        self.assertTrue(np.allclose(base_sum_layer.weights, torch_sum_layer.weights.detach().numpy()))
        self.assertEqual(base_sum_layer.n_out, torch_sum_layer.n_out)

    def test_sum_layer_backend_conversion_2(self):

        base_sum_layer = BaseSumLayer(
            n_nodes=3,
            children=[
                BaseGaussian(Scope([0])),
                BaseGaussian(Scope([0])),
                BaseGaussian(Scope([0])),
            ],
        )

        torch_sum_layer = toTorch(base_sum_layer)
        self.assertTrue(np.allclose(base_sum_layer.weights, torch_sum_layer.weights.detach().numpy()))
        self.assertEqual(base_sum_layer.n_out, torch_sum_layer.n_out)
    """
    def test_sum_layer_layerbased_conversion(self):

        sum_layer = SumLayer(
            n_nodes=3,
            children=[
                Gaussian(Scope([0])),
                Gaussian(Scope([0])),
                Gaussian(Scope([0])),
            ],
        )

        layer_based_sum_layer = toLayerBased(sum_layer)
        self.assertTrue(np.allclose(layer_based_sum_layer.weights.detach().numpy(), sum_layer.weights.detach().numpy()))
        self.assertEqual(layer_based_sum_layer.n_out, sum_layer.n_out)
        node_based_sum_layer = toNodeBased(layer_based_sum_layer)
        self.assertTrue(np.allclose(node_based_sum_layer.weights.detach().numpy(), sum_layer.weights.detach().numpy()))
        self.assertEqual(node_based_sum_layer.n_out, sum_layer.n_out)

        node_based_sum_layer2 = toNodeBased(sum_layer)
        self.assertTrue(np.allclose(node_based_sum_layer2.weights.detach().numpy(), sum_layer.weights.detach().numpy()))
        self.assertEqual(node_based_sum_layer2.n_out, sum_layer.n_out)
        layer_based_sum_layer2 = toLayerBased(layer_based_sum_layer)
        self.assertTrue(np.allclose(layer_based_sum_layer2.weights.detach().numpy(), sum_layer.weights.detach().numpy()))
        self.assertEqual(layer_based_sum_layer2.n_out, sum_layer.n_out)

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        sum_layer = SumLayer(
            n_nodes=3,
            children=[
                Gaussian(Scope([0])),
                Gaussian(Scope([0])),
                Gaussian(Scope([0])),
            ],
        )
        weights = sum_layer.weights.detach().numpy()
        n_out = sum_layer.n_out
        for backend in backends:
            tl.set_backend(backend)
            sum_layer_updated = updateBackend(sum_layer)
            self.assertTrue(n_out == sum_layer_updated.n_out)
            # check conversion from torch to python
            self.assertTrue(
                np.allclose(
                    weights,
                    tl_toNumpy(sum_layer_updated.weights)
                )
            )


if __name__ == "__main__":
    unittest.main()
