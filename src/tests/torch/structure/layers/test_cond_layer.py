from spflow.torch.structure.layers.cond_layer import (
    SPNCondSumLayer,
    marginalize,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.leaves.parametric.gaussian import (
    Gaussian,
    toBase,
    toTorch,
)
from spflow.base.structure.layers.cond_layer import (
    SPNCondSumLayer as BaseSPNCondSumLayer,
)
from spflow.base.structure.nodes.leaves.parametric.gaussian import (
    Gaussian as BaseGaussian,
)
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.meta.scope.scope import Scope
from ..nodes.dummy_node import DummyNode
import torch
import numpy as np
import unittest
import itertools


class TestNode(unittest.TestCase):
    def test_sum_layer_initialization(self):

        # dummy children over same scope
        input_nodes = [
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
        ]

        # ----- check attributes after correct initialization -----

        l = SPNCondSumLayer(n_nodes=3, children=input_nodes)
        # make sure scopes are correct
        self.assertTrue(
            np.all(
                l.scopes_out == [Scope([0, 1]), Scope([0, 1]), Scope([0, 1])]
            )
        )

        # ----- children of different scopes -----
        self.assertRaises(
            ValueError,
            SPNCondSumLayer,
            n_nodes=3,
            children=[
                DummyNode(Scope([0, 1])),
                DummyNode(Scope([0, 1])),
                DummyNode(Scope([0])),
            ],
        )

        # ----- no children -----
        self.assertRaises(ValueError, SPNCondSumLayer, n_nodes=3, children=[])

        # ----- invalid number of nodes -----
        self.assertRaises(
            ValueError, SPNCondSumLayer, n_nodes=0, children=input_nodes
        )

        # -----number of cond_f functions -----
        SPNCondSumLayer(
            children=input_nodes,
            n_nodes=2,
            cond_f=[
                lambda data: {"weights": [0.2, 0.2, 0.6]},
                lambda data: {"weights": [0.3, 0.5, 0.2]},
            ],
        )
        self.assertRaises(
            ValueError,
            SPNCondSumLayer,
            children=input_nodes,
            n_nodes=2,
            cond_f=[lambda data: {"weights": [0.5, 0.3, 0.2]}],
        )

    def test_retrieve_params(self):

        # dummy children over same scope
        input_nodes = [
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
        ]

        # ----- same weights for all nodes -----
        weights = torch.tensor([[0.3, 0.3, 0.4]])

        # two dimensional weight array
        l = SPNCondSumLayer(
            n_nodes=3,
            children=input_nodes,
            cond_f=lambda data: {"weights": weights},
        )

        for node_weights in l.retrieve_params(
            torch.tensor([[1]]), DispatchContext()
        ):
            self.assertTrue(torch.all(node_weights == weights))

        # one dimensional weight array
        l.set_cond_f(lambda data: {"weights": weights.squeeze(0)})

        for node_weights in l.retrieve_params(
            torch.tensor([[1]]), DispatchContext()
        ):
            self.assertTrue(torch.all(node_weights == weights))

        # ----- different weights for all nodes -----
        weights = torch.tensor(
            [[0.3, 0.3, 0.4], [0.5, 0.2, 0.3], [0.1, 0.7, 0.2]]
        )

        l.set_cond_f(lambda data: {"weights": weights})

        for weights_actual, node_weights in zip(
            weights, l.retrieve_params(torch.tensor([[1]]), DispatchContext())
        ):
            self.assertTrue(torch.all(node_weights == weights_actual))

        # ----- two dimensional weight array of wrong shape -----
        weights = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3]])

        l.set_cond_f(lambda data: {"weights": weights})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        l.set_cond_f(lambda data: {"weights": weights.T})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        l.set_cond_f(lambda data: {"weights": np.expand_dims(weights, 0)})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        l.set_cond_f(lambda data: {"weights": np.expand_dims(weights, -1)})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        # ----- incorrect number of weights -----
        l.set_cond_f(
            lambda data: {
                "weights": np.array(
                    [[0.3, 0.3, 0.3, 0.1], [0.5, 0.2, 0.2, 0.1]]
                )
            }
        )
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        l.set_cond_f(
            lambda data: {"weights": np.array([[0.3, 0.7], [0.5, 0.5]])}
        )
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        # ----- weights not summing up to one per row -----
        l.set_cond_f(
            lambda data: {
                "weights": np.array(
                    [[0.3, 0.3, 0.4], [0.5, 0.7, 0.3], [0.1, 0.7, 0.2]]
                )
            }
        )
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        # ----- non-positive weights -----
        l.set_cond_f(
            lambda data: {
                "weights": np.array(
                    [[0.3, 0.3, 0.4], [0.5, 0.0, 0.5], [0.1, 0.7, 0.2]]
                )
            }
        )
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

    def test_sum_layer_structural_marginalization(self):

        # dummy children over same scope
        input_nodes = [
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([0, 1])),
        ]
        l = SPNCondSumLayer(n_nodes=3, children=input_nodes)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- marginalize over partial scope -----
        l_marg = marginalize(l, [0])
        self.assertTrue(
            l_marg.scopes_out == [Scope([1]), Scope([1]), Scope([1])]
        )

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])
        self.assertTrue(
            l_marg.scopes_out == [Scope([0, 1]), Scope([0, 1]), Scope([0, 1])]
        )

    def test_sum_layer_backend_conversion_1(self):

        torch_sum_layer = SPNCondSumLayer(
            n_nodes=3,
            children=[
                Gaussian(Scope([0])),
                Gaussian(Scope([0])),
                Gaussian(Scope([0])),
            ],
        )
        base_sum_layer = toBase(torch_sum_layer)

        self.assertEqual(base_sum_layer.n_out, torch_sum_layer.n_out)

    def test_sum_layer_backend_conversion_2(self):

        base_sum_layer = BaseSPNCondSumLayer(
            n_nodes=3,
            children=[
                BaseGaussian(Scope([0])),
                BaseGaussian(Scope([0])),
                BaseGaussian(Scope([0])),
            ],
        )
        torch_sum_layer = toTorch(base_sum_layer)

        self.assertEqual(base_sum_layer.n_out, torch_sum_layer.n_out)


if __name__ == "__main__":
    unittest.main()
