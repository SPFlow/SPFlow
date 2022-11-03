from spflow.torch.structure.spn.layers.product_layer import (
    SPNProductLayer,
    marginalize,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.leaves.parametric.gaussian import (
    Gaussian,
    toBase,
    toTorch,
)
from spflow.base.structure.spn.layers.product_layer import (
    SPNProductLayer as BaseSPNProductLayer,
)
from spflow.base.structure.nodes.leaves.parametric.gaussian import (
    Gaussian as BaseGaussian,
)
from spflow.meta.data.scope import Scope
from ..nodes.dummy_node import DummyNode
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    def test_product_layer_initialization(self):

        # dummy children pair-wise disjoint scopes
        input_nodes = [
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([3])),
            DummyNode(Scope([2])),
        ]

        # ----- check attributes after correct initialization -----

        l = SPNProductLayer(n_nodes=3, children=input_nodes)
        # make sure scopes are correct
        self.assertTrue(
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
        self.assertRaises(ValueError, SPNProductLayer, 3, input_nodes)

        # ----- no children -----
        self.assertRaises(ValueError, SPNProductLayer, 3, [])

    def test_product_layer_structural_marginalization(self):

        # dummy children over pair-wise disjoint scopes
        input_nodes = [
            DummyNode(Scope([0, 1])),
            DummyNode(Scope([3])),
            DummyNode(Scope([2])),
        ]
        l = SPNProductLayer(n_nodes=3, children=input_nodes)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1, 2, 3]) == None)

        # ----- marginalize over partial scope -----
        l_marg = marginalize(l, [3])
        self.assertTrue(
            l_marg.scopes_out
            == [Scope([0, 1, 2]), Scope([0, 1, 2]), Scope([0, 1, 2])]
        )
        # number of children should be reduced by one (i.e., marginalized over)
        self.assertTrue(len(list(l_marg.children())) == 2)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [4])
        self.assertTrue(
            l_marg.scopes_out
            == [Scope([0, 1, 2, 3]), Scope([0, 1, 2, 3]), Scope([0, 1, 2, 3])]
        )

        # ----- pruning -----
        l = SPNProductLayer(n_nodes=3, children=input_nodes[:2])

        l_marg = marginalize(l, [0, 1], prune=True)
        self.assertTrue(isinstance(l_marg, DummyNode))

    def test_product_layer_backend_conversion_1(self):

        torch_product_layer = SPNProductLayer(
            n_nodes=3,
            children=[
                Gaussian(Scope([0])),
                Gaussian(Scope([1])),
                Gaussian(Scope([2])),
            ],
        )

        base_product_layer = toBase(torch_product_layer)
        self.assertEqual(base_product_layer.n_out, torch_product_layer.n_out)

    def test_product_layer_backend_conversion_2(self):

        base_product_layer = BaseSPNProductLayer(
            n_nodes=3,
            children=[
                BaseGaussian(Scope([0])),
                BaseGaussian(Scope([1])),
                BaseGaussian(Scope([2])),
            ],
        )

        torch_product_layer = toTorch(base_product_layer)
        self.assertEqual(base_product_layer.n_out, torch_product_layer.n_out)


if __name__ == "__main__":
    unittest.main()
