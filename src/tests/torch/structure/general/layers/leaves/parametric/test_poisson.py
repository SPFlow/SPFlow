import unittest

import numpy as np
import torch

from spflow.base.structure.spn import PoissonLayer as BasePoissonLayer
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import AutoLeaf, marginalize, toBase, toTorch
from spflow.torch.structure.spn import Poisson, PoissonLayer


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_initialization(self):

        # ----- check attributes after correct initialization -----
        l_values = [0.5, 2.3, 1.0]
        l = PoissonLayer(scope=Scope([1]), n_nodes=3, l=l_values)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
        # make sure parameter properties works correctly
        for l_layer_node, l_value in zip(l.l, l_values):
            self.assertTrue(torch.allclose(l_layer_node, torch.tensor(l_value)))

        # ----- float/int parameter values -----
        l_value = 0.73
        l = PoissonLayer(scope=Scope([1]), n_nodes=3, l=l_value)

        for l_layer_node in l.l:
            self.assertTrue(torch.allclose(l_layer_node, torch.tensor(l_value)))

        # ----- list parameter values -----
        l_values = [0.17, 0.8, 0.53]
        l = PoissonLayer(scope=Scope([1]), n_nodes=3, l=l_values)

        for l_layer_node, l_value in zip(l.l, l_values):
            self.assertTrue(torch.allclose(l_layer_node, torch.tensor(l_value)))

        # wrong number of values
        self.assertRaises(ValueError, PoissonLayer, Scope([0]), l_values[:-1], n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            PoissonLayer,
            Scope([0]),
            [l_values for _ in range(3)],
            n_nodes=3,
        )

        # ----- numpy parameter values -----

        l = PoissonLayer(scope=Scope([1]), n_nodes=3, l=np.array(l_values))

        for l_layer_node, l_value in zip(l.l, l_values):
            self.assertTrue(torch.allclose(l_layer_node, torch.tensor(l_value)))

        # wrong number of values
        self.assertRaises(
            ValueError,
            PoissonLayer,
            Scope([0]),
            np.array(l_values[:-1]),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            PoissonLayer,
            Scope([0]),
            np.array([l_values for _ in range(3)]),
            n_nodes=3,
        )

        # ---- different scopes -----
        l = PoissonLayer(scope=Scope([1]), n_nodes=3)
        for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, PoissonLayer, Scope([0]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, PoissonLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, PoissonLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = PoissonLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)

        for layer_scope, node_scope in zip(l.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            PoissonLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # Poisson feature type class
        self.assertTrue(
            PoissonLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Poisson]),
                    FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # Poisson feature type instance
        self.assertTrue(
            PoissonLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Poisson(1.0)]),
                    FeatureContext(Scope([1]), [FeatureTypes.Poisson(1.0)]),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            PoissonLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1]), [FeatureTypes.Poisson(1.0)]),
                ]
            )
        )

        # conditional scope
        self.assertFalse(PoissonLayer.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

        # multivariate signature
        self.assertFalse(
            PoissonLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        poisson = PoissonLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
            ]
        )
        self.assertTrue(poisson.scopes_out == [Scope([0]), Scope([1])])

        poisson = PoissonLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Poisson]),
                FeatureContext(Scope([1]), [FeatureTypes.Poisson]),
            ]
        )
        self.assertTrue(poisson.scopes_out == [Scope([0]), Scope([1])])

        poisson = PoissonLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Poisson(l=1.5)]),
                FeatureContext(Scope([1]), [FeatureTypes.Poisson(l=2.0)]),
            ]
        )
        self.assertTrue(poisson.scopes_out == [Scope([0]), Scope([1])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            PoissonLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            PoissonLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            PoissonLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(PoissonLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            PoissonLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Poisson]),
                    FeatureContext(Scope([1]), [FeatureTypes.Poisson]),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        poisson = AutoLeaf(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Poisson(l=1.5)]),
                FeatureContext(Scope([1]), [FeatureTypes.Poisson(l=2.0)]),
            ]
        )
        self.assertTrue(isinstance(poisson, PoissonLayer))
        self.assertTrue(poisson.scopes_out == [Scope([0]), Scope([1])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = PoissonLayer(scope=Scope([1]), l=[0.73, 0.29], n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(torch.allclose(l.l, l_marg.l))

        # ---------- different scopes -----------

        l = PoissonLayer(scope=[Scope([1]), Scope([0])], l=[0.73, 0.29])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, Poisson))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertTrue(torch.allclose(l_marg.l, torch.tensor(0.29)))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, PoissonLayer))
        self.assertEqual(len(l_marg.scopes_out), 1)
        self.assertTrue(torch.allclose(l_marg.l, torch.tensor(0.29)))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(torch.allclose(l.l, l_marg.l))

    def test_layer_dist(self):

        l_values = [0.73, 0.29, 0.5]
        l = PoissonLayer(scope=Scope([1]), l=l_values, n_nodes=3)

        # ----- full dist -----
        dist = l.dist()

        for l_value, l_dist in zip(l_values, dist.rate):
            self.assertTrue(torch.allclose(torch.tensor(l_value).double(), l_dist))

        # ----- partial dist -----
        dist = l.dist([1, 2])

        for l_value, l_dist in zip(l_values[1:], dist.rate):
            self.assertTrue(torch.allclose(torch.tensor(l_value).double(), l_dist))

        dist = l.dist([1, 0])

        for l_value, l_dist in zip(reversed(l_values[:-1]), dist.rate):
            self.assertTrue(torch.allclose(torch.tensor(l_value).double(), l_dist))

    def test_layer_backend_conversion_1(self):

        torch_layer = PoissonLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 0.9, 0.31])
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.l, torch_layer.l.detach().numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

    def test_layer_backend_conversion_2(self):

        base_layer = BasePoissonLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 0.9, 0.31])
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.l, torch_layer.l.detach().numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
