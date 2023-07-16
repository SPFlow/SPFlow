import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.spn import GeometricLayer as BaseGeometricLayer
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.torch.structure.spn import Geometric as GeometricTorch
from spflow.torch.structure.spn import GeometricLayer as GeometricLayerTorch
from spflow.torch.structure.general.layers.leaves.parametric.geometric import updateBackend

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.general.layers.leaves.parametric.general_geometric import GeometricLayer


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_initialization(self):

        # ----- check attributes after correct initialization -----
        p_values = [0.5, 0.3, 0.9]
        l = GeometricLayer(scope=Scope([1]), n_nodes=3, p=p_values)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
        # make sure parameter properties works correctly
        for p_layer_node, p_value in zip(l.p, p_values):
            self.assertTrue(torch.allclose(p_layer_node, torch.tensor(p_value)))

        # ----- float/int parameter values -----
        p_value = 0.73
        l = GeometricLayer(scope=Scope([1]), n_nodes=3, p=p_value)

        for p_layer_node in l.p:
            self.assertTrue(torch.allclose(p_layer_node, torch.tensor(p_value)))

        # ----- list parameter values -----
        p_values = [0.17, 0.8, 0.53]
        l = GeometricLayer(scope=Scope([1]), n_nodes=3, p=p_values)

        for p_layer_node, p_value in zip(l.p, p_values):
            self.assertTrue(torch.allclose(p_layer_node, torch.tensor(p_value)))

        # wrong number of values
        self.assertRaises(ValueError, GeometricLayer, Scope([0]), p_values[:-1], n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            GeometricLayer,
            Scope([0]),
            [p_values for _ in range(3)],
            n_nodes=3,
        )

        # ----- numpy parameter values -----

        p = GeometricLayer(scope=Scope([1]), n_nodes=3, p=np.array(p_values))

        for p_layer_node, p_value in zip(l.p, p_values):
            self.assertTrue(torch.allclose(p_layer_node, torch.tensor(p_value)))

        # wrong number of values
        self.assertRaises(
            ValueError,
            GeometricLayer,
            Scope([0]),
            np.array(p_values[:-1]),
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            GeometricLayer,
            Scope([0]),
            np.array([p_values for _ in range(3)]),
            n_nodes=3,
        )

        # ---- different scopes -----
        l = GeometricLayer(scope=Scope([1]), n_nodes=3)
        for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, GeometricLayer, Scope([0]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, GeometricLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, GeometricLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = GeometricLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)

        for layer_scope, node_scope in zip(l.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)

    def test_accept(self):

        # discrete meta type
        self.assertTrue(
            GeometricLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # feature type instance
        self.assertTrue(
            GeometricLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Geometric]),
                    FeatureContext(Scope([1]), [FeatureTypes.Geometric]),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            GeometricLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1]), [FeatureTypes.Geometric]),
                ]
            )
        )

        # conditional scope
        self.assertFalse(GeometricLayer.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Geometric])]))

        # multivariate signature
        self.assertFalse(
            GeometricLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Geometric, FeatureTypes.Geometric],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        geometric = GeometricLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
            ]
        )
        self.assertTrue(geometric.scopes_out == [Scope([0]), Scope([1])])

        geometric = GeometricLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Geometric]),
                FeatureContext(Scope([1]), [FeatureTypes.Geometric]),
            ]
        )
        self.assertTrue(geometric.scopes_out == [Scope([0]), Scope([1])])

        geometric = GeometricLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Geometric(0.5)]),
                FeatureContext(Scope([1]), [FeatureTypes.Geometric(0.5)]),
            ]
        )
        self.assertTrue(geometric.scopes_out == [Scope([0]), Scope([1])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            GeometricLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            GeometricLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            GeometricLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(GeometricLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            GeometricLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Geometric]),
                    FeatureContext(Scope([1]), [FeatureTypes.Geometric]),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        geometric = AutoLeaf(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Geometric(p=0.75)]),
                FeatureContext(Scope([1]), [FeatureTypes.Geometric(p=0.25)]),
            ]
        )
        self.assertTrue(geometric.scopes_out == [Scope([0]), Scope([1])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = GeometricLayer(scope=Scope([1]), p=[0.73, 0.29], n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(torch.allclose(l.p, l_marg.p))

        # ---------- different scopes -----------

        l = GeometricLayer(scope=[Scope([1]), Scope([0])], p=[0.73, 0.29])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, GeometricTorch))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertTrue(torch.allclose(l_marg.p, torch.tensor(0.29)))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, GeometricLayerTorch))
        self.assertEqual(len(l_marg.scopes_out), 1)
        self.assertTrue(torch.allclose(l_marg.p, torch.tensor(0.29)))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(torch.allclose(l.p, l_marg.p))

    def test_layer_dist(self):

        p_values = [0.73, 0.29, 0.5]
        l = GeometricLayer(scope=Scope([1]), p=p_values, n_nodes=3)

        # ----- full dist -----
        dist = l.dist()

        for p_value, p_dist in zip(p_values, dist.probs):
            self.assertTrue(torch.allclose(torch.tensor(p_value).double(), p_dist))

        # ----- partial dist -----
        dist = l.dist([1, 2])

        for p_value, p_dist in zip(p_values[1:], dist.probs):
            self.assertTrue(torch.allclose(torch.tensor(p_value).double(), p_dist))

        dist = l.dist([1, 0])

        for p_value, p_dist in zip(reversed(p_values[:-1]), dist.probs):
            self.assertTrue(torch.allclose(torch.tensor(p_value).double(), p_dist))

    def test_layer_backend_conversion_1(self):

        torch_layer = GeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.p, torch_layer.p.detach().numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

    def test_layer_backend_conversion_2(self):

        base_layer = BaseGeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.p, torch_layer.p.detach().numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        geometric = GeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
        for backend in backends:
            tl.set_backend(backend)
            geometric_updated = updateBackend(geometric)
            self.assertTrue(np.all(geometric.scopes_out == geometric_updated.scopes_out))
            # check conversion from torch to python
            self.assertTrue(
                np.allclose(
                    np.array([*geometric.get_params()[0]]),
                    np.array([*geometric_updated.get_params()[0]]),
                )
            )




if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
