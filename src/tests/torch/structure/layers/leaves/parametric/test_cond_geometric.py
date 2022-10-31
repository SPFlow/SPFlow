from spflow.torch.structure.layers.leaves.parametric.cond_geometric import (
    CondGeometricLayer,
    marginalize,
    toTorch,
    toBase,
)
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.torch.structure.nodes.leaves.parametric.cond_geometric import (
    CondGeometric,
)
from spflow.base.structure.layers.leaves.parametric.cond_geometric import (
    CondGeometricLayer as BaseCondGeometricLayer,
)
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
import torch
import numpy as np
import unittest


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_initialization(self):

        # ----- check attributes after correct initialization -----
        l = CondGeometricLayer(scope=Scope([1], [0]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(l.scopes_out == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])])
        )

        # ---- different scopes -----
        l = CondGeometricLayer(scope=Scope([1], [0]), n_nodes=3)
        for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, CondGeometricLayer, Scope([0], [1]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, CondGeometricLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, CondGeometricLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
        l = CondGeometricLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3)

        for layer_scope, node_scope in zip(l.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)

        # -----number of cond_f functions -----
        CondGeometricLayer(
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"p": 0.5}, lambda data: {"p": 0.5}],
        )
        self.assertRaises(
            ValueError,
            CondGeometricLayer,
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"p": 0.5}],
        )

    def test_retrieve_params(self):

        # ----- float/int parameter values -----
        p_value = 0.73
        l = CondGeometricLayer(
            scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"p": p_value}
        )

        for p_layer_node in l.retrieve_params(
            torch.tensor([[1]]), DispatchContext()
        ):
            self.assertTrue(torch.allclose(p_layer_node, torch.tensor(p_value)))

        # ----- list parameter values -----
        p_values = [0.17, 0.8, 0.53]
        l = CondGeometricLayer(
            scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"p": p_values}
        )

        for p_layer_node, p_value in zip(
            l.retrieve_params(torch.tensor([[1]]), DispatchContext()), p_values
        ):
            self.assertTrue(torch.allclose(p_layer_node, torch.tensor(p_value)))

        # wrong number of values
        l.set_cond_f(lambda data: {"p": p_values[:-1]})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        # wrong number of dimensions (nested list)
        l.set_cond_f(lambda data: {"p": [p_values for _ in range(3)]})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        # ----- numpy parameter values -----
        l.set_cond_f(lambda data: {"p": np.array(p_values)})
        for p_node, p_actual in zip(
            l.retrieve_params(torch.tensor([[1.0]]), DispatchContext()),
            p_values,
        ):
            self.assertTrue(p_node == p_actual)

        # wrong number of values
        l.set_cond_f(lambda data: {"p": np.array(p_values[:-1])})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        # wrong number of dimensions (nested list)
        l.set_cond_f(lambda data: {"p": np.array([p_values for _ in range(3)])})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

    def test_accept(self):

        # discrete meta type
        self.assertTrue(CondGeometricLayer.accepts([([FeatureTypes.Discrete], Scope([0], [2])), ([FeatureTypes.Discrete], Scope([1], [2]))]))

        # Geometric feature type class
        self.assertTrue(CondGeometricLayer.accepts([([FeatureTypes.Geometric], Scope([0], [2])), ([FeatureTypes.Geometric], Scope([1], [2]))]))

        # Geometric feature type instance
        self.assertTrue(CondGeometricLayer.accepts([([FeatureTypes.Geometric(0.5)], Scope([0], [2])), ([FeatureTypes.Geometric(0.5)], Scope([1], [2]))]))

        # invalid feature type
        self.assertFalse(CondGeometricLayer.accepts([([FeatureTypes.Continuous], Scope([0], [2])), ([FeatureTypes.Discrete], Scope([1], [2]))]))

        # non-conditional scope
        self.assertFalse(CondGeometricLayer.accepts([([FeatureTypes.Discrete], Scope([0]))]))

        # scope length does not match number of types
        self.assertFalse(CondGeometricLayer.accepts([([FeatureTypes.Discrete], Scope([0, 1], [2]))]))

        # multivariate signature
        self.assertFalse(CondGeometricLayer.accepts([([FeatureTypes.Discrete, FeatureTypes.Discrete], Scope([0, 1], [2]))]))

    def test_initialization_from_signatures(self):

        geometric = CondGeometricLayer.from_signatures([([FeatureTypes.Discrete], Scope([0], [2])), ([FeatureTypes.Discrete], Scope([1], [2]))])
        self.assertTrue(geometric.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        geometric = CondGeometricLayer.from_signatures([([FeatureTypes.Geometric], Scope([0], [2])), ([FeatureTypes.Geometric], Scope([1], [2]))])
        self.assertTrue(geometric.scopes_out == [Scope([0], [2]), Scope([1], [2])])
    
        geometric = CondGeometricLayer.from_signatures([([FeatureTypes.Geometric(p=0.75)], Scope([0], [2])), ([FeatureTypes.Geometric(p=0.25)], Scope([1], [2]))])
        self.assertTrue(geometric.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(ValueError, CondGeometricLayer.from_signatures, [([FeatureTypes.Continuous], Scope([0], [2]))])

        # non-conditional scope
        self.assertRaises(ValueError, CondGeometricLayer.from_signatures, [([FeatureTypes.Discrete], Scope([0]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, CondGeometricLayer.from_signatures, [([FeatureTypes.Discrete], Scope([0, 1], [2]))])

        # multivariate signature
        self.assertRaises(ValueError, CondGeometricLayer.from_signatures, [([FeatureTypes.Discrete, FeatureTypes.Discrete], Scope([0, 1], [2]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondGeometricLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(CondGeometricLayer, AutoLeaf.infer([([FeatureTypes.Geometric], Scope([0], [2])), ([FeatureTypes.Geometric], Scope([1], [2]))]))

        # make sure AutoLeaf can return correctly instantiated object
        geometric = AutoLeaf([([FeatureTypes.Geometric(p=0.75)], Scope([0], [2])), ([FeatureTypes.Geometric(p=0.25)], Scope([1], [2]))])
        self.assertTrue(geometric.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = CondGeometricLayer(scope=Scope([1], [0]), n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

        # ---------- different scopes -----------

        l = CondGeometricLayer(scope=[Scope([1], [2]), Scope([0], [2])])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, CondGeometric))
        self.assertEqual(l_marg.scope, Scope([0], [2]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, CondGeometricLayer))
        self.assertEqual(len(l_marg.scopes_out), 1)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])

    def test_layer_dist(self):

        p_values = torch.tensor([0.73, 0.29, 0.5])
        l = CondGeometricLayer(scope=Scope([1], [0]), n_nodes=3)

        # ----- full dist -----
        dist = l.dist(p_values)

        for p_value, p_dist in zip(p_values, dist.probs):
            self.assertTrue(torch.allclose(p_value, p_dist))

        # ----- partial dist -----
        dist = l.dist(p_values, [1, 2])

        for p_value, p_dist in zip(p_values[1:], dist.probs):
            self.assertTrue(torch.allclose(p_value, p_dist))

        dist = l.dist(p_values, [1, 0])

        for p_value, p_dist in zip(reversed(p_values[:-1]), dist.probs):
            self.assertTrue(torch.allclose(p_value, p_dist))

    def test_layer_backend_conversion_1(self):

        torch_layer = CondGeometricLayer(
            scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])]
        )
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

    def test_layer_backend_conversion_2(self):

        base_layer = BaseCondGeometricLayer(
            scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])]
        )
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
