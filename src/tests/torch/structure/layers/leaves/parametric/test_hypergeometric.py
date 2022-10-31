from spflow.torch.structure.layers.leaves.parametric.hypergeometric import (
    HypergeometricLayer,
    marginalize,
    toTorch,
    toBase,
)
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.torch.structure.nodes.leaves.parametric.hypergeometric import (
    Hypergeometric,
)
from spflow.base.structure.layers.leaves.parametric.hypergeometric import (
    HypergeometricLayer as BaseHypergeometricLayer,
)
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
        N_values = [10, 5, 7]
        M_values = [8, 2, 6]
        n_values = [3, 4, 5]
        l = HypergeometricLayer(
            scope=[Scope([1]), Scope([0]), Scope([2])],
            N=N_values,
            M=M_values,
            n=n_values,
        )
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(l.scopes_out == [Scope([1]), Scope([0]), Scope([2])])
        )
        # make sure parameter properties works correctly
        for (
            N_layer_node,
            M_layer_node,
            n_layer_node,
            N_value,
            M_value,
            n_value,
        ) in zip(l.N, l.M, l.n, N_values, M_values, n_values):
            self.assertTrue(torch.allclose(N_layer_node, torch.tensor(N_value)))
            self.assertTrue(torch.allclose(M_layer_node, torch.tensor(M_value)))
            self.assertTrue(torch.allclose(n_layer_node, torch.tensor(n_value)))

        # ----- float/int parameter values -----
        N_value = 6
        M_value = 4
        n_value = 5
        l = HypergeometricLayer(
            scope=Scope([1]), n_nodes=3, N=N_value, M=M_value, n=n_value
        )

        for N_layer_node, M_layer_node, n_layer_node in zip(l.N, l.M, l.n):
            self.assertTrue(torch.all(N_layer_node == N_value))
            self.assertTrue(torch.all(M_layer_node == M_value))
            self.assertTrue(torch.all(n_layer_node == n_value))

        # ----- list parameter values -----
        N_values = [10, 5, 7]
        M_values = [8, 2, 6]
        n_values = [3, 4, 5]
        l = HypergeometricLayer(
            scope=[Scope([0]), Scope([1]), Scope([2])],
            N=N_values,
            M=M_values,
            n=n_values,
        )

        for (
            N_layer_node,
            M_layer_node,
            n_layer_node,
            N_value,
            M_value,
            n_value,
        ) in zip(l.N, l.M, l.n, N_values, M_values, n_values):
            self.assertTrue(torch.allclose(N_layer_node, torch.tensor(N_value)))
            self.assertTrue(torch.allclose(M_layer_node, torch.tensor(M_value)))
            self.assertTrue(torch.allclose(n_layer_node, torch.tensor(n_value)))

        # wrong number of values
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([0]), Scope([1]), Scope([2])],
            N_values[:-1],
            M_values,
            n_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([0]), Scope([1]), Scope([2])],
            N_values,
            M_values[:-1],
            n_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([0]), Scope([1]), Scope([2])],
            N_values,
            M_values,
            n_values[:-1],
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([0]), Scope([1]), Scope([2])],
            [N_values for _ in range(3)],
            M_values,
            n_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([0]), Scope([1]), Scope([2])],
            N_values,
            [M_values for _ in range(3)],
            n_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([0]), Scope([1]), Scope([2])],
            N_values,
            M_values,
            [n_values for _ in range(3)],
            n_nodes=3,
        )

        # ----- numpy parameter values -----

        l = HypergeometricLayer(
            scope=[Scope([0]), Scope([1]), Scope([2])],
            N=np.array(N_values),
            M=np.array(M_values),
            n=np.array(n_values),
        )

        for (
            N_layer_node,
            M_layer_node,
            n_layer_node,
            N_value,
            M_value,
            n_value,
        ) in zip(l.N, l.M, l.n, N_values, M_values, n_values):
            self.assertTrue(torch.allclose(N_layer_node, torch.tensor(N_value)))
            self.assertTrue(torch.allclose(M_layer_node, torch.tensor(M_value)))
            self.assertTrue(torch.allclose(n_layer_node, torch.tensor(n_value)))

        # wrong number of values
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([0]), Scope([1]), Scope([2])],
            np.array(N_values[:-1]),
            np.array(M_values),
            np.array(n_values),
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([0]), Scope([1]), Scope([2])],
            np.array(N_values),
            np.array(M_values[:-1]),
            np.array(n_values),
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([0]), Scope([1]), Scope([2])],
            np.array(N_values),
            np.array(M_values),
            np.array(n_values[:-1]),
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([0]), Scope([1]), Scope([2])],
            np.array([N_values for _ in range(3)]),
            np.array(M_values),
            np.array(n_values),
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([0]), Scope([1]), Scope([2])],
            np.array(N_values),
            np.array([M_values for _ in range(3)]),
            np.array(n_values),
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([0]), Scope([1]), Scope([2])],
            np.array(N_values),
            np.array(M_values),
            np.array([n_values for _ in range(3)]),
        )

        # ---- different scopes -----
        l = HypergeometricLayer(scope=Scope([1]), n_nodes=3, N=5, M=3, n=2)
        for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            Scope([0]),
            n_nodes=0,
            N=5,
            M=3,
            n=2,
        )

        # ----- invalid scope -----
        self.assertRaises(
            ValueError, HypergeometricLayer, Scope([]), n_nodes=3, N=5, M=3, n=2
        )
        self.assertRaises(
            ValueError, HypergeometricLayer, [], n_nodes=3, N=5, M=3, n=2
        )

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = HypergeometricLayer(
            scope=[Scope([1]), Scope([0])], n_nodes=3, N=5, M=3, n=2
        )

        for layer_scope, node_scope in zip(l.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)

    def test_accept(self):

        # discrete meta type (should reject)
        self.assertFalse(HypergeometricLayer.accepts([([FeatureTypes.Discrete], Scope([0])), ([FeatureTypes.Discrete], Scope([1]))]))

        # Bernoulli feature type class (should reject)
        self.assertFalse(HypergeometricLayer.accepts([([FeatureTypes.Hypergeometric], Scope([0])), ([FeatureTypes.Hypergeometric(N=4, M=2, n=3)], Scope([1]))]))

        # Bernoulli feature type instance
        self.assertTrue(HypergeometricLayer.accepts([([FeatureTypes.Hypergeometric(N=4, M=2, n=3)], Scope([0])), ([FeatureTypes.Hypergeometric(N=6, M=5, n=4)], Scope([1]))]))

        # invalid feature type
        self.assertFalse(HypergeometricLayer.accepts([([FeatureTypes.Continuous], Scope([0])), ([FeatureTypes.Hypergeometric(N=6, M=5, n=4)], Scope([1]))]))

        # conditional scope
        self.assertFalse(HypergeometricLayer.accepts([([FeatureTypes.Hypergeometric(N=4, M=2, n=3)], Scope([0], [1]))]))

        # scope length does not match number of types
        self.assertFalse(HypergeometricLayer.accepts([([FeatureTypes.Hypergeometric(N=4, M=2, n=3)], Scope([0, 1]))]))

        # multivariate signature
        self.assertFalse(HypergeometricLayer.accepts([([FeatureTypes.Hypergeometric(N=4, M=2, n=3), FeatureTypes.Hypergeometric(N=4, M=2, n=3)], Scope([0, 1]))]))

    def test_initialization_from_signatures(self):

        hypergeometric = HypergeometricLayer.from_signatures([([FeatureTypes.Hypergeometric(N=4, M=2, n=3)], Scope([0])), ([FeatureTypes.Hypergeometric(N=6, M=5, n=4)], Scope([1]))])
        self.assertTrue(torch.all(hypergeometric.N == torch.tensor([4, 6])))
        self.assertTrue(torch.all(hypergeometric.M == torch.tensor([2, 5])))
        self.assertTrue(torch.all(hypergeometric.n == torch.tensor([3, 4])))
        self.assertTrue(hypergeometric.scopes_out == [Scope([0]), Scope([1])])

        # ----- invalid arguments -----

        # discrete meta type
        self.assertRaises(ValueError, HypergeometricLayer.from_signatures, [([FeatureTypes.Discrete], Scope([0]))])

        # Bernoulli feature type class
        self.assertRaises(ValueError, HypergeometricLayer.from_signatures, [([FeatureTypes.Hypergeometric], Scope([0]))])

        # invalid feature type
        self.assertRaises(ValueError, HypergeometricLayer.from_signatures, [([FeatureTypes.Continuous], Scope([0]))])

        # conditional scope
        self.assertRaises(ValueError, HypergeometricLayer.from_signatures, [([FeatureTypes.Discrete], Scope([0], [1]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, HypergeometricLayer.from_signatures, [([FeatureTypes.Discrete], Scope([0, 1]))])

        # multivariate signature
        self.assertRaises(ValueError, Hypergeometric.from_signatures, [([FeatureTypes.Discrete, FeatureTypes.Discrete], Scope([0, 1]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(HypergeometricLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(HypergeometricLayer, AutoLeaf.infer([([FeatureTypes.Hypergeometric(N=4, M=2, n=3)], Scope([0])), ([FeatureTypes.Hypergeometric(N=6, M=5, n=4)], Scope([1]))]))

        # make sure AutoLeaf can return correctly instantiated object
        hypergeometric = AutoLeaf([([FeatureTypes.Hypergeometric(N=4, M=2, n=3)], Scope([0])), ([FeatureTypes.Hypergeometric(N=6, M=5, n=4)], Scope([1]))])
        self.assertTrue(isinstance(hypergeometric, HypergeometricLayer))
        self.assertTrue(torch.all(hypergeometric.N == torch.tensor([4, 6])))
        self.assertTrue(torch.all(hypergeometric.M == torch.tensor([2, 5])))
        self.assertTrue(torch.all(hypergeometric.n == torch.tensor([3, 4])))
        self.assertTrue(hypergeometric.scopes_out == [Scope([0]), Scope([1])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = HypergeometricLayer(scope=Scope([1]), n_nodes=2, N=5, M=3, n=4)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(torch.allclose(l.N, l_marg.N))
        self.assertTrue(torch.allclose(l.M, l_marg.M))
        self.assertTrue(torch.allclose(l.n, l_marg.n))

        # ---------- different scopes -----------

        l = HypergeometricLayer(
            scope=[Scope([1]), Scope([0])], N=[5, 7], M=[3, 6], n=[4, 3]
        )

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, Hypergeometric))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertTrue(torch.allclose(l_marg.N, torch.tensor(7)))
        self.assertTrue(torch.allclose(l_marg.M, torch.tensor(6)))
        self.assertTrue(torch.allclose(l_marg.n, torch.tensor(3)))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, HypergeometricLayer))
        self.assertEqual(len(l_marg.scopes_out), 1)
        self.assertTrue(torch.allclose(l_marg.N, torch.tensor(7)))
        self.assertTrue(torch.allclose(l_marg.M, torch.tensor(6)))
        self.assertTrue(torch.allclose(l_marg.n, torch.tensor(3)))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(torch.allclose(l.N, l_marg.N))
        self.assertTrue(torch.allclose(l.M, l_marg.M))
        self.assertTrue(torch.allclose(l.n, l_marg.n))

    def test_layer_backend_conversion_1(self):

        torch_layer = HypergeometricLayer(
            scope=[Scope([0]), Scope([1]), Scope([0])],
            N=[10, 5, 10],
            M=[8, 2, 8],
            n=[3, 4, 3],
        )
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.N, torch_layer.N.numpy()))
        self.assertTrue(np.allclose(base_layer.M, torch_layer.M.numpy()))
        self.assertTrue(np.allclose(base_layer.n, torch_layer.n.numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

    def test_layer_backend_conversion_2(self):

        base_layer = BaseHypergeometricLayer(
            scope=[Scope([0]), Scope([1]), Scope([0])],
            N=[10, 5, 10],
            M=[8, 2, 8],
            n=[3, 4, 3],
        )
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.N, torch_layer.N.numpy()))
        self.assertTrue(np.allclose(base_layer.M, torch_layer.M.numpy()))
        self.assertTrue(np.allclose(base_layer.n, torch_layer.n.numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
