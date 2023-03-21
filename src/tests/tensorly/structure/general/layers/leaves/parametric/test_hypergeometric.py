import unittest

import tensorly as tl

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.spn import Hypergeometric, HypergeometricLayer, marginalize
from spflow.meta.data import FeatureContext, FeatureTypes, Scope


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = HypergeometricLayer(scope=Scope([1]), n_nodes=3, N=1, M=1, n=1)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(tl.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
        # make sure parameter properties works correctly
        N_values = l.N
        M_values = l.M
        n_values = l.n
        for node, node_N, node_M, node_n in zip(l.nodes, N_values, M_values, n_values):
            self.assertTrue(tl.all(node.N == node_N))
            self.assertTrue(tl.all(node.M == node_M))
            self.assertTrue(tl.all(node.n == node_n))

        # ----- float/int parameter values -----
        N_value = 5
        M_value = 3
        n_value = 2
        l = HypergeometricLayer(scope=Scope([1]), n_nodes=3, N=N_value, M=M_value, n=n_value)

        for node in l.nodes:
            self.assertTrue(tl.all(node.N == N_value))
            self.assertTrue(tl.all(node.M == M_value))
            self.assertTrue(tl.all(node.n == n_value))

        # ----- list parameter values -----
        N_values = [3, 5, 4]
        M_values = [2, 1, 3]
        n_values = [2, 2, 3]
        l = HypergeometricLayer(
            scope=[Scope([1]), Scope([0]), Scope([2])],
            n_nodes=3,
            N=N_values,
            M=M_values,
            n=n_values,
        )

        for node, node_N, node_M, node_n in zip(l.nodes, N_values, M_values, n_values):
            self.assertTrue(tl.all(node.N == node_N))
            self.assertTrue(tl.all(node.M == node_M))
            self.assertTrue(tl.all(node.n == node_n))

        # wrong number of values
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            N_values[:-1],
            M_values,
            n_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            N_values,
            M_values[:-1],
            n_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            N_values,
            M_values,
            n_values[:-1],
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            [N_values for _ in range(3)],
            M_values,
            n_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            N_values,
            [M_values for _ in range(3)],
            n_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            N_values,
            M_values,
            [n_values for _ in range(3)],
            n_nodes=3,
        )

        # ----- numpy parameter values -----

        l = HypergeometricLayer(
            scope=[Scope([1]), Scope([0]), Scope([2])],
            N=tl.tensor(N_values),
            M=tl.tensor(M_values),
            n=tl.tensor(n_values),
        )

        for node, node_N, node_M, node_n in zip(l.nodes, N_values, M_values, n_values):
            self.assertTrue(tl.all(node.N == node_N))
            self.assertTrue(tl.all(node.M == node_M))
            self.assertTrue(tl.all(node.n == node_n))

        # wrong number of values
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            tl.tensor(N_values[:-1]),
            tl.tensor(M_values),
            tl.tensor(n_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            tl.tensor(N_values),
            tl.tensor(M_values[:-1]),
            tl.tensor(n_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            tl.tensor(N_values),
            tl.tensor(M_values),
            tl.tensor(n_values[:-1]),
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            tl.tensor([N_values for _ in range(3)]),
            tl.tensor(M_values),
            tl.tensor(n_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            tl.tensor(N_values),
            tl.tensor([M_values for _ in range(3)]),
            tl.tensor(n_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            [Scope([1]), Scope([0]), Scope([2])],
            tl.tensor(N_values),
            tl.tensor(M_values),
            tl.tensor([n_values for _ in range(3)]),
            n_nodes=3,
        )

        # ---- different scopes -----
        l = HypergeometricLayer(scope=Scope([1]), N=1, M=1, n=1, n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(
            ValueError,
            HypergeometricLayer,
            Scope([0]),
            N=1,
            M=1,
            n=1,
            n_nodes=0,
        )

        # ----- invalid scope -----
        self.assertRaises(ValueError, HypergeometricLayer, Scope([]), N=1, M=1, n=1, n_nodes=3)
        self.assertRaises(ValueError, HypergeometricLayer, [], N=1, M=1, n=1, n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = HypergeometricLayer(scope=[Scope([1]), Scope([0])], N=1, M=1, n=1, n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

    def test_accept(self):

        # discrete meta type (should reject)
        self.assertFalse(
            HypergeometricLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # feature type instance
        self.assertTrue(
            HypergeometricLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)]),
                    FeatureContext(Scope([1]), [FeatureTypes.Hypergeometric(N=6, M=5, n=4)]),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            HypergeometricLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1]), [FeatureTypes.Hypergeometric(N=6, M=5, n=4)]),
                ]
            )
        )

        # conditional scope
        self.assertFalse(
            HypergeometricLayer.accepts(
                [
                    FeatureContext(
                        Scope([0], [1]),
                        [FeatureTypes.Hypergeometric(N=4, M=2, n=3)],
                    )
                ]
            )
        )

        # multivariate signature
        self.assertFalse(
            HypergeometricLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [
                            FeatureTypes.Hypergeometric(N=4, M=2, n=3),
                            FeatureTypes.Hypergeometric(N=4, M=2, n=3),
                        ],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        hypergeometric = HypergeometricLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)]),
                FeatureContext(Scope([1]), [FeatureTypes.Hypergeometric(N=6, M=5, n=4)]),
            ]
        )
        self.assertTrue(tl.all(hypergeometric.N == tl.tensor([4, 6])))
        self.assertTrue(tl.all(hypergeometric.M == tl.tensor([2, 5])))
        self.assertTrue(tl.all(hypergeometric.n == tl.tensor([3, 4])))
        self.assertTrue(hypergeometric.scopes_out == [Scope([0]), Scope([1])])

        # ----- invalid arguments -----

        # discrete meta type
        self.assertRaises(
            ValueError,
            HypergeometricLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # invalid feature type
        self.assertRaises(
            ValueError,
            HypergeometricLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            HypergeometricLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            Hypergeometric.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(HypergeometricLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            HypergeometricLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)]),
                    FeatureContext(Scope([1]), [FeatureTypes.Hypergeometric(N=6, M=5, n=4)]),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        hypergeometric = AutoLeaf(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Hypergeometric(N=4, M=2, n=3)]),
                FeatureContext(Scope([1]), [FeatureTypes.Hypergeometric(N=6, M=5, n=4)]),
            ]
        )
        self.assertTrue(isinstance(hypergeometric, HypergeometricLayer))
        self.assertTrue(tl.all(hypergeometric.N == tl.tensor([4, 6])))
        self.assertTrue(tl.all(hypergeometric.M == tl.tensor([2, 5])))
        self.assertTrue(tl.all(hypergeometric.n == tl.tensor([3, 4])))
        self.assertTrue(hypergeometric.scopes_out == [Scope([0]), Scope([1])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = HypergeometricLayer(scope=Scope([1]), N=4, M=2, n=3, n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(tl.all(l.N == l_marg.N))
        self.assertTrue(tl.all(l.M == l_marg.M))
        self.assertTrue(tl.all(l.n == l_marg.n))

        # ---------- different scopes -----------

        l = HypergeometricLayer(scope=[Scope([1]), Scope([0])], N=[2, 6], M=[2, 4], n=[2, 5])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, Hypergeometric))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertEqual(l_marg.N, tl.tensor([6]))
        self.assertEqual(l_marg.M, tl.tensor([4]))
        self.assertEqual(l_marg.n, tl.tensor([5]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, HypergeometricLayer))
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertEqual(l_marg.N, tl.tensor([6]))
        self.assertEqual(l_marg.M, tl.tensor([4]))
        self.assertEqual(l_marg.n, tl.tensor([5]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(tl.all(l.N == l_marg.N))
        self.assertTrue(tl.all(l.M == l_marg.M))
        self.assertTrue(tl.all(l.n == l_marg.n))

    def test_get_params(self):

        layer = HypergeometricLayer(scope=Scope([1]), N=5, M=3, n=4, n_nodes=2)

        N, M, n, *others = layer.get_params()

        self.assertTrue(len(others) == 0)
        self.assertTrue(tl.allclose(N, tl.tensor([5, 5])))
        self.assertTrue(tl.allclose(M, tl.tensor([3, 3])))
        self.assertTrue(tl.allclose(n, tl.tensor([4, 4])))


if __name__ == "__main__":
    unittest.main()
