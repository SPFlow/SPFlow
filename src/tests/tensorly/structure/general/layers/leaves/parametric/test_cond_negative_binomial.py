import unittest

import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_unsqueeze,tl_allclose

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.spn import (
    marginalize,
)
from spflow.tensorly.structure.general.nodes.leaves import CondNegativeBinomial
from spflow.tensorly.structure.general.layers.leaves import CondNegativeBinomialLayer
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = CondNegativeBinomialLayer(scope=Scope([1], [0]), n_nodes=3, n=2)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(tl.all(l.scopes_out == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])]))

        # ----- n initialization -----
        l = CondNegativeBinomialLayer(
            scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
            n=[3, 5, 2],
        )
        # wrong number of n values
        self.assertRaises(
            ValueError,
            CondNegativeBinomialLayer,
            [Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
            n=[3, 5],
            n_nodes=3,
        )
        # wrong shape of n values
        self.assertRaises(
            ValueError,
            CondNegativeBinomialLayer,
            [Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
            [[3, 5] for _ in range(3)],
            n_nodes=3,
        )

        # n numpy array
        l = CondNegativeBinomialLayer(
            scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
            n=tl.tensor([3, 5, 2]),
        )
        # wrong number of n values
        self.assertRaises(
            ValueError,
            CondNegativeBinomialLayer,
            scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
            n=tl.tensor([3, 5]),
        )
        # wrong shape of n values
        self.assertRaises(
            ValueError,
            CondNegativeBinomialLayer,
            scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
            n=tl.tensor([[3, 5, 2]]),
        )

        # ---- different scopes -----
        l = CondNegativeBinomialLayer(
            scope=[Scope([0], [3]), Scope([1], [3]), Scope([2], [3])],
            n=5,
            n_nodes=3,
        )
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.n, 5)
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, CondNegativeBinomialLayer, Scope([0], [1]), 2, n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, CondNegativeBinomialLayer, Scope([]), 2, n_nodes=3)
        self.assertRaises(ValueError, CondNegativeBinomialLayer, [], 2, n_nodes=3)

        # ----- invalid values for 'n' over same scope -----
        self.assertRaises(
            ValueError,
            CondNegativeBinomialLayer,
            Scope([0], [1]),
            n=[2, 5],
            n_nodes=2,
        )

        # ----- individual scopes and parameters -----
        scopes = [Scope([1], [2]), Scope([0], [2])]
        l = CondNegativeBinomialLayer(scope=[Scope([1], [2]), Scope([0], [2])], n=2, n_nodes=3)

        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

        # -----number of cond_f functions -----
        CondNegativeBinomialLayer(
            Scope([0], [1]),
            n=3,
            n_nodes=2,
            cond_f=[lambda data: {"p": 0.5}, lambda data: {"p": 0.5}],
        )
        self.assertRaises(
            ValueError,
            CondNegativeBinomialLayer,
            Scope([0], [1]),
            n=3,
            n_nodes=2,
            cond_f=[lambda data: {"p": 0.5}],
        )

    def test_retrieve_params(self):

        # ----- float/int parameter values -----
        n_value = 2
        p_value = 0.5
        l = CondNegativeBinomialLayer(
            scope=Scope([1], [0]),
            n_nodes=3,
            n=n_value,
            cond_f=lambda data: {"p": p_value},
        )

        for p in l.retrieve_params(tl.tensor([[1.0]]), DispatchContext()):
            self.assertTrue(p == p_value)

        # ----- list parameter values -----
        n_values = [1, 5, 4]
        p_values = [0.25, 0.5, 0.3]

        l = CondNegativeBinomialLayer(
            scope=[Scope([0], [3]), Scope([1], [3]), Scope([2], [3])],
            n_nodes=3,
            n=n_values,
            cond_f=lambda data: {"p": p_values},
        )

        for n_actual, p_actual, n_node, p_node in zip(
            n_values,
            p_values,
            l.n,
            l.retrieve_params(tl.tensor([[1.0]]), DispatchContext()),
        ):
            self.assertTrue(n_node == n_actual)
            self.assertTrue(p_node == p_actual)

        # wrong number of values
        l.set_cond_f(lambda data: {"p": p_values[:-1]})
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())

        # wrong number of dimensions (nested list)
        l.set_cond_f(lambda data: {"p": [p_values for _ in range(3)]})
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())

        # ----- numpy parameter values -----
        l.set_cond_f(lambda data: {"p": tl.tensor(p_values)})
        for p_node, p_actual in zip(l.retrieve_params(tl.tensor([[1.0]]), DispatchContext()), p_values):
            self.assertTrue(p_node == p_actual)

        # wrong number of values
        l.set_cond_f(lambda data: {"p": tl.tensor(p_values[:-1])})
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())

        # wrong number of dimensions (nested list)
        l.set_cond_f(lambda data: {"p": tl.tensor([p_values for _ in range(3)])})
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())

        l.set_cond_f(lambda data: {"p": tl_unsqueeze(tl.tensor(p_values), 0)})
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())

        l.set_cond_f(lambda data: {"p": tl_unsqueeze(tl.tensor(p_values), 1)})
        self.assertRaises(ValueError, l.retrieve_params, tl.tensor([[1]]), DispatchContext())

    def test_accept(self):

        # discrete meta type (should reject)
        self.assertFalse(
            CondNegativeBinomialLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # feature type instance
        self.assertTrue(
            CondNegativeBinomialLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.NegativeBinomial(n=3)]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.NegativeBinomial(n=3)]),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondNegativeBinomialLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.NegativeBinomial(n=3)]),
                ]
            )
        )

        # non-conditional scope
        self.assertFalse(
            CondNegativeBinomialLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.NegativeBinomial(n=3)])])
        )

        # multivariate signature
        self.assertFalse(
            CondNegativeBinomialLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [
                            FeatureTypes.NegativeBinomial(n=3),
                            FeatureTypes.Binomial(n=3),
                        ],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        negative_binomial = CondNegativeBinomialLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.NegativeBinomial(n=3)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.NegativeBinomial(n=5)]),
            ]
        )
        self.assertTrue(negative_binomial.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        # ----- invalid arguments -----

        # discrete meta type
        self.assertRaises(
            ValueError,
            CondNegativeBinomialLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondNegativeBinomialLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondNegativeBinomialLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.NegativeBinomial(3)])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondNegativeBinomialLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [
                        FeatureTypes.NegativeBinomial(3),
                        FeatureTypes.NegativeBinomial(5),
                    ],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondNegativeBinomialLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondNegativeBinomialLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(Scope([0], [1]), [FeatureTypes.NegativeBinomial(n=3)]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.NegativeBinomial(n=5)]),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        negative_binomial = AutoLeaf(
            [
                FeatureContext(
                    Scope([0], [2]),
                    [FeatureTypes.NegativeBinomial(n=3, p=0.75)],
                ),
                FeatureContext(
                    Scope([1], [2]),
                    [FeatureTypes.NegativeBinomial(n=5, p=0.25)],
                ),
            ]
        )
        self.assertTrue(negative_binomial.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = CondNegativeBinomialLayer(scope=Scope([1], [0]), n=2, n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])
        self.assertTrue(tl.all(l.n == l_marg.n))

        # ---------- different scopes -----------

        l = CondNegativeBinomialLayer(scope=[Scope([1], [2]), Scope([0], [2])], n=[2, 6])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, CondNegativeBinomial))
        self.assertEqual(l_marg.scope, Scope([0], [2]))
        self.assertEqual(l_marg.n, tl.tensor([6]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, CondNegativeBinomialLayer))
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertEqual(l_marg.n, tl.tensor([6]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])
        self.assertTrue(tl.all(l.n == l_marg.n))

    def test_get_params(self):

        l = CondNegativeBinomialLayer(scope=Scope([1], [0]), n=[2, 2], n_nodes=2)

        n, *others = l.get_params()

        self.assertTrue(len(others) == 0)
        self.assertTrue(tl_allclose(n, tl.tensor([2, 2])))


if __name__ == "__main__":
    unittest.main()
