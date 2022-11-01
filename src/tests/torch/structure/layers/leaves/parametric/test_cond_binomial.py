from spflow.torch.structure.layers.leaves.parametric.cond_binomial import (
    CondBinomialLayer,
    marginalize,
    toTorch,
    toBase,
)
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.torch.structure.nodes.leaves.parametric.cond_binomial import (
    CondBinomial,
)
from spflow.base.structure.layers.leaves.parametric.cond_binomial import (
    CondBinomialLayer as BaseCondBinomialLayer,
)
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
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
        n_values = [3, 2, 7]
        l = CondBinomialLayer(
            scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
            n=n_values,
        )
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(
                l.scopes_out
                == [Scope([1], [3]), Scope([0], [3]), Scope([2], [3])]
            )
        )
        # make sure parameter properties works correctly
        for n_layer_node, n_value in zip(l.n, n_values):
            self.assertTrue(torch.allclose(n_layer_node, torch.tensor(n_value)))

        # ----- n initialization -----
        l = CondBinomialLayer(
            scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
            n=[3, 5, 2],
        )
        # wrong number of n values
        self.assertRaises(
            ValueError,
            CondBinomialLayer,
            scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
            n=[3, 5],
        )
        # wrong shape of n values
        self.assertRaises(
            ValueError,
            CondBinomialLayer,
            scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
            n=[[3, 5, 2]],
        )

        # n numpy array
        l = CondBinomialLayer(
            scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
            n=np.array([3, 5, 2]),
        )
        # wrong number of n values
        self.assertRaises(
            ValueError,
            CondBinomialLayer,
            scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
            n=np.array([3, 5]),
        )
        # wrong shape of n values
        self.assertRaises(
            ValueError,
            CondBinomialLayer,
            scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
            n=np.array([[3, 5, 2]]),
        )

        # ---- different scopes -----
        l = CondBinomialLayer(scope=Scope([1], [0]), n_nodes=3, n=2)
        for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(
            ValueError, CondBinomialLayer, Scope([0], [1]), n_nodes=0, n=2
        )

        # ----- invalid scope -----
        self.assertRaises(
            ValueError, CondBinomialLayer, Scope([]), n_nodes=3, n=2
        )
        self.assertRaises(ValueError, CondBinomialLayer, [], n_nodes=3, n=2)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
        l = CondBinomialLayer(
            scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3, n=2
        )

        for layer_scope, node_scope in zip(l.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)

        # -----number of cond_f functions -----
        CondBinomialLayer(
            Scope([0], [1]),
            n=2,
            n_nodes=2,
            cond_f=[lambda data: {"p": 0.5}, lambda data: {"p": 0.5}],
        )
        self.assertRaises(
            ValueError,
            CondBinomialLayer,
            Scope([0], [1]),
            n=2,
            n_nodes=2,
            cond_f=[lambda data: {"p": 0.5}],
        )

    def test_retrieve_params(self):

        # ----- float/int parameter values -----
        n_value = 5
        p_value = 0.13
        l = CondBinomialLayer(
            scope=Scope([1], [0]),
            n_nodes=3,
            n=n_value,
            cond_f=lambda data: {"p": p_value},
        )

        for n_layer_node, p_layer_node in zip(
            l.n, l.retrieve_params(torch.tensor([[1]]), DispatchContext())
        ):
            self.assertTrue(torch.all(n_layer_node == n_value))
            self.assertTrue(torch.all(p_layer_node == p_value))

        # ----- list parameter values -----
        n_values = [3, 2, 7]
        p_values = [0.17, 0.8, 0.53]
        l = CondBinomialLayer(
            scope=[Scope([0], [3]), Scope([1], [3]), Scope([2], [3])],
            n=n_values,
            cond_f=lambda data: {"p": p_values},
        )

        for n_layer_node, p_layer_node, n_value, p_value in zip(
            l.n,
            l.retrieve_params(torch.tensor([[1]]), DispatchContext()),
            n_values,
            p_values,
        ):
            self.assertTrue(torch.allclose(n_layer_node, torch.tensor(n_value)))
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

        l.set_cond_f(lambda data: {"p": np.expand_dims(np.array(p_values), 0)})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        l.set_cond_f(lambda data: {"p": np.expand_dims(np.array(p_values), 1)})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

    def test_accept(self):

        # discrete meta type (should reject)
        self.assertFalse(
            CondBinomialLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # feature type instance
        self.assertTrue(
            CondBinomialLayer.accepts(
                [
                    FeatureContext(
                        Scope([0], [2]), [FeatureTypes.Binomial(n=3)]
                    ),
                    FeatureContext(
                        Scope([1], [2]), [FeatureTypes.Binomial(n=3)]
                    ),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondBinomialLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                    FeatureContext(
                        Scope([1], [2]), [FeatureTypes.Binomial(n=3)]
                    ),
                ]
            )
        )

        # non-conditional scope
        self.assertFalse(
            CondBinomialLayer.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Binomial(n=3)])]
            )
        )

        # multivariate signature
        self.assertFalse(
            CondBinomialLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [
                            FeatureTypes.Binomial(n=3),
                            FeatureTypes.Binomial(n=3),
                        ],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        binomial = CondBinomialLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Binomial(n=3)]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Binomial(n=5)]),
            ]
        )
        self.assertTrue(torch.all(binomial.n == torch.tensor([3, 5])))
        self.assertTrue(
            binomial.scopes_out == [Scope([0], [2]), Scope([1], [3])]
        )

        # ----- invalid arguments -----

        # discrete meta type
        self.assertRaises(
            ValueError,
            CondBinomialLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondBinomialLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondBinomialLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Binomial(3)])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondBinomialLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Binomial(3), FeatureTypes.Binomial(5)],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondBinomialLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondBinomialLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(
                        Scope([0], [2]), [FeatureTypes.Binomial(n=3)]
                    ),
                    FeatureContext(
                        Scope([1], [3]), [FeatureTypes.Binomial(n=5)]
                    ),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        binomial = AutoLeaf(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Binomial(n=3)]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Binomial(n=5)]),
            ]
        )
        self.assertTrue(isinstance(binomial, CondBinomialLayer))
        self.assertTrue(torch.all(binomial.n == torch.tensor([3, 5])))
        self.assertTrue(
            binomial.scopes_out == [Scope([0], [2]), Scope([1], [3])]
        )

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = CondBinomialLayer(scope=Scope([1], [0]), n_nodes=2, n=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

        # ---------- different scopes -----------

        l = CondBinomialLayer(
            scope=[Scope([1], [2]), Scope([0], [2])], n=[3, 2]
        )

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, CondBinomial))
        self.assertEqual(l_marg.scope, Scope([0], [2]))
        self.assertTrue(torch.allclose(l_marg.n, torch.tensor(2)))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, CondBinomialLayer))
        self.assertEqual(len(l_marg.scopes_out), 1)
        self.assertTrue(torch.allclose(l_marg.n, torch.tensor(2)))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])
        self.assertTrue(torch.allclose(l.n, l_marg.n))

    def test_layer_dist(self):

        n_values = [3, 2, 7]
        p_values = torch.tensor([0.73, 0.29, 0.5])
        l = CondBinomialLayer(
            scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
            n=n_values,
            n_nodes=3,
        )

        # ----- full dist -----
        dist = l.dist(p_values)

        for n_value, p_value, n_dist, p_dist in zip(
            n_values, p_values, dist.total_count, dist.probs
        ):
            self.assertTrue(
                torch.allclose(torch.tensor(n_value).double(), n_dist)
            )
            self.assertTrue(torch.allclose(p_value, p_dist))

        # ----- partial dist -----
        dist = l.dist(p_values, [1, 2])

        for n_value, p_value, n_dist, p_dist in zip(
            n_values[1:], p_values[1:], dist.total_count, dist.probs
        ):
            self.assertTrue(
                torch.allclose(torch.tensor(n_value).double(), n_dist)
            )
            self.assertTrue(torch.allclose(p_value, p_dist))

        dist = l.dist(p_values, [1, 0])

        for n_value, p_value, n_dist, p_dist in zip(
            reversed(n_values[:-1]),
            reversed(p_values[:-1]),
            dist.total_count,
            dist.probs,
        ):
            self.assertTrue(
                torch.allclose(torch.tensor(n_value).double(), n_dist)
            )
            self.assertTrue(torch.allclose(p_value, p_dist))

    def test_layer_backend_conversion_1(self):

        torch_layer = CondBinomialLayer(
            scope=[Scope([0], [3]), Scope([1], [3]), Scope([0], [3])],
            n=[2, 5, 2],
        )
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.n, torch_layer.n.numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

    def test_layer_backend_conversion_2(self):

        base_layer = BaseCondBinomialLayer(
            scope=[Scope([0], [3]), Scope([1], [3]), Scope([0], [3])],
            n=[2, 5, 2],
        )
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.n, torch_layer.n.numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
