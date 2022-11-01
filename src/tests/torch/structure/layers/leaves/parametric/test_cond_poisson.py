from spflow.torch.structure.layers.leaves.parametric.cond_poisson import (
    CondPoissonLayer,
    marginalize,
    toTorch,
    toBase,
)
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.torch.structure.nodes.leaves.parametric.cond_poisson import (
    CondPoisson,
)
from spflow.base.structure.layers.leaves.parametric.cond_poisson import (
    CondPoissonLayer as BaseCondPoissonLayer,
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
        l = CondPoissonLayer(scope=Scope([1], [0]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(
                l.scopes_out
                == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])]
            )
        )

        # ---- different scopes -----
        l = CondPoissonLayer(scope=Scope([1], [0]), n_nodes=3)
        for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(
            ValueError, CondPoissonLayer, Scope([0], [1]), n_nodes=0
        )

        # ----- invalid scope -----
        self.assertRaises(ValueError, CondPoissonLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, CondPoissonLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
        l = CondPoissonLayer(
            scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3
        )

        for layer_scope, node_scope in zip(l.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)

        # -----number of cond_f functions -----
        CondPoissonLayer(
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"l": 1}, lambda data: {"l": 1}],
        )
        self.assertRaises(
            ValueError,
            CondPoissonLayer,
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"l": 1}],
        )

    def test_retrieve_params(self):

        # ----- float/int parameter values -----
        l_value = 0.73
        l = CondPoissonLayer(
            scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"l": l_value}
        )

        for l_layer_node in l.retrieve_params(
            torch.tensor([[1]]), DispatchContext()
        ):
            self.assertTrue(torch.allclose(l_layer_node, torch.tensor(l_value)))

        # ----- list parameter values -----
        l_values = [0.17, 0.8, 0.53]
        l = CondPoissonLayer(
            scope=Scope([1], [0]),
            n_nodes=3,
            cond_f=lambda data: {"l": l_values},
        )

        for l_layer_node, l_value in zip(
            l.retrieve_params(torch.tensor([[1]]), DispatchContext()), l_values
        ):
            self.assertTrue(torch.allclose(l_layer_node, torch.tensor(l_value)))

        # wrong number of values
        l.set_cond_f(lambda data: {"l": l_values[:-1]})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        # wrong number of dimensions (nested list)
        l.set_cond_f(lambda data: {"l": [l_values for _ in range(3)]})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        # ----- numpy parameter values -----
        l.set_cond_f(lambda data: {"l": np.array(l_values)})
        for l_node, l_actual in zip(
            l.retrieve_params(torch.tensor([[1.0]]), DispatchContext()),
            l_values,
        ):
            self.assertTrue(l_node == l_actual)

        # wrong number of values
        l.set_cond_f(lambda data: {"l": np.array(l_values[:-1])})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        # wrong number of dimensions (nested list)
        l.set_cond_f(lambda data: {"l": np.array([l_values for _ in range(3)])})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            CondPoissonLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # Poisson feature type class
        self.assertTrue(
            CondPoissonLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Poisson]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # Poisson feature type instance
        self.assertTrue(
            CondPoissonLayer.accepts(
                [
                    FeatureContext(
                        Scope([0], [2]), [FeatureTypes.Poisson(1.0)]
                    ),
                    FeatureContext(
                        Scope([1], [2]), [FeatureTypes.Poisson(1.0)]
                    ),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondPoissonLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                    FeatureContext(
                        Scope([1], [2]), [FeatureTypes.Poisson(1.0)]
                    ),
                ]
            )
        )

        # non-conditional scope
        self.assertFalse(
            CondPoissonLayer.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # multivariate signature
        self.assertFalse(
            CondPoissonLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        poisson = CondPoissonLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Discrete]),
            ]
        )
        self.assertTrue(
            poisson.scopes_out == [Scope([0], [2]), Scope([1], [2])]
        )

        poisson = CondPoissonLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Poisson]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Poisson]),
            ]
        )
        self.assertTrue(
            poisson.scopes_out == [Scope([0], [2]), Scope([1], [2])]
        )

        poisson = CondPoissonLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Poisson(l=1.5)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Poisson(l=2.0)]),
            ]
        )
        self.assertTrue(
            poisson.scopes_out == [Scope([0], [2]), Scope([1], [2])]
        )

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondPoissonLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondPoissonLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondPoissonLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondPoissonLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondPoissonLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Poisson]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Poisson]),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        poisson = AutoLeaf(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Poisson(l=1.5)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Poisson(l=2.0)]),
            ]
        )
        self.assertTrue(isinstance(poisson, CondPoissonLayer))
        self.assertTrue(
            poisson.scopes_out == [Scope([0], [2]), Scope([1], [2])]
        )

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = CondPoissonLayer(scope=Scope([1], [0]), n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

        # ---------- different scopes -----------

        l = CondPoissonLayer(scope=[Scope([1], [2]), Scope([0], [2])])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, CondPoisson))
        self.assertEqual(l_marg.scope, Scope([0], [2]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, CondPoissonLayer))
        self.assertEqual(len(l_marg.scopes_out), 1)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])

    def test_layer_dist(self):

        l_values = torch.tensor([0.73, 0.29, 0.5])
        l = CondPoissonLayer(scope=Scope([1], [0]), n_nodes=3)

        # ----- full dist -----
        dist = l.dist(l_values)

        for l_value, l_dist in zip(l_values, dist.rate):
            self.assertTrue(torch.allclose(l_value, l_dist))

        # ----- partial dist -----
        dist = l.dist(l_values, [1, 2])

        for l_value, l_dist in zip(l_values[1:], dist.rate):
            self.assertTrue(torch.allclose(l_value, l_dist))

        dist = l.dist(l_values, [1, 0])

        for l_value, l_dist in zip(reversed(l_values[:-1]), dist.rate):
            self.assertTrue(torch.allclose(l_value, l_dist))

    def test_layer_backend_conversion_1(self):

        torch_layer = CondPoissonLayer(
            scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])]
        )
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

    def test_layer_backend_conversion_2(self):

        base_layer = BaseCondPoissonLayer(
            scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])]
        )
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
