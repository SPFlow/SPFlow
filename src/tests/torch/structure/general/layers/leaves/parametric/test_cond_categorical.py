import unittest

import numpy as np
import torch

from spflow.base.structure.spn import CondCategoricalLayer as BaseCondCategoricalLayer
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure import AutoLeaf, marginalize, toTorch, toBase
from spflow.torch.structure.spn import CondCategorical, CondCategoricalLayer


class TestCondCategorical(unittest.TestCase):

    def test_initialization(self):

        
        # ----- check attributes after correct initialization -----
        layer = CondCategoricalLayer(scope=Scope([0], [1]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(layer.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(layer.scopes_out == [Scope([0], [1]), Scope([0], [1]), Scope([0], [1])]))

        # ---- different scopes -----
        layer = CondCategoricalLayer(scope=Scope([0], [1]), n_nodes=3)
        for layer_scope, node_scope in zip(layer.scopes_out, layer.scopes_out):
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, CondCategoricalLayer, Scope([0], [1]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, CondCategoricalLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, CondCategoricalLayer, Scope([0]), n_nodes=3)
        self.assertRaises(ValueError, CondCategoricalLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
        layer = CondCategoricalLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3)

        for layer_scope, node_scope in zip(layer.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)

        # -----number of cond_f functions -----
        CondCategoricalLayer(
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"k": 2, "p": [0.5, 0.5]}, lambda data: {"k": 2, "p": [0.3, 0.7]}],
        )
        self.assertRaises(
            ValueError,
            CondCategoricalLayer,
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"k": 2, "p": [0.5, 0.5]}],
        )


    
    def test_retrieve_param(self):

        # ----- float/int parameter values -----
        k = 2
        p = [0.5, 0.5]
        layer = CondCategoricalLayer(scope=Scope([0], [1]), n_nodes=3, cond_f=lambda data: {"k": k, "p": p})

        for p in layer.retrieve_params(torch.tensor([[1.0]]), DispatchContext()):
            self.assertTrue(torch.all(p == p))

        # ----- list parameter values -----
        k = [2, 2, 2]
        p = [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]]
        layer = CondCategoricalLayer(
            scope=Scope([0], [1]),
            n_nodes=3,
            cond_f=lambda data: {"k": k, "p": p},
        )

        layer_k, layer_p = layer.retrieve_params(torch.tensor([[1.0]]), DispatchContext())
        self.assertTrue(torch.all(layer_k == torch.tensor(k)))
        self.assertTrue(torch.allclose(layer_p, torch.tensor(p)))

        # wrong number of values
        layer.set_cond_f(lambda data: {"k": k, "p": p[:-1]})
        self.assertRaises(
            ValueError,
            layer.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        # wrong number of dimensions (nested list)
        layer.set_cond_f(lambda data: {"k": k, "p": [p for _ in range(3)]})
        self.assertRaises(
            ValueError,
            layer.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )


    
    def test_accept(self):

        # discrete meta type
        self.assertTrue(
            CondCategoricalLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # Bernoulli feature type class
        self.assertTrue(
            CondCategoricalLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Categorical]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # Bernoulli feature type instance
        self.assertTrue(
            CondCategoricalLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Categorical(k=2, p=[0.3, 0.7])]),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondCategoricalLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # non-conditional scope
        self.assertFalse(CondCategoricalLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

        # multivariate signature
        self.assertFalse(
            CondCategoricalLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )

    
    
    def test_initialization_from_signatures(self):

        layer = CondCategoricalLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Discrete]),
            ]
        )
        self.assertTrue(layer.scopes_out == [Scope([0], [2]), Scope([1], [3])])

        layer = CondCategoricalLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Categorical]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Categorical]),
            ]
        )
        self.assertTrue(layer.scopes_out == [Scope([0], [2]), Scope([1], [3])])

        layer = CondCategoricalLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Categorical(k=2, p=[0.3, 0.7])]),
            ]
        )
        self.assertTrue(layer.scopes_out == [Scope([0], [2]), Scope([1], [3])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondCategoricalLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondCategoricalLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondCategoricalLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )


    
    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondCategoricalLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondCategoricalLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Categorical()]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Categorical()]),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        categorical = AutoLeaf(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Categorical(k=2, p=[0.3, 0.7])]),
            ]
        )
        self.assertTrue(categorical.scopes_out == [Scope([0], [2]), Scope([1], [3])])


    
    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        layer = CondCategoricalLayer(scope=Scope([0], [1]), n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(layer, [0]) == None)

        # ----- marginalize over non-scope rvs -----
        layer_marg = marginalize(layer, [2])

        self.assertTrue(layer_marg.scopes_out == [Scope([0], [1]), Scope([0], [1])])

        # ---------- different scopes -----------

        layer = CondCategoricalLayer(scope=[Scope([1], [2]), Scope([0], [2])])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(layer, [0, 1]) == None)

        # ----- partially marginalize -----
        layer_marg = marginalize(layer, [1], prune=True)
        self.assertTrue(isinstance(layer_marg, CondCategorical))
        self.assertEqual(layer_marg.scope, Scope([0], [2]))

        layer_marg = marginalize(layer, [1], prune=False)
        self.assertTrue(isinstance(layer_marg, CondCategoricalLayer))
        self.assertEqual(len(layer_marg.scopes_out), 1)

        # ----- marginalize over non-scope rvs -----
        layer_marg = marginalize(layer, [2])

        self.assertTrue(layer_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])

    
    def test_layer_dist(self):

        k = [2, 2, 2]
        probs = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]])
        layer = CondCategoricalLayer(scope=Scope([0], [1]), cond_f=lambda data: {"k": k}, n_nodes=3)

        # ----- full dist -----
        dist = layer.dist(probs)

        for p_value, p_dist in zip(probs, dist.probs):
            self.assertTrue(torch.allclose(p_value, p_dist))

        # ----- partial dist -----
        dist = layer.dist(probs, [1, 2])

        for p_value, p_dist in zip(probs[1:], dist.probs):
            self.assertTrue(torch.allclose(p_value, p_dist))

        dist = layer.dist(probs, [1, 0])

        for p_value, p_dist in zip(reversed(probs[:-1]), dist.probs):
            self.assertTrue(torch.allclose(p_value, p_dist))


    
    def test_layer_backend_conversion_1(self):

        torch_layer = CondCategoricalLayer(scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])])
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

        base_layer = BaseCondCategoricalLayer(scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])])
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()



