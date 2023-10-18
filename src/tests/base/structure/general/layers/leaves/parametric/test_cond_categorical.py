import unittest

import numpy as np

from spflow.base.structure import AutoLeaf
from spflow.base.structure.spn import CondCategorical, CondCategoricalLayer, marginalize
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext


class TestCondCategorical(unittest.TestCase):
    def test_initialization(self):

        layer = CondCategoricalLayer(scope=Scope([0], [1]), n_nodes=3)
        self.assertEqual(len(layer.nodes), 3)
        self.assertTrue(np.all(layer.scopes_out == [Scope([0], [1])]*3))

        scopes = [Scope([0], [1]), Scope([2], [1])]
        layer = CondCategoricalLayer(scope=scopes, n_nodes=2)
        for node, node_scope in zip(layer.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

        #check number of conditional functions
        CondCategoricalLayer(Scope([0], [1]), n_nodes=2, cond_f=[lambda data: {"k": 2, "p": [0.5, 0.5]}, lambda data: {"k": 2, "p": [0.3, 0.7]}])
        self.assertRaises(ValueError, CondCategoricalLayer, Scope([0], [1]), n_nodes=2, cond_f=[lambda data: {"k": 2, "p": [0.5, 0.5]}])


        # invalid stuff
        self.assertRaises(ValueError, CondCategoricalLayer, Scope([0], [1]), n_nodes=0)
        self.assertRaises(ValueError, CondCategoricalLayer, Scope([]))
        self.assertRaises(ValueError, CondCategoricalLayer, [])


    def test_retrieve_params(self):

        flat_k = 2
        flat_p = [0.5, 0.5]
        layer = CondCategoricalLayer(Scope([0], [1]), n_nodes=3, cond_f=lambda data: {"k": flat_k, "p": flat_p})

        for k, p in zip(*layer.retrieve_params(np.array([[1.0]]), DispatchContext())):
            self.assertEqual(k, flat_k)
            self.assertTrue(np.all(p == np.array(flat_p)))


        list_k = [2, 2, 2]
        list_p = [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]]

        layer.set_cond_f(lambda data: {"k": list_k, "p": list_p})
        layer_k, layer_p = layer.retrieve_params(np.array([[1.0]]), DispatchContext())
        for node_k, k in zip(layer_k, list_k):
            self.assertEqual(node_k, k)
        for node_p, p in zip(layer_p, list_p):
            self.assertTrue(np.all(node_p == np.array(p)))


           # wrong number of values
        layer.set_cond_f(lambda data: {"k": list_k, "p": list_p[:-1]})
        self.assertRaises(ValueError, layer.retrieve_params, np.array([[1]]), DispatchContext())

        # wrong number of dimensions (nested list)
        layer.set_cond_f(lambda data: {"k": list_k, "p": [list_p for _ in range(3)]})
        self.assertRaises(ValueError, layer.retrieve_params, np.array([[1]]), DispatchContext())

        # ----- numpy parameter values -----
        layer.set_cond_f(lambda data: {"k": np.array(list_k), "p": np.array(list_p)})
        layer_k, layer_p = layer.retrieve_params(np.array([[1.0]]), DispatchContext())
        for node_k, k in zip(layer_k, list_k):
            self.assertEqual(node_k, k)
        for node_p, p in zip(layer_p, list_p):
            self.assertTrue(np.all(node_p == p))
        #for p_node, p_actual in zip(layer.retrieve_params(np.array([[1.0]]), DispatchContext())[1], list_p):
        #    self.assertTrue(p_node == p_actual)


    def test_accept(self):
        # discrete meta type
        self.assertTrue(CondCategoricalLayer.accepts(
            [
                FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete]), 
                FeatureContext(Scope([2], [3]), [FeatureTypes.Discrete])
            ]
        ))
        
        # Categorical feature type class
        self.assertTrue(CondCategoricalLayer.accepts(
            [
                FeatureContext(Scope([0], [1]), [FeatureTypes.Categorical]), 
                FeatureContext(Scope([2], [3]), [FeatureTypes.Categorical])
            ]
        ))
        
        # Categorical feature type instance
        self.assertTrue(CondCategoricalLayer.accepts(
            [
                FeatureContext(Scope([0], [1]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])]), 
                FeatureContext(Scope([2], [3]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])])
            ]
        ))

        # invalid feature types
        self.assertFalse(CondCategoricalLayer.accepts(
            [
                FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous]), 
                FeatureContext(Scope([2], [3]), [FeatureTypes.Continuous])
            ]
        ))

        # non-conditional scope (invalid)
        self.assertFalse(CondCategoricalLayer.accepts(
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
        ))

        # multivariate scope (invalid)
        self.assertFalse(CondCategoricalLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]), 
                    [FeatureTypes.Discrete, FeatureTypes.Discrete]
                )
            ]
        ))


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
                FeatureContext(Scope([0], [2]), [FeatureTypes.Categorical(k = 2, p=[0.5, 0.5])]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Categorical(k = 2, p=[0.3, 0.7])]),
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
        layer = AutoLeaf(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Categorical(k = 2, p=[0.5, 0.5])]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Categorical(k = 2, p=[0.3, 0.7])]),
            ]
        )
        self.assertTrue(layer.scopes_out == [Scope([0], [2]), Scope([1], [3])])


    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        layer = CondCategoricalLayer(scope=Scope([1], [0]), n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(layer, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        layer_marg = marginalize(layer, [2])

        self.assertTrue(layer_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

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
        self.assertEqual(len(layer_marg.nodes), 1)

        # ----- marginalize over non-scope rvs -----
        layer_marg = marginalize(layer, [2])

        self.assertTrue(layer_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])


if __name__ == "__main__":
    unittest.main()
