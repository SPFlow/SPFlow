import unittest

import numpy as np

from spflow.base.structure import AutoLeaf
from spflow.base.structure.spn import Categorical, CategoricalLayer, marginalize
from spflow.meta.data import FeatureContext, FeatureTypes, Scope


class TestCategorical(unittest.TestCase):
    def test_initialization(self):

        # default init
        layer = CategoricalLayer(scope=Scope([0]), n_nodes=3)
        self.assertEqual(len(layer.nodes), 3)
        self.assertTrue(np.all(layer.scopes_out == [Scope([0])]*3))
        probs = layer.p
        for node, node_probs in zip(layer.nodes, probs):
            self.assertTrue(np.all(node.p == node_probs))
            self.assertTrue(node.k == 2)

        # set shared p for all nodes
        flat_k = 2
        flat_probs = [0.3, 0.7]
        layer = CategoricalLayer(scope=Scope([0]), k=flat_k, p=flat_probs, n_nodes=3)
        for node in layer.nodes:
            self.assertTrue(np.all(node.p == flat_probs))

        # set p per node
        # TODO: different k per node would lead to in inhomogeneous np array of the layer probabilities
        # TODO: either FORCE the nodes to have the same k (which makes sense having same scope, but not if different)
        # TODO: OR either declare the dtype of the np.ndarray as 'object' or use lists instead of np arrays
        list_k = [1, 2, 4]
        list_probs = [[1.0], [0.3, 0.7], [0.1, 0.2, 0.3, 0.4]]
        # layer = CategoricalLayer(scope=Scope([0]), k=list_k, p=list_probs, n_nodes=3)

        # for node, node_k, node_probs in zip(layer.nodes, list_k, list_probs):
        #     self.assertTrue(node.k == node_k)
        #     self.assertTrue(np.all(node.p == node_probs))
        self.assertRaises(ValueError, CategoricalLayer, Scope([0]), list_k, list_probs, 3)


        # wrong values
        self.assertRaises(ValueError, CategoricalLayer, Scope([0]), list_k[0:2], list_probs, 3)
        self.assertRaises(ValueError, CategoricalLayer, Scope([0]), list_k, list_probs[0:2], 3)
        self.assertRaises(ValueError, CategoricalLayer, Scope([0]), list_k, list_probs, 2)
        self.assertRaises(ValueError, CategoricalLayer, Scope([0]), [2, 3], [[1.0], [1.0]], 2)
        self.assertRaises(ValueError, CategoricalLayer, Scope([0]), [1, 2], [[0.5], [1.0, 1.0]], 2)
        self.assertRaises(ValueError, CategoricalLayer, Scope([0]), n_nodes=0)
        self.assertRaises(ValueError, CategoricalLayer, Scope([]))

        scopes = [Scope([0]), Scope([1]), Scope([2])]
        layer = CategoricalLayer(scope=scopes, n_nodes=3)
        for node, node_scope in zip(layer.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

    
    def test_accept(self):
        # discrete meta type
        self.assertTrue(CategoricalLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete]), 
                                                  FeatureContext(Scope([1]), [FeatureTypes.Discrete])]))
        
        # feature type class
        self.assertTrue(CategoricalLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Categorical]), 
                                                  FeatureContext(Scope([1]), [FeatureTypes.Discrete])]))
        
        # feature type instance
        self.assertTrue(CategoricalLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Categorical(k=1, p=[1.0])]), 
                                                  FeatureContext(Scope([1]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])])]))


        # invalid feature type
        self.assertFalse(CategoricalLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous]), 
                                                   FeatureContext(Scope([1]), [FeatureTypes.Continuous])]))
        
        # conditional scope (invalid)
        self.assertFalse(CategoricalLayer.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

        # multivariate signature (invalid)
        self.assertFalse(CategoricalLayer.accepts([FeatureContext(Scope([0, 1]), [FeatureTypes.Discrete, FeatureTypes.Discrete])]))


    def test_initialization_from_signatures(self):

        layer = CategoricalLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]), 
                FeatureContext(Scope([1]), [FeatureTypes.Discrete])
            ]
        )
        self.assertTrue(np.all(layer.k == np.array([2, 2])))
        self.assertTrue(np.all(layer.p == np.array([[0.5, 0.5], [0.5, 0.5]])))
        self.assertTrue(layer.scopes_out == [Scope([0]), Scope([1])])

        layer = CategoricalLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Categorical]), 
                FeatureContext(Scope([1]), [FeatureTypes.Categorical])
            ]
        )
        self.assertTrue(np.all(layer.k == np.array([2, 2])))
        self.assertTrue(np.all(layer.p == np.array([[0.5, 0.5], [0.5, 0.5]])))
        self.assertTrue(layer.scopes_out == [Scope([0]), Scope([1])])

        
        layer = CategoricalLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Categorical(k=2, p=[0.3, 0.7])]), 
                FeatureContext(Scope([1]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])])
            ]
        )
        self.assertTrue(np.all(layer.k == np.array([2, 2])))
        self.assertTrue(np.all(layer.p == np.array([[0.3, 0.7], [0.5, 0.5]])))
        self.assertTrue(layer.scopes_out == [Scope([0]), Scope([1])])


        
        # invalid feature type
        self.assertRaises(
            ValueError,
            CategoricalLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            CategoricalLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CategoricalLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ],
        )


        
    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CategoricalLayer))

        feature_ctx_1 = FeatureContext(Scope([0]), [FeatureTypes.Categorical(k=2, p=[0.3, 0.7])])
        feature_ctx_2 = FeatureContext(Scope([1]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])])

        # make sure leaf is correctly inferred
        self.assertEqual(CategoricalLayer, AutoLeaf.infer([feature_ctx_1, feature_ctx_2]))

        # make sure AutoLeaf can return correctly instantiated object
        categorical = AutoLeaf([feature_ctx_1, feature_ctx_2])
        self.assertTrue(np.all(categorical.k == np.array([2, 2])))
        self.assertTrue(np.all(categorical.p == np.array([[0.3, 0.7], [0.5, 0.5]])))
        self.assertTrue(categorical.scopes_out == [Scope([0]), Scope([1])])


    def test_layer_structural_marginalization(self):

        layer = CategoricalLayer(scope=Scope([0]), k=2, p=[[0.3, 0.7], [0.5, 0.5]], n_nodes=2)

        # marginalize scope
        self.assertTrue(marginalize(layer, [0]) == None)
        
        # marginalize out of scope (= no effect)
        layer_marg = marginalize(layer, [1])

        self.assertTrue(layer_marg.scopes_out == [Scope([0]), Scope([0])])
        self.assertTrue(np.all(layer.p == layer_marg.p))

        # different scopes
        layer = CategoricalLayer(scope=[Scope([0]), Scope([1])], k=2, p=[0.5, 0.5], n_nodes=2)
        self.assertTrue(marginalize(layer, [0, 1]) == None)

        layer_marg = marginalize(layer, [1], prune=True)
        self.assertTrue(isinstance(layer_marg, Categorical))
        self.assertEqual(layer_marg.scope, Scope([0]))
        self.assertTrue(np.all(layer_marg.p == np.array([0.5, 0.5])))

        layer_marg = marginalize(layer, [1], prune=False)
        self.assertTrue(isinstance(layer_marg, CategoricalLayer))
        self.assertEqual(len(layer_marg.nodes), 1)
        self.assertTrue(np.all(layer_marg.p == np.array([0.5, 0.5])))

        layer_marg = marginalize(layer, [2])
        self.assertTrue(layer_marg.scopes_out == [Scope([0]), Scope([1])])
        self.assertTrue(np.all(layer.p == layer_marg.p))


    def test_get_params(self):

        layer = CategoricalLayer(scope=Scope([0]), k=2, p=[[0.5, 0.5], [0.5, 0.5]], n_nodes=2)

        k, p, *_ = layer.get_params()

        self.assertTrue(len(_) == 0)
        self.assertTrue(np.allclose(p, np.array([[0.5, 0.5], [0.5, 0.5]])))


if __name__ == "__main__":
    unittest.main()


        
        


        