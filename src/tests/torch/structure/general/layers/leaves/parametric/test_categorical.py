import unittest

import numpy as np
import torch

from spflow.base.structure.spn import CategoricalLayer as BaseCategoricalLayer
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import AutoLeaf, marginalize, toBase, toTorch
from spflow.torch.structure.spn import Categorical, CategoricalLayer


class TestCategorical(unittest.TestCase):

    def test_initialization(self):
        k = 2
        probs = [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]]

        layer = CategoricalLayer(scope=Scope([0]), n_nodes=3, k=k, p=probs)
        self.assertEqual(len(layer.scopes_out), 3)
        self.assertTrue(np.all(layer.scopes_out == [Scope([0]), Scope([0]), Scope([0])]))

        for node_p, p in zip(layer.p, probs):
            self.assertTrue(torch.allclose(node_p, torch.tensor(p)))

        k = 2
        p = [0.3, 0.7]
        layer = CategoricalLayer(scope=Scope([0]), n_nodes=3, k=k, p=p)
        for node_k in layer.k:
            self.assertTrue(torch.all(node_k == torch.tensor(k)))
        for node_p in layer.p:
            self.assertTrue(torch.all(node_p == torch.tensor(p)))


        # wrong number of values
        self.assertRaises(ValueError, CategoricalLayer, Scope([0]), 3, probs[:-1], n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            CategoricalLayer,
            Scope([0]),
            3,
            [probs for _ in range(3)],
            n_nodes=3,
        )

        # ---- different scopes -----
        layer = CategoricalLayer(scope=Scope([1]), n_nodes=3)
        for layer_scope, node_scope in zip(layer.scopes_out, layer.scopes_out):
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, CategoricalLayer, Scope([0]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, CategoricalLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, CategoricalLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        layer = CategoricalLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)

        for layer_scope, node_scope in zip(layer.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)


    def test_accept(self):

        # discrete meta type
        self.assertTrue(
            CategoricalLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # feature type class
        self.assertTrue(
            CategoricalLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Categorical]),
                    FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
                ]
            )
        )

        # feature type instance
        self.assertTrue(
            CategoricalLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])]),
                    FeatureContext(Scope([1]), [FeatureTypes.Categorical(k=2, p=[0.3, 0.7])]),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CategoricalLayer.accepts(
                [
                    FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # conditional scope
        self.assertFalse(CategoricalLayer.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

        # multivariate signature
        self.assertFalse(
            CategoricalLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )


    def test_initialization_from_signatures(self):

        layer = CategoricalLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
            ]
        )
        self.assertTrue(torch.allclose(layer.k, torch.tensor([2, 2])))
        self.assertTrue(torch.allclose(layer.p, torch.tensor([[0.5, 0.5], [0.5, 0.5]])))
        self.assertTrue(layer.scopes_out == [Scope([0]), Scope([1])])

        layer = CategoricalLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Categorical]),
                FeatureContext(Scope([1]), [FeatureTypes.Categorical]),
            ]
        )
        self.assertTrue(torch.allclose(layer.k, torch.tensor([2, 2])))
        self.assertTrue(torch.allclose(layer.p, torch.tensor([[0.5, 0.5], [0.5, 0.5]])))
        self.assertTrue(layer.scopes_out == [Scope([0]), Scope([1])])

        layer = CategoricalLayer.from_signatures(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])]),
                FeatureContext(Scope([1]), [FeatureTypes.Categorical(k=2, p=[0.3, 0.7])]),
            ]
        )
        self.assertTrue(torch.allclose(layer.k, torch.tensor([2, 2])))
        self.assertTrue(torch.allclose(layer.p, torch.tensor([[0.5, 0.5], [0.3, 0.7]])))
        self.assertTrue(layer.scopes_out == [Scope([0]), Scope([1])])

        # ----- invalid arguments -----

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

        feature_ctx_1 = FeatureContext(Scope([0]), [FeatureTypes.Categorical(k=2, p=[0.5, 0.5])])
        feature_ctx_2 = FeatureContext(Scope([1]), [FeatureTypes.Categorical(k=2, p=[0.3, 0.7])])

        # make sure leaf is correctly inferred
        self.assertEqual(CategoricalLayer, AutoLeaf.infer([feature_ctx_1, feature_ctx_2]))

        # make sure AutoLeaf can return correctly instantiated object
        layer = AutoLeaf([feature_ctx_1, feature_ctx_2])
        self.assertTrue(torch.allclose(layer.k, torch.tensor([2, 2])))
        self.assertTrue(torch.allclose(layer.p, torch.tensor([[0.5, 0.5], [0.3, 0.7]])))
        self.assertTrue(layer.scopes_out == [Scope([0]), Scope([1])])


    
    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        layer = CategoricalLayer(scope=Scope([0]), k=[2, 2], p=[[0.5, 0.5], [0.3, 0.7]], n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(layer, [0]) == None)

        # ----- marginalize over non-scope rvs -----
        layer_marg = marginalize(layer, [1])

        self.assertTrue(layer_marg.scopes_out == [Scope([0]), Scope([0])])
        self.assertTrue(torch.allclose(layer.p, layer_marg.p))

        # ---------- different scopes -----------

        layer = CategoricalLayer(scope=[Scope([0]), Scope([1])], k=[2, 2], p=[[0.5, 0.5], [0.3, 0.7]])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(layer, [0, 1]) == None)

        # ----- partially marginalize -----
        layer_marg = marginalize(layer, [1], prune=True)
        self.assertTrue(isinstance(layer_marg, Categorical))
        self.assertEqual(layer_marg.scope, Scope([0]))
        self.assertEqual(layer_marg.k, torch.tensor(2))
        self.assertTrue(torch.allclose(layer_marg.p, torch.tensor([0.5, 0.5])))

        layer_marg = marginalize(layer, [1], prune=False)
        self.assertTrue(isinstance(layer_marg, CategoricalLayer))
        self.assertEqual(len(layer_marg.scopes_out), 1)
        self.assertEqual(layer_marg.k, torch.tensor(2))
        self.assertTrue(torch.allclose(layer_marg.p, torch.tensor([0.5, 0.5])))

        # ----- marginalize over non-scope rvs -----
        layer_marg = marginalize(layer, [2])

        self.assertTrue(layer_marg.scopes_out == [Scope([0]), Scope([1])])
        self.assertTrue(torch.allclose(layer.p, layer_marg.p))


    
    def test_layer_dist(self):

        k = [2, 2, 2]
        probs = [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]]
        layer = CategoricalLayer(scope=Scope([1]), k=k, p=probs, n_nodes=3)

        # ----- full dist -----
        dist = layer.dist()

        for p, p_dist in zip(probs, dist.probs):
            self.assertTrue(torch.allclose(torch.tensor(p), p_dist))

        # ----- partial dist -----
        dist = layer.dist([1, 2])

        for p, p_dist in zip(probs[1:], dist.probs):
            self.assertTrue(torch.allclose(torch.tensor(p), p_dist))

        dist = layer.dist([1, 0])

        for p, p_dist in zip(reversed(probs[:-1]), dist.probs):
            self.assertTrue(torch.allclose(torch.tensor(p), p_dist))

    
    
    def test_layer_backend_conversion_1(self):

        torch_layer = CategoricalLayer(scope=[Scope([0]), Scope([1]), Scope([0])], k=2, p=[[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]])
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.p, torch_layer.p.detach().numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

    def test_layer_backend_conversion_2(self):

        base_layer = BaseCategoricalLayer(scope=[Scope([0]), Scope([1]), Scope([0])], k=2, p=[[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]])
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.p, torch_layer.p.detach().numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()





