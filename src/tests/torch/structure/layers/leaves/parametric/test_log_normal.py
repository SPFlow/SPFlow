from spflow.torch.structure.layers.leaves.parametric.log_normal import (
    LogNormalLayer,
    marginalize,
    toTorch,
    toBase,
)
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.torch.structure.nodes.leaves.parametric.log_normal import LogNormal
from spflow.base.structure.layers.leaves.parametric.log_normal import (
    LogNormalLayer as BaseLogNormalLayer,
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
        mean_values = [0.5, -2.3, 1.0]
        std_values = [1.3, 1.0, 0.2]
        l = LogNormalLayer(
            scope=Scope([1]), n_nodes=3, mean=mean_values, std=std_values
        )
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])])
        )
        # make sure parameter properties works correctly
        for mean_layer_node, std_layer_node, mean_value, std_value in zip(
            l.mean, l.std, mean_values, std_values
        ):
            self.assertTrue(
                torch.allclose(mean_layer_node, torch.tensor(mean_value))
            )
            self.assertTrue(
                torch.allclose(std_layer_node, torch.tensor(std_value))
            )

        # ----- float/int parameter values -----
        mean_value = 0.73
        std_value = 1.9
        l = LogNormalLayer(
            scope=Scope([1]), n_nodes=3, mean=mean_value, std=std_value
        )

        for mean_layer_node, std_layer_node in zip(l.mean, l.std):
            self.assertTrue(
                torch.allclose(mean_layer_node, torch.tensor(mean_value))
            )
            self.assertTrue(
                torch.allclose(std_layer_node, torch.tensor(std_value))
            )

        # ----- list parameter values -----
        mean_values = [0.17, -0.8, 0.53]
        std_values = [0.9, 1.34, 0.98]
        l = LogNormalLayer(
            scope=Scope([1]), n_nodes=3, mean=mean_values, std=std_values
        )

        for mean_layer_node, std_layer_node, mean_value, std_value in zip(
            l.mean, l.std, mean_values, std_values
        ):
            self.assertTrue(
                torch.allclose(mean_layer_node, torch.tensor(mean_value))
            )
            self.assertTrue(
                torch.allclose(std_layer_node, torch.tensor(std_value))
            )

        # wrong number of values
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            mean_values[:-1],
            std_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            mean_values,
            std_values[:-1],
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            [mean_values for _ in range(3)],
            std_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            mean_values,
            [std_values for _ in range(3)],
            n_nodes=3,
        )

        # ----- numpy parameter values -----

        l = LogNormalLayer(
            scope=Scope([1]),
            n_nodes=3,
            mean=np.array(mean_values),
            std=np.array(std_values),
        )

        for mean_layer_node, std_layer_node, mean_value, std_value in zip(
            l.mean, l.std, mean_values, std_values
        ):
            self.assertTrue(
                torch.allclose(mean_layer_node, torch.tensor(mean_value))
            )
            self.assertTrue(
                torch.allclose(std_layer_node, torch.tensor(std_value))
            )

        # wrong number of values
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            np.array(mean_values[:-1]),
            np.array(std_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            np.array(mean_values),
            np.array(std_values[:-1]),
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            np.array(mean_values),
            np.array([std_values for _ in range(3)]),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            LogNormalLayer,
            Scope([0]),
            np.array([mean_values for _ in range(3)]),
            np.array(std_values),
            n_nodes=3,
        )

        # ---- different scopes -----
        l = LogNormalLayer(scope=Scope([1]), n_nodes=3)
        for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, LogNormalLayer, Scope([0]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, LogNormalLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, LogNormalLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = LogNormalLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)

        for layer_scope, node_scope in zip(l.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(LogNormalLayer.accepts([([FeatureTypes.Continuous], Scope([0])), ([FeatureTypes.Continuous], Scope([1]))]))

        # LogNormal feature type class
        self.assertTrue(LogNormalLayer.accepts([([FeatureTypes.LogNormal], Scope([0])), ([FeatureTypes.Continuous], Scope([1]))]))

        # LogNormal feature type instance
        self.assertTrue(LogNormalLayer.accepts([([FeatureTypes.LogNormal(0.0, 1.0)], Scope([0])), ([FeatureTypes.Continuous], Scope([1]))]))

        # invalid feature type
        self.assertFalse(LogNormalLayer.accepts([([FeatureTypes.Discrete], Scope([0])), ([FeatureTypes.Continuous], Scope([1]))]))

        # conditional scope
        self.assertFalse(LogNormalLayer.accepts([([FeatureTypes.Continuous], Scope([0], [1]))]))

        # scope length does not match number of types
        self.assertFalse(LogNormalLayer.accepts([([FeatureTypes.Continuous], Scope([0, 1]))]))

        # multivariate signature
        self.assertFalse(LogNormalLayer.accepts([([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1]))]))

    def test_initialization_from_signatures(self):

        log_normal = LogNormalLayer.from_signatures([([FeatureTypes.Continuous], Scope([0])), ([FeatureTypes.Continuous], Scope([1]))])
        self.assertTrue(torch.all(log_normal.mean == torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.all(log_normal.std == torch.tensor([1.0, 1.0])))
        self.assertTrue(log_normal.scopes_out == [Scope([0]), Scope([1])])

        log_normal = LogNormalLayer.from_signatures([([FeatureTypes.LogNormal], Scope([0])), ([FeatureTypes.LogNormal], Scope([1]))])
        self.assertTrue(torch.all(log_normal.mean == torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.all(log_normal.std == torch.tensor([1.0, 1.0])))
        self.assertTrue(log_normal.scopes_out == [Scope([0]), Scope([1])])

        log_normal = LogNormalLayer.from_signatures([([FeatureTypes.LogNormal(-1.0, 1.5)], Scope([0])), ([FeatureTypes.LogNormal(1.0, 0.5)], Scope([1]))])
        self.assertTrue(torch.all(log_normal.mean == torch.tensor([-1.0, 1.0])))
        self.assertTrue(torch.all(log_normal.std == torch.tensor([1.5, 0.5])))
        self.assertTrue(log_normal.scopes_out == [Scope([0]), Scope([1])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(ValueError, LogNormalLayer.from_signatures, [([FeatureTypes.Discrete], Scope([0]))])

        # conditional scope
        self.assertRaises(ValueError, LogNormalLayer.from_signatures, [([FeatureTypes.Continuous], Scope([0], [1]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, LogNormalLayer.from_signatures, [([FeatureTypes.Continuous], Scope([0, 1]))])

        # multivariate signature
        self.assertRaises(ValueError, LogNormalLayer.from_signatures, [([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(LogNormalLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(LogNormalLayer, AutoLeaf.infer([([FeatureTypes.LogNormal], Scope([0])), ([FeatureTypes.LogNormal], Scope([1]))]))

        # make sure AutoLeaf can return correctly instantiated object
        log_normal = AutoLeaf([([FeatureTypes.LogNormal(mean=-1.0, std=1.5)], Scope([0])), ([FeatureTypes.LogNormal(mean=1.0, std=0.5)], Scope([1]))])
        self.assertTrue(torch.all(log_normal.mean == torch.tensor([-1.0, 1.0])))
        self.assertTrue(torch.all(log_normal.std == torch.tensor([1.5, 0.5])))
        self.assertTrue(log_normal.scopes_out == [Scope([0]), Scope([1])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = LogNormalLayer(
            scope=Scope([1]), mean=[0.73, -0.29], std=[0.41, 1.9], n_nodes=2
        )

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(torch.allclose(l.mean, l_marg.mean))
        self.assertTrue(torch.allclose(l.std, l_marg.std))

        # ---------- different scopes -----------

        l = LogNormalLayer(
            scope=[Scope([1]), Scope([0])], mean=[0.73, -0.29], std=[0.41, 1.9]
        )

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, LogNormal))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertTrue(torch.allclose(l_marg.mean, torch.tensor(-0.29)))
        self.assertTrue(torch.allclose(l_marg.std, torch.tensor(1.9)))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, LogNormalLayer))
        self.assertEqual(len(l_marg.scopes_out), 1)
        self.assertTrue(torch.allclose(l_marg.mean, torch.tensor(-0.29)))
        self.assertTrue(torch.allclose(l_marg.std, torch.tensor(1.9)))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
        self.assertTrue(torch.allclose(l.mean, l_marg.mean))
        self.assertTrue(torch.allclose(l.std, l_marg.std))

    def test_layer_dist(self):

        mean_values = [0.73, -0.29, 0.5]
        std_values = [0.9, 1.34, 0.98]
        l = LogNormalLayer(
            scope=Scope([1]), mean=mean_values, std=std_values, n_nodes=3
        )

        # ----- full dist -----
        dist = l.dist()

        for mean_value, std_value, mean_dist, std_dist in zip(
            mean_values, std_values, dist.loc, dist.scale
        ):
            self.assertTrue(torch.allclose(torch.tensor(mean_value), mean_dist))
            self.assertTrue(torch.allclose(torch.tensor(std_value), std_dist))

        # ----- partial dist -----
        dist = l.dist([1, 2])

        for mean_value, std_value, mean_dist, std_dist in zip(
            mean_values[1:], std_values[1:], dist.loc, dist.scale
        ):
            self.assertTrue(torch.allclose(torch.tensor(mean_value), mean_dist))
            self.assertTrue(torch.allclose(torch.tensor(std_value), std_dist))

        dist = l.dist([1, 0])

        for mean_value, std_value, mean_dist, std_dist in zip(
            reversed(mean_values[:-1]),
            reversed(std_values[:-1]),
            dist.loc,
            dist.scale,
        ):
            self.assertTrue(torch.allclose(torch.tensor(mean_value), mean_dist))
            self.assertTrue(torch.allclose(torch.tensor(std_value), std_dist))

    def test_layer_backend_conversion_1(self):

        torch_layer = LogNormalLayer(
            scope=[Scope([0]), Scope([1]), Scope([0])],
            mean=[0.2, 0.9, 0.31],
            std=[1.9, 0.3, 0.71],
        )
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(
            np.allclose(base_layer.mean, torch_layer.mean.detach().numpy())
        )
        self.assertTrue(
            np.allclose(base_layer.std, torch_layer.std.detach().numpy())
        )
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

    def test_layer_backend_conversion_2(self):

        base_layer = BaseLogNormalLayer(
            scope=[Scope([0]), Scope([1]), Scope([0])],
            mean=[0.2, 0.9, 0.31],
            std=[1.9, 0.3, 0.71],
        )
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(
            np.allclose(base_layer.mean, torch_layer.mean.detach().numpy())
        )
        self.assertTrue(
            np.allclose(base_layer.std, torch_layer.std.detach().numpy())
        )
        self.assertEqual(base_layer.n_out, torch_layer.n_out)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
