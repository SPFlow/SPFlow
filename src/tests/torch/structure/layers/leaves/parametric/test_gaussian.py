from spflow.torch.structure.layers.leaves.parametric.gaussian import GaussianLayer, marginalize, toTorch, toBase
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.layers.leaves.parametric.gaussian import GaussianLayer as BaseGaussianLayer
from spflow.meta.scope.scope import Scope
import torch
import numpy as np
import unittest
import itertools


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
        l = GaussianLayer(scope=Scope([1]), n_nodes=3, mean=mean_values, std=std_values)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
        # make sure parameter properties works correctly
        for mean_layer_node, std_layer_node, mean_value, std_value in zip(l.mean, l.std, mean_values, std_values):
            self.assertTrue(torch.allclose(mean_layer_node, torch.tensor(mean_value)))
            self.assertTrue(torch.allclose(std_layer_node, torch.tensor(std_value)))

        # ----- float/int parameter values -----
        mean_value = 0.73
        std_value = 1.9
        l = GaussianLayer(scope=Scope([1]), n_nodes=3, mean=mean_value, std=std_value)

        for mean_layer_node, std_layer_node in zip(l.mean, l.std):
            self.assertTrue(torch.allclose(mean_layer_node, torch.tensor(mean_value)))
            self.assertTrue(torch.allclose(std_layer_node, torch.tensor(std_value)))

        # ----- list parameter values -----
        mean_values = [0.17, -0.8, 0.53]
        std_values = [0.9, 1.34, 0.98]
        l = GaussianLayer(scope=Scope([1]), n_nodes=3, mean=mean_values, std=std_values)

        for mean_layer_node, std_layer_node, mean_value, std_value in zip(l.mean, l.std, mean_values, std_values):
            self.assertTrue(torch.allclose(mean_layer_node, torch.tensor(mean_value)))
            self.assertTrue(torch.allclose(std_layer_node, torch.tensor(std_value)))

        # wrong number of values
        self.assertRaises(ValueError, GaussianLayer, Scope([0]), mean_values, std_values[:-1], n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, GaussianLayer, Scope([0]), [mean_values for _ in range(3)], [std_values for _ in range(3)], n_nodes=3)

        # ----- numpy parameter values -----

        l = GaussianLayer(scope=Scope([1]), n_nodes=3, mean=np.array(mean_values), std=np.array(std_values))

        for mean_layer_node, std_layer_node, mean_value, std_value in zip(l.mean, l.std, mean_values, std_values):
            self.assertTrue(torch.allclose(mean_layer_node, torch.tensor(mean_value)))
            self.assertTrue(torch.allclose(std_layer_node, torch.tensor(std_value)))

        # wrong number of values
        self.assertRaises(ValueError, GaussianLayer, Scope([0]), np.array(mean_values), np.array(std_values[:-1]), n_nodes=3)
        self.assertRaises(ValueError, GaussianLayer, Scope([0]), np.array([mean_values for _ in range(3)]), np.array([std_values for _ in range(3)]), n_nodes=3)

        # ---- different scopes -----
        l = GaussianLayer(scope=Scope([1]), n_nodes=3)
        for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, GaussianLayer, Scope([0]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, GaussianLayer, Scope([]), n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1]), Scope([0]), Scope([0])]
        l = GaussianLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)
        
        for layer_scope, node_scope in zip(l.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = GaussianLayer(scope=Scope([1]), mean=[0.73, -0.29], std=[0.41, 1.9], n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
        self.assertTrue(torch.allclose(l.mean, l_marg.mean))
        self.assertTrue(torch.allclose(l.std, l_marg.std))

        # ---------- different scopes -----------

        l = GaussianLayer(scope=[Scope([1]), Scope([0])], mean=[0.73, -0.29], std=[0.41, 1.9])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, Gaussian))
        self.assertEqual(l_marg.scope, Scope([0]))
        self.assertTrue(torch.allclose(l_marg.mean, torch.tensor(-0.29)))
        self.assertTrue(torch.allclose(l_marg.std, torch.tensor(1.9)))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, GaussianLayer))
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
        l = GaussianLayer(scope=Scope([1]), mean=mean_values, std=std_values, n_nodes=3)

        # ----- full dist -----
        dist = l.dist()

        for mean_value, std_value, mean_dist, std_dist in zip(mean_values, std_values, dist.loc ,dist.scale):
            self.assertTrue(torch.allclose(torch.tensor(mean_value), mean_dist))
            self.assertTrue(torch.allclose(torch.tensor(std_value), std_dist))
        
        # ----- partial dist -----
        dist = l.dist([1,2])

        for mean_value, std_value, mean_dist, std_dist in zip(mean_values[1:], std_values[1:], dist.loc, dist.scale):
            self.assertTrue(torch.allclose(torch.tensor(mean_value), mean_dist))
            self.assertTrue(torch.allclose(torch.tensor(std_value), std_dist))

        dist = l.dist([1,0])

        for mean_value, std_value, mean_dist, std_dist in zip(reversed(mean_values[:-1]), reversed(std_values[:-1]), dist.loc, dist.scale):
            self.assertTrue(torch.allclose(torch.tensor(mean_value), mean_dist))
            self.assertTrue(torch.allclose(torch.tensor(std_value), std_dist))

    def test_layer_backend_conversion_1(self):
        
        torch_layer = GaussianLayer(scope=[Scope([0]), Scope([1]), Scope([0])], mean=[0.2, 0.9, 0.31], std=[1.9, 0.3, 0.71])
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.mean, torch_layer.mean.detach().numpy()))
        self.assertTrue(np.allclose(base_layer.std, torch_layer.std.detach().numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)
    
    def test_layer_backend_conversion_2(self):

        base_layer = BaseGaussianLayer(scope=[Scope([0]), Scope([1]), Scope([0])], mean=[0.2, 0.9, 0.31], std=[1.9, 0.3, 0.71])
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertTrue(np.allclose(base_layer.mean, torch_layer.mean.detach().numpy()))
        self.assertTrue(np.allclose(base_layer.std, torch_layer.std.detach().numpy()))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()