from spflow.torch.structure.layers.leaves.parametric.multivariate_gaussian import MultivariateGaussianLayer, marginalize, toTorch, toBase
from spflow.torch.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.layers.leaves.parametric.multivariate_gaussian import MultivariateGaussianLayer as BaseMultivariateGaussianLayer
from spflow.meta.scope.scope import Scope
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

        l = MultivariateGaussianLayer(scope=Scope([1,0]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1,0]), Scope([1,0]), Scope([1,0])]))
        mean_values = l.mean
        cov_values = l.cov
        # make sure parameter properties works correctly
        for node, node_mean, node_cov in zip(l.nodes,  mean_values, cov_values):
            self.assertTrue(torch.all(node.mean == node_mean))
            self.assertTrue(torch.all(node.cov == node_cov))
    
        # ----- single mean/cov list parameter values ----- 
        mean_value = [0.0, -1.0, 2.3]
        cov_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        l = MultivariateGaussianLayer(scope=Scope([1,0,2]), n_nodes=3, mean=mean_value, cov=cov_value)

        for node in l.nodes:
            self.assertTrue(torch.all(node.mean == torch.tensor(mean_value)))
            self.assertTrue(torch.all(node.cov == torch.tensor(cov_value)))
        
        # ----- multiple mean/cov list parameter values -----
        mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
        cov_values = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
            [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]]
        ]
        l = MultivariateGaussianLayer(scope=Scope([0,1,2]), n_nodes=3, mean=mean_values, cov=cov_values)

        for node, node_mean, node_cov in zip(l.nodes, mean_values, cov_values):
            self.assertTrue(torch.all(node.mean == torch.tensor(node_mean)))
            self.assertTrue(torch.allclose(node.cov, torch.tensor(node_cov)))
        
        # wrong number of values
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), mean_values[:-1], cov_values, n_nodes=3)
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), mean_values, cov_values[:-1], n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), mean_values, [cov_values for _ in range(3)], n_nodes=3)
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), [mean_values for _ in range(3)], cov_values, n_nodes=3)

        # ----- numpy parameter values -----

        l = MultivariateGaussianLayer(scope=Scope([0,1,2]), n_nodes=3, mean=np.array(mean_values), cov=np.array(cov_values))

        for node, node_mean, node_cov in zip(l.nodes, mean_values, cov_values):
            self.assertTrue(torch.all(node.mean == torch.tensor(node_mean)))
            self.assertTrue(torch.allclose(node.cov, torch.tensor(node_cov)))
        
        # wrong number of values
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), np.array(mean_values[:-1]), np.array(cov_values), n_nodes=3)
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), np.array(mean_values), np.array(cov_values[:-1]), n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), mean_values, np.array([cov_values for _ in range(3)]), n_nodes=3)
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), np.array([mean_values for _ in range(3)]), cov_values, n_nodes=3)

        # ---- different scopes -----
        l = MultivariateGaussianLayer(scope=[Scope([0,1,2]), Scope([1,3]), Scope([2])], n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)
        
        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, MultivariateGaussianLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1,2,3]), Scope([0,1,4]), Scope([0,2,3])]
        l = MultivariateGaussianLayer(scope=scopes, n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = MultivariateGaussianLayer(scope=[Scope([0,1]), Scope([0,1])], mean=[[-0.2, 1.3],[3.7, -0.9]], cov=[[[1.3, 0.0],[0.0, 1.0]],[[0.5, 0.0],[0.0, 0.7]]])
        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([0,1]), Scope([0,1])])
        self.assertTrue(all([torch.all(m1 == m2) for m1, m2 in zip(l.mean, l_marg.mean)]))
        self.assertTrue(all([torch.all(c1 == c2) for c1, c2 in zip(l.cov, l_marg.cov)]))
    
        # ---------- different scopes -----------

        l = MultivariateGaussianLayer(scope=[Scope([0,2]), Scope([1,3])], mean=[[-0.2, 1.3],[3.7, -0.9]], cov=[[[1.3, 0.0],[0.0, 1.1]],[[0.5, 0.0],[0.0, 0.7]]])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1,2,3]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [0,2], prune=True)
        self.assertTrue(isinstance(l_marg, MultivariateGaussian))
        self.assertEqual(l_marg.scope, Scope([1,3]))
        self.assertTrue(torch.all(l_marg.mean == torch.tensor([3.7, -0.9])))
        self.assertTrue(torch.allclose(l_marg.cov, torch.tensor([[0.5, 0.0], [0.0, 0.7]])))

        l_marg = marginalize(l, [0,1,2], prune=True)
        self.assertTrue(isinstance(l_marg, Gaussian))
        self.assertEqual(l_marg.scope, Scope([3]))
        self.assertTrue(torch.all(l_marg.mean == torch.tensor(-0.9)))
        self.assertTrue(torch.all(l_marg.std == torch.tensor(np.sqrt(0.7))))

        l_marg = marginalize(l, [0,2], prune=False)
        self.assertTrue(isinstance(l_marg, MultivariateGaussianLayer))
        self.assertEqual(l_marg.scopes_out, [Scope([1,3])])
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertTrue(torch.all(l_marg.mean[0] == torch.tensor([3.7, -0.9])))
        self.assertTrue(torch.allclose(l_marg.cov[0], torch.tensor([[0.5, 0.0], [0.0, 0.7]])))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [4])

        self.assertTrue(l_marg.scopes_out == [Scope([0,2]), Scope([1,3])])
        self.assertTrue(all([torch.all(m1 == m2) for m1, m2 in zip(l.mean, l_marg.mean)]))
        self.assertTrue(all([torch.all(c1 == c2) for c1, c2 in zip(l.cov, l_marg.cov)]))

    def test_layer_dist(self):

        mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
        cov_values = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
            [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]]
        ]
        l = MultivariateGaussianLayer(scope=Scope([0,1,2]), mean=mean_values, cov=cov_values, n_nodes=3)

        # ----- full dist -----
        dist_list = l.dist()

        for mean_value, cov_value, dist in zip(mean_values, cov_values, dist_list):
            self.assertTrue(torch.allclose(torch.tensor(mean_value), dist.mean))
            self.assertTrue(torch.allclose(torch.tensor(cov_value), dist.covariance_matrix))

        # ----- partial dist -----
        dist_list = l.dist([1,2])

        for mean_value, cov_value, dist in zip(mean_values[1:], cov_values[1:], dist_list):
            self.assertTrue(torch.allclose(torch.tensor(mean_value), dist.mean))
            self.assertTrue(torch.allclose(torch.tensor(cov_value), dist.covariance_matrix))

        dist_list = l.dist([1,0])

        for mean_value, cov_value, dist in zip(reversed(mean_values[:-1]), reversed(cov_values[:-1]), dist_list):
            self.assertTrue(torch.allclose(torch.tensor(mean_value), dist.mean))
            self.assertTrue(torch.allclose(torch.tensor(cov_value), dist.covariance_matrix))

    def test_layer_backend_conversion_1(self):

        mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
        cov_values = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
            [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]]
        ]
        torch_layer = MultivariateGaussianLayer(scope=[Scope([0,1,2]), Scope([1,2,3]), Scope([0,1,2])], mean=mean_values, cov=cov_values)
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

        for torch_mean, base_mean, torch_cov, base_cov in zip(torch_layer.mean, base_layer.mean, torch_layer.cov, base_layer.cov):
            self.assertTrue(np.allclose(base_mean, torch_mean.detach().numpy()))
            self.assertTrue(np.allclose(base_cov, torch_cov.detach().numpy()))

    def test_layer_backend_conversion_2(self):

        mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
        cov_values = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
            [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]]
        ]
        base_layer = BaseMultivariateGaussianLayer(scope=[Scope([0,1,2]), Scope([1,2,3]), Scope([0,1,2])], mean=mean_values, cov=cov_values)
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

        for torch_mean, base_mean, torch_cov, base_cov in zip(torch_layer.mean, base_layer.mean, torch_layer.cov, base_layer.cov):
            self.assertTrue(np.allclose(base_mean, torch_mean.detach().numpy()))
            self.assertTrue


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()