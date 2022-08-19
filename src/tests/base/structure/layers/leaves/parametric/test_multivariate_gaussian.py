from spflow.base.structure.layers.leaves.parametric.multivariate_gaussian import MultivariateGaussianLayer, marginalize
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.meta.scope.scope import Scope
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = MultivariateGaussianLayer(scope=Scope([1,0]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1,0]), Scope([1,0]), Scope([1,0])]))
        # make sure parameter properties works correctly
        mean_values = l.mean
        cov_values = l.cov
        for node, node_mean, node_cov in zip(l.nodes, mean_values, cov_values):
            self.assertTrue(np.all(node.mean == node_mean))
            self.assertTrue(np.all(node.cov == node_cov))

        # ----- float/int parameter values ----- 
        mean_value = [0.0, -1.0, 2.3]
        cov_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        l = MultivariateGaussianLayer(scope=Scope([1,0,2]), n_nodes=3, mean=mean_value, cov=cov_value)

        for node in l.nodes:
            self.assertTrue(np.all(node.mean == mean_value))
            self.assertTrue(np.all(node.cov == cov_value))

        # ----- list parameter values -----
        mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
        cov_values = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
            [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]]
        ]
        l = MultivariateGaussianLayer(scope=Scope([0,1,2]), n_nodes=3, mean=mean_values, cov=cov_values)

        for node, node_mean, node_cov in zip(l.nodes, mean_values, cov_values):
            self.assertTrue(np.all(node.mean == node_mean))
            self.assertTrue(np.all(node.cov == node_cov))
        
        # wrong number of values
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), mean_values[:-1], cov_values, n_nodes=3)
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), mean_values, cov_values[:-1], n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), mean_values, [cov_values for _ in range(3)], n_nodes=3)
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), [mean_values for _ in range(3)], cov_values, n_nodes=3)

        # ----- numpy parameter values -----

        l = MultivariateGaussianLayer(scope=Scope([0,1,2]), n_nodes=3, mean=np.array(mean_values), cov=np.array(cov_values))

        for node, node_mean, node_cov in zip(l.nodes, mean_values, cov_values):
            self.assertTrue(np.all(node.mean == node_mean))
            self.assertTrue(np.all(node.cov == node_cov))
        
        # wrong number of values
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), np.array(mean_values[:-1]), np.array(cov_values), n_nodes=3)
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), np.array(mean_values), np.array(cov_values[:-1]), n_nodes=3)
        # wrong number of dimensions (nested list)
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), mean_values, np.array([cov_values for _ in range(3)]), n_nodes=3)
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0,1,2]), np.array([mean_values for _ in range(3)]), cov_values, n_nodes=3)

        # ---- different scopes -----
        l = MultivariateGaussianLayer(scope=Scope([0,1,2]), n_nodes=3)
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
        self.assertTrue(np.all(l.mean == l_marg.mean))
        self.assertTrue(np.all(l.cov == l_marg.cov))
    
        # ---------- different scopes -----------

        l = MultivariateGaussianLayer(scope=[Scope([0,2]), Scope([1,3])], mean=[[-0.2, 1.3],[3.7, -0.9]], cov=[[[1.3, 0.0],[0.0, 1.1]],[[0.5, 0.0],[0.0, 0.7]]])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1,2,3]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [0,2], prune=True)
        self.assertTrue(isinstance(l_marg, MultivariateGaussian))
        self.assertEqual(l_marg.scope, Scope([1,3]))
        self.assertTrue(np.all(l_marg.mean == np.array([3.7, -0.9])))
        self.assertTrue(np.all(l_marg.cov == np.array([[0.5, 0.0], [0.0, 0.7]])))

        l_marg = marginalize(l, [0,1,2], prune=True)
        self.assertTrue(isinstance(l_marg, Gaussian))
        self.assertEqual(l_marg.scope, Scope([3]))
        self.assertTrue(np.all(l_marg.mean == np.array(-0.9)))
        self.assertTrue(np.all(l_marg.std == np.array(np.sqrt(0.7))))

        l_marg = marginalize(l, [0,2], prune=False)
        self.assertTrue(isinstance(l_marg, MultivariateGaussianLayer))
        self.assertEqual(l_marg.scopes_out, [Scope([1,3])])
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertTrue(np.all(l_marg.mean == np.array([3.7, -0.9])))
        self.assertTrue(np.all(l_marg.cov == np.array([[0.5, 0.0], [0.0, 0.7]])))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [4])

        self.assertTrue(l_marg.scopes_out == [Scope([0,2]), Scope([1,3])])
        self.assertTrue(np.all(l.mean == l_marg.mean))
        self.assertTrue(np.all(l.cov == l_marg.cov))

    def test_get_params(self):

        layer = MultivariateGaussianLayer(scope=Scope([0,1]), mean=[[-0.73, 0.29], [0.36, -1.4]], cov=[[[1.0, 0.92], [0.92, 1.2]], [[1.0, 0.3],[0.3, 1.4]]], n_nodes=2)

        mean, cov, *others = layer.get_params()

        self.assertTrue(len(others) == 0)
        self.assertTrue(np.allclose(mean, np.array([[-0.73, 0.29], [0.36, -1.4]])))
        self.assertTrue(np.allclose(cov, np.array([[[1.0, 0.92], [0.92, 1.2]], [[1.0, 0.3],[0.3, 1.4]]])))


if __name__ == "__main__":
    unittest.main()