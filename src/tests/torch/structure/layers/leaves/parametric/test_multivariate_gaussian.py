from spflow.torch.structure.layers.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussianLayer,
    marginalize,
    toTorch,
    toBase,
)
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.torch.structure.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian,
)
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.layers.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussianLayer as BaseMultivariateGaussianLayer,
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

        l = MultivariateGaussianLayer(scope=Scope([1, 0]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(
                l.scopes_out == [Scope([1, 0]), Scope([1, 0]), Scope([1, 0])]
            )
        )
        mean_values = l.mean
        cov_values = l.cov
        # make sure parameter properties works correctly
        for node, node_mean, node_cov in zip(l.nodes, mean_values, cov_values):
            self.assertTrue(torch.all(node.mean == node_mean))
            self.assertTrue(torch.all(node.cov == node_cov))

        # ----- single mean/cov list parameter values -----
        mean_value = [0.0, -1.0, 2.3]
        cov_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        l = MultivariateGaussianLayer(
            scope=Scope([1, 0, 2]), n_nodes=3, mean=mean_value, cov=cov_value
        )

        for node in l.nodes:
            self.assertTrue(torch.all(node.mean == torch.tensor(mean_value)))
            self.assertTrue(torch.all(node.cov == torch.tensor(cov_value)))

        # ----- multiple mean/cov list parameter values -----
        mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
        cov_values = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
            [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]],
        ]
        l = MultivariateGaussianLayer(
            scope=Scope([0, 1, 2]), n_nodes=3, mean=mean_values, cov=cov_values
        )

        for node, node_mean, node_cov in zip(l.nodes, mean_values, cov_values):
            self.assertTrue(torch.all(node.mean == torch.tensor(node_mean)))
            self.assertTrue(torch.allclose(node.cov, torch.tensor(node_cov)))

        # wrong number of values
        self.assertRaises(
            ValueError,
            MultivariateGaussianLayer,
            Scope([0, 1, 2]),
            mean_values[:-1],
            cov_values,
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            MultivariateGaussianLayer,
            Scope([0, 1, 2]),
            mean_values,
            cov_values[:-1],
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            MultivariateGaussianLayer,
            Scope([0, 1, 2]),
            mean_values,
            [cov_values for _ in range(3)],
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            MultivariateGaussianLayer,
            Scope([0, 1, 2]),
            [mean_values for _ in range(3)],
            cov_values,
            n_nodes=3,
        )

        # ----- numpy parameter values -----

        l = MultivariateGaussianLayer(
            scope=Scope([0, 1, 2]),
            n_nodes=3,
            mean=np.array(mean_values),
            cov=np.array(cov_values),
        )

        for node, node_mean, node_cov in zip(l.nodes, mean_values, cov_values):
            self.assertTrue(torch.all(node.mean == torch.tensor(node_mean)))
            self.assertTrue(torch.allclose(node.cov, torch.tensor(node_cov)))

        # wrong number of values
        self.assertRaises(
            ValueError,
            MultivariateGaussianLayer,
            Scope([0, 1, 2]),
            np.array(mean_values[:-1]),
            np.array(cov_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            MultivariateGaussianLayer,
            Scope([0, 1, 2]),
            np.array(mean_values),
            np.array(cov_values[:-1]),
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            MultivariateGaussianLayer,
            Scope([0, 1, 2]),
            mean_values,
            np.array([cov_values for _ in range(3)]),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            MultivariateGaussianLayer,
            Scope([0, 1, 2]),
            np.array([mean_values for _ in range(3)]),
            cov_values,
            n_nodes=3,
        )

        # ---- different scopes -----
        l = MultivariateGaussianLayer(
            scope=[Scope([0, 1, 2]), Scope([1, 3]), Scope([2])], n_nodes=3
        )
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(
            ValueError, MultivariateGaussianLayer, Scope([0, 1, 2]), n_nodes=0
        )

        # ----- invalid scope -----
        self.assertRaises(
            ValueError, MultivariateGaussianLayer, Scope([]), n_nodes=3
        )
        self.assertRaises(ValueError, MultivariateGaussianLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1, 2, 3]), Scope([0, 1, 4]), Scope([0, 2, 3])]
        l = MultivariateGaussianLayer(scope=scopes, n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

    def test_accept(self):

        # continuous meta types
        self.assertTrue(MultivariateGaussianLayer.accepts([([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1])), ([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([1, 2]))]))

        # Gaussian feature type class
        self.assertTrue(MultivariateGaussianLayer.accepts([([FeatureTypes.Gaussian, FeatureTypes.Gaussian], Scope([0, 1])), ([FeatureTypes.Gaussian, FeatureTypes.Gaussian], Scope([1, 2]))]))

        # Gaussian feature type instance
        self.assertTrue(MultivariateGaussianLayer.accepts([([FeatureTypes.Gaussian(0.0, 1.0), FeatureTypes.Gaussian(0.0, 1.0)], Scope([0, 1])), ([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([1, 2]))]))

        # continuous meta and Gaussian feature types
        self.assertTrue(MultivariateGaussianLayer.accepts([([FeatureTypes.Continuous, FeatureTypes.Gaussian], Scope([0, 1]))]))

        # invalid feature type
        self.assertFalse(MultivariateGaussianLayer.accepts([([FeatureTypes.Discrete, FeatureTypes.Continuous], Scope([0, 1]))]))

        # conditional scope
        self.assertFalse(MultivariateGaussianLayer.accepts([([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1], [2]))]))

        # scope length does not match number of types
        self.assertFalse(MultivariateGaussianLayer.accepts([([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1, 2]))]))

    def test_initialization_from_signatures(self):

        multivariate_gaussian = MultivariateGaussianLayer.from_signatures([([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1])), ([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([1, 2]))])
        self.assertTrue(all([torch.all(mean == torch.zeros(2)) for mean in multivariate_gaussian.mean]))
        self.assertTrue(all([torch.all(cov == torch.eye(2)) for cov in multivariate_gaussian.cov]))
        self.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1]), Scope([1, 2])])

        multivariate_gaussian = MultivariateGaussianLayer.from_signatures([([FeatureTypes.Gaussian, FeatureTypes.Gaussian], Scope([0, 1])), ([FeatureTypes.Gaussian, FeatureTypes.Gaussian], Scope([1, 2]))])
        self.assertTrue(all([torch.all(mean == torch.zeros(2)) for mean in multivariate_gaussian.mean]))
        self.assertTrue(all([torch.all(cov == torch.eye(2)) for cov in multivariate_gaussian.cov]))
        self.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1]), Scope([1, 2])])
    
        multivariate_gaussian = MultivariateGaussianLayer.from_signatures([([FeatureTypes.Gaussian(-1.0, 1.5), FeatureTypes.Gaussian(1.0, 0.5)], Scope([0, 1])), ([FeatureTypes.Gaussian(1.0, 0.5), FeatureTypes.Gaussian(-1.0, 1.5)], Scope([1, 2]))])
        self.assertTrue(torch.all(multivariate_gaussian.mean[0] == torch.tensor([-1.0, 1.0])))
        self.assertTrue(torch.all(multivariate_gaussian.mean[1] == torch.tensor([1.0, -1.0])))
        self.assertTrue(torch.allclose(multivariate_gaussian.cov[0], torch.tensor([[1.5, 0.0], [0.0, 0.5]])))
        self.assertTrue(torch.allclose(multivariate_gaussian.cov[1], torch.tensor([[0.5, 0.0], [0.0, 1.5]])))
        self.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1]), Scope([1, 2])])
        
        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(ValueError, MultivariateGaussianLayer.from_signatures, [([FeatureTypes.Discrete, FeatureTypes.Continuous], Scope([0, 1]))])

        # conditional scope
        self.assertRaises(ValueError, MultivariateGaussianLayer.from_signatures, [([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1], [2]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, MultivariateGaussianLayer.from_signatures, [([FeatureTypes.Continuous, FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(MultivariateGaussianLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(MultivariateGaussianLayer, AutoLeaf.infer([([FeatureTypes.Gaussian, FeatureTypes.Gaussian], Scope([0, 1])), ([FeatureTypes.Gaussian, FeatureTypes.Gaussian], Scope([1, 2]))]))

        # make sure AutoLeaf can return correctly instantiated object
        multivariate_gaussian = AutoLeaf([([FeatureTypes.Gaussian(mean=-1.0, std=1.5), FeatureTypes.Gaussian(mean=1.0, std=0.5)], Scope([0, 1])), ([FeatureTypes.Gaussian(1.0, 0.5), FeatureTypes.Gaussian(-1.0, 1.5)], Scope([1, 2]))])
        self.assertTrue(torch.all(multivariate_gaussian.mean[0] == torch.tensor([-1.0, 1.0])))
        self.assertTrue(torch.all(multivariate_gaussian.mean[1] == torch.tensor([1.0, -1.0])))
        self.assertTrue(torch.allclose(multivariate_gaussian.cov[0], torch.tensor([[1.5, 0.0], [0.0, 0.5]])))
        self.assertTrue(torch.allclose(multivariate_gaussian.cov[1], torch.tensor([[0.5, 0.0], [0.0, 1.5]])))
        self.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1]), Scope([1, 2])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = MultivariateGaussianLayer(
            scope=[Scope([0, 1]), Scope([0, 1])],
            mean=[[-0.2, 1.3], [3.7, -0.9]],
            cov=[[[1.3, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 0.7]]],
        )
        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([0, 1]), Scope([0, 1])])
        self.assertTrue(
            all([torch.all(m1 == m2) for m1, m2 in zip(l.mean, l_marg.mean)])
        )
        self.assertTrue(
            all([torch.all(c1 == c2) for c1, c2 in zip(l.cov, l_marg.cov)])
        )

        # ---------- different scopes -----------

        l = MultivariateGaussianLayer(
            scope=[Scope([0, 2]), Scope([1, 3])],
            mean=[[-0.2, 1.3], [3.7, -0.9]],
            cov=[[[1.3, 0.0], [0.0, 1.1]], [[0.5, 0.0], [0.0, 0.7]]],
        )

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1, 2, 3]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [0, 2], prune=True)
        self.assertTrue(isinstance(l_marg, MultivariateGaussian))
        self.assertEqual(l_marg.scope, Scope([1, 3]))
        self.assertTrue(torch.all(l_marg.mean == torch.tensor([3.7, -0.9])))
        self.assertTrue(
            torch.allclose(l_marg.cov, torch.tensor([[0.5, 0.0], [0.0, 0.7]]))
        )

        l_marg = marginalize(l, [0, 1, 2], prune=True)
        self.assertTrue(isinstance(l_marg, Gaussian))
        self.assertEqual(l_marg.scope, Scope([3]))
        self.assertTrue(torch.all(l_marg.mean == torch.tensor(-0.9)))
        self.assertTrue(torch.all(l_marg.std == torch.tensor(np.sqrt(0.7))))

        l_marg = marginalize(l, [0, 2], prune=False)
        self.assertTrue(isinstance(l_marg, MultivariateGaussianLayer))
        self.assertEqual(l_marg.scopes_out, [Scope([1, 3])])
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertTrue(torch.all(l_marg.mean[0] == torch.tensor([3.7, -0.9])))
        self.assertTrue(
            torch.allclose(
                l_marg.cov[0], torch.tensor([[0.5, 0.0], [0.0, 0.7]])
            )
        )

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [4])

        self.assertTrue(l_marg.scopes_out == [Scope([0, 2]), Scope([1, 3])])
        self.assertTrue(
            all([torch.all(m1 == m2) for m1, m2 in zip(l.mean, l_marg.mean)])
        )
        self.assertTrue(
            all([torch.all(c1 == c2) for c1, c2 in zip(l.cov, l_marg.cov)])
        )

    def test_layer_dist(self):

        mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
        cov_values = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
            [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]],
        ]
        l = MultivariateGaussianLayer(
            scope=Scope([0, 1, 2]), mean=mean_values, cov=cov_values, n_nodes=3
        )

        # ----- full dist -----
        dist_list = l.dist()

        for mean_value, cov_value, dist in zip(
            mean_values, cov_values, dist_list
        ):
            self.assertTrue(torch.allclose(torch.tensor(mean_value), dist.mean))
            self.assertTrue(
                torch.allclose(torch.tensor(cov_value), dist.covariance_matrix)
            )

        # ----- partial dist -----
        dist_list = l.dist([1, 2])

        for mean_value, cov_value, dist in zip(
            mean_values[1:], cov_values[1:], dist_list
        ):
            self.assertTrue(torch.allclose(torch.tensor(mean_value), dist.mean))
            self.assertTrue(
                torch.allclose(torch.tensor(cov_value), dist.covariance_matrix)
            )

        dist_list = l.dist([1, 0])

        for mean_value, cov_value, dist in zip(
            reversed(mean_values[:-1]), reversed(cov_values[:-1]), dist_list
        ):
            self.assertTrue(torch.allclose(torch.tensor(mean_value), dist.mean))
            self.assertTrue(
                torch.allclose(torch.tensor(cov_value), dist.covariance_matrix)
            )

    def test_layer_backend_conversion_1(self):

        mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
        cov_values = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
            [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]],
        ]
        torch_layer = MultivariateGaussianLayer(
            scope=[Scope([0, 1, 2]), Scope([1, 2, 3]), Scope([0, 1, 2])],
            mean=mean_values,
            cov=cov_values,
        )
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

        for torch_mean, base_mean, torch_cov, base_cov in zip(
            torch_layer.mean, base_layer.mean, torch_layer.cov, base_layer.cov
        ):
            self.assertTrue(np.allclose(base_mean, torch_mean.detach().numpy()))
            self.assertTrue(np.allclose(base_cov, torch_cov.detach().numpy()))

    def test_layer_backend_conversion_2(self):

        mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
        cov_values = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
            [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]],
        ]
        base_layer = BaseMultivariateGaussianLayer(
            scope=[Scope([0, 1, 2]), Scope([1, 2, 3]), Scope([0, 1, 2])],
            mean=mean_values,
            cov=cov_values,
        )
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

        for torch_mean, base_mean, torch_cov, base_cov in zip(
            torch_layer.mean, base_layer.mean, torch_layer.cov, base_layer.cov
        ):
            self.assertTrue(np.allclose(base_mean, torch_mean.detach().numpy()))
            self.assertTrue


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
