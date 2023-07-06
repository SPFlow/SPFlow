import unittest

import numpy as np
import torch

from spflow.base.structure.spn import CondGaussianLayer as BaseCondGaussianLayer
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.torch.structure.spn import CondGaussian as CondGaussianTorch
from spflow.torch.structure.spn import CondGaussianLayer as CondGaussianLayerTorch

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.general.layers.leaves.parametric.general_cond_gaussian import CondGaussianLayer


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_initialization(self):

        # ----- check attributes after correct initialization -----

        l = CondGaussianLayer(scope=Scope([1], [0]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.scopes_out), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])]))

        # ---- different scopes -----
        l = CondGaussianLayer(scope=Scope([1], [0]), n_nodes=3)
        for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
            self.assertEqual(layer_scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, CondGaussianLayer, Scope([0], [1]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, CondGaussianLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, CondGaussianLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
        l = CondGaussianLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3)

        for layer_scope, node_scope in zip(l.scopes_out, scopes):
            self.assertEqual(layer_scope, node_scope)

        # -----number of cond_f functions -----
        CondGaussianLayer(
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[
                lambda data: {"mean": 0.0, "std": 1.0},
                lambda data: {"mean": 0.0, "std": 1.0},
            ],
        )
        self.assertRaises(
            ValueError,
            CondGaussianLayer,
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"mean": 0.0, "std": 1.0}],
        )

    def test_retrieve_params(self):

        # ----- float/int parameter values -----
        mean_value = 0.73
        std_value = 1.9
        l = CondGaussianLayer(
            scope=Scope([1], [0]),
            n_nodes=3,
            cond_f=lambda data: {"mean": mean_value, "std": std_value},
        )

        for mean_layer_node, std_layer_node in zip(*l.retrieve_params(torch.tensor([[1]]), DispatchContext())):
            self.assertTrue(torch.allclose(mean_layer_node, torch.tensor(mean_value)))
            self.assertTrue(torch.allclose(std_layer_node, torch.tensor(std_value)))

        # ----- list parameter values -----
        mean_values = [0.17, -0.8, 0.53]
        std_values = [0.9, 1.34, 0.98]
        l = CondGaussianLayer(
            scope=Scope([1], [0]),
            n_nodes=3,
            cond_f=lambda data: {"mean": mean_values, "std": std_values},
        )

        for mean_value, std_value, mean_layer_node, std_layer_node in zip(
            mean_values, std_values, *l.retrieve_params(torch.tensor([[1]]), DispatchContext())
        ):
            self.assertTrue(torch.allclose(mean_layer_node, torch.tensor(mean_value)))
            self.assertTrue(torch.allclose(std_layer_node, torch.tensor(std_value)))

        # wrong number of values
        l.set_cond_f(lambda data: {"mean": mean_values[:-1], "std": std_values})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )
        l.set_cond_f(lambda data: {"mean": mean_values, "std": std_values[:-1]})
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        # wrong number of dimensions (nested list)
        l.set_cond_f(
            lambda data: {
                "mean": [mean_values for _ in range(3)],
                "std": std_values,
            }
        )
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )
        l.set_cond_f(
            lambda data: {
                "mean": mean_values,
                "std": [std_values for _ in range(3)],
            }
        )
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        # ----- numpy parameter values -----
        l.set_cond_f(
            lambda data: {
                "mean": np.array(mean_values),
                "std": np.array(std_values),
            }
        )
        for mean_actual, std_actual, mean_node, std_node in zip(
            mean_values, std_values, *l.retrieve_params(torch.tensor([[1.0]]), DispatchContext())
        ):
            self.assertTrue(mean_node == mean_actual)
            self.assertTrue(std_node == std_actual)

        # wrong number of values
        l.set_cond_f(
            lambda data: {
                "mean": np.array(mean_values[:-1]),
                "std": np.array(std_values),
            }
        )
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )
        l.set_cond_f(
            lambda data: {
                "mean": np.array(mean_values),
                "std": np.array(std_values[:-1]),
            }
        )
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

        # wrong number of dimensions (nested list)
        l.set_cond_f(
            lambda data: {
                "mean": np.array([mean_values for _ in range(3)]),
                "std": np.array(std_values),
            }
        )
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )
        l.set_cond_f(
            lambda data: {
                "mean": np.array(mean_values),
                "std": np.array([std_values for _ in range(3)]),
            }
        )
        self.assertRaises(
            ValueError,
            l.retrieve_params,
            torch.tensor([[1]]),
            DispatchContext(),
        )

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            CondGaussianLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # feature type class
        self.assertTrue(
            CondGaussianLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian]),
                    FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # feature type instance
        self.assertTrue(
            CondGaussianLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian(0.0, 1.0)]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondGaussianLayer.accepts(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
                ]
            )
        )

        # non-conditional scope
        self.assertFalse(CondGaussianLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

        # multivariate signature
        self.assertFalse(
            CondGaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        gaussian = CondGaussianLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
            ]
        )
        self.assertTrue(gaussian.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        gaussian = CondGaussianLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Gaussian]),
            ]
        )
        self.assertTrue(gaussian.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        gaussian = CondGaussianLayer.from_signatures(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian(0.0, 1.0)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Gaussian(0.0, 1.0)]),
            ]
        )
        self.assertTrue(gaussian.scopes_out == [Scope([0], [2]), Scope([1], [2])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondGaussianLayer.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondGaussianLayer.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondGaussianLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondGaussianLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondGaussianLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian]),
                    FeatureContext(Scope([1], [2]), [FeatureTypes.Gaussian]),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        gaussian = AutoLeaf(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian(mean=-1.0, std=1.5)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Gaussian(mean=1.0, std=0.5)]),
            ]
        )
        self.assertTrue(isinstance(gaussian, CondGaussianLayerTorch))
        self.assertTrue(gaussian.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = CondGaussianLayer(scope=Scope([1], [0]), n_nodes=2)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

        # ---------- different scopes -----------

        l = CondGaussianLayer(scope=[Scope([1], [2]), Scope([0], [2])])

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [1], prune=True)
        self.assertTrue(isinstance(l_marg, CondGaussianTorch))
        self.assertEqual(l_marg.scope, Scope([0], [2]))

        l_marg = marginalize(l, [1], prune=False)
        self.assertTrue(isinstance(l_marg, CondGaussianLayerTorch))
        self.assertEqual(len(l_marg.scopes_out), 1)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])

    def test_layer_dist(self):

        mean_values = torch.tensor([0.73, -0.29, 0.5])
        std_values = torch.tensor([0.9, 1.34, 0.98])
        l = CondGaussianLayer(scope=Scope([1], [0]), n_nodes=3)

        # ----- full dist -----
        dist = l.dist(mean_values, std_values)

        for mean_value, std_value, mean_dist, std_dist in zip(mean_values, std_values, dist.loc, dist.scale):
            self.assertTrue(torch.allclose(mean_value, mean_dist))
            self.assertTrue(torch.allclose(std_value, std_dist))

        # ----- partial dist -----
        dist = l.dist(mean_values, std_values, [1, 2])

        for mean_value, std_value, mean_dist, std_dist in zip(mean_values[1:], std_values[1:], dist.loc, dist.scale):
            self.assertTrue(torch.allclose(mean_value, mean_dist))
            self.assertTrue(torch.allclose(std_value, std_dist))

        dist = l.dist(mean_values, std_values, [1, 0])

        for mean_value, std_value, mean_dist, std_dist in zip(
            reversed(mean_values[:-1]),
            reversed(std_values[:-1]),
            dist.loc,
            dist.scale,
        ):
            self.assertTrue(torch.allclose(mean_value, mean_dist))
            self.assertTrue(torch.allclose(std_value, std_dist))

    def test_layer_backend_conversion_1(self):

        torch_layer = CondGaussianLayer(scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])])
        base_layer = toBase(torch_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)

    def test_layer_backend_conversion_2(self):

        base_layer = BaseCondGaussianLayer(scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])])
        torch_layer = toTorch(base_layer)

        self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
        self.assertEqual(base_layer.n_out, torch_layer.n_out)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
