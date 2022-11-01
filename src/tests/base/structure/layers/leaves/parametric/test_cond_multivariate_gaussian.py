from spflow.base.structure.layers.leaves.parametric.cond_multivariate_gaussian import (
    CondMultivariateGaussianLayer,
    marginalize,
)
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.leaves.parametric.cond_multivariate_gaussian import (
    CondMultivariateGaussian,
)
from spflow.base.structure.nodes.leaves.parametric.cond_gaussian import (
    CondGaussian,
)
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
import numpy as np
import unittest


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = CondMultivariateGaussianLayer(scope=Scope([1, 0], [2]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(
            np.all(
                l.scopes_out
                == [Scope([1, 0], [2]), Scope([1, 0], [2]), Scope([1, 0], [2])]
            )
        )

        # ---- different scopes -----
        l = CondMultivariateGaussianLayer(
            scope=Scope([0, 1, 2], [3]), n_nodes=3
        )
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(
            ValueError,
            CondMultivariateGaussianLayer,
            Scope([0, 1, 2], [3]),
            n_nodes=0,
        )

        # ----- invalid scope -----
        self.assertRaises(
            ValueError, CondMultivariateGaussianLayer, Scope([]), n_nodes=3
        )
        self.assertRaises(
            ValueError, CondMultivariateGaussianLayer, [], n_nodes=3
        )

        # ----- individual scopes and parameters -----
        scopes = [
            Scope([1, 2, 3], [5]),
            Scope([0, 1, 4], [5]),
            Scope([0, 2, 3], [5]),
        ]
        l = CondMultivariateGaussianLayer(scope=scopes, n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

        # -----number of cond_f functions -----
        CondMultivariateGaussianLayer(
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[
                lambda data: {"mean": [0.0], "cov": [[1.0]]},
                lambda data: {"mean": [0.0], "cov": [[1.0]]},
            ],
        )
        self.assertRaises(
            ValueError,
            CondMultivariateGaussianLayer,
            Scope([0], [1]),
            n_nodes=2,
            cond_f=[lambda data: {"mean": [0.0], "cov": [[1.0]]}],
        )

    def test_retrieve_params(self):

        # ----- single mean/cov list parameter values -----
        mean_value = [0.0, -1.0, 2.3]
        cov_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        l = CondMultivariateGaussianLayer(
            scope=Scope([1, 0, 2], [3]),
            n_nodes=3,
            cond_f=lambda data: {"mean": mean_value, "cov": cov_value},
        )

        for mean_node, cov_node in zip(
            *l.retrieve_params(np.array([[1.0]]), DispatchContext())
        ):
            self.assertTrue(np.all(mean_node == np.array(mean_value)))
            self.assertTrue(np.all(mean_node == np.array(mean_value)))

        # ----- multiple mean/cov list parameter values -----
        mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
        cov_values = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
            [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]],
        ]
        l.set_cond_f(lambda data: {"mean": mean_values, "cov": cov_values})

        for mean_actual, cov_actual, mean_node, cov_node in zip(
            mean_values,
            cov_values,
            *l.retrieve_params(np.array([[1.0]]), DispatchContext())
        ):
            self.assertTrue(np.all(mean_node == np.array(mean_actual)))
            self.assertTrue(np.all(cov_node == np.array(cov_actual)))

        # wrong number of values
        l.set_cond_f(lambda data: {"mean": mean_values[:-1], "cov": cov_values})
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        l.set_cond_f(lambda data: {"mean": mean_values, "cov": cov_values[:-1]})
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        # wrong number of dimensions (nested list)
        l.set_cond_f(
            lambda data: {
                "mean": [mean_values for _ in range(3)],
                "cov": cov_values,
            }
        )
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        l.set_cond_f(
            lambda data: {
                "mean": mean_values,
                "cov": [cov_values for _ in range(3)],
            }
        )
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        # ----- numpy parameter values -----
        l.set_cond_f(
            lambda data: {
                "mean": np.array(mean_values),
                "cov": np.array(cov_values),
            }
        )
        for mean_actual, cov_actual, mean_node, cov_node in zip(
            mean_values,
            cov_values,
            *l.retrieve_params(np.array([[1.0]]), DispatchContext())
        ):
            self.assertTrue(np.all(mean_node == np.array(mean_actual)))
            self.assertTrue(np.all(cov_node == np.array(cov_actual)))

        # wrong number of values
        l.set_cond_f(
            lambda data: {
                "mean": np.array(mean_values[:-1]),
                "cov": np.array(cov_values),
            }
        )
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        l.set_cond_f(
            lambda data: {
                "mean": np.array(mean_values),
                "cov": np.array(cov_values[:-1]),
            }
        )
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        # wrong number of dimensions (nested list)
        l.set_cond_f(
            lambda data: {
                "mean": np.array([mean_values for _ in range(3)]),
                "cov": np.array(cov_value),
            }
        )
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

        l.set_cond_f(
            lambda data: {
                "mean": np.array(mean_values),
                "cov": np.array([cov_values for _ in range(3)]),
            }
        )
        self.assertRaises(
            ValueError, l.retrieve_params, np.array([[1]]), DispatchContext()
        )

    def test_accept(self):

        # continuous meta types
        self.assertTrue(
            CondMultivariateGaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [3]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    ),
                    FeatureContext(
                        Scope([1, 2], [3]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    ),
                ]
            )
        )

        # Gaussian feature type class
        self.assertTrue(
            CondMultivariateGaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [3]),
                        [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                    ),
                    FeatureContext(
                        Scope([1, 2], [3]),
                        [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                    ),
                ]
            )
        )

        # Gaussian feature type instance
        self.assertTrue(
            CondMultivariateGaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [3]),
                        [
                            FeatureTypes.Gaussian(0.0, 1.0),
                            FeatureTypes.Gaussian(0.0, 1.0),
                        ],
                    ),
                    FeatureContext(
                        Scope([1, 2], [3]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    ),
                ]
            )
        )

        # continuous meta and Gaussian feature types
        self.assertTrue(
            CondMultivariateGaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Gaussian],
                    )
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondMultivariateGaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Discrete, FeatureTypes.Continuous],
                    )
                ]
            )
        )

        # non-conditional scope
        self.assertFalse(
            CondMultivariateGaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        multivariate_gaussian = CondMultivariateGaussianLayer.from_signatures(
            [
                FeatureContext(
                    Scope([0, 1], [3]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                ),
                FeatureContext(
                    Scope([1, 2], [3]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                ),
            ]
        )
        self.assertTrue(
            multivariate_gaussian.scopes_out
            == [Scope([0, 1], [3]), Scope([1, 2], [3])]
        )

        multivariate_gaussian = CondMultivariateGaussianLayer.from_signatures(
            [
                FeatureContext(
                    Scope([0, 1], [3]),
                    [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                ),
                FeatureContext(
                    Scope([1, 2], [3]),
                    [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                ),
            ]
        )
        self.assertTrue(
            multivariate_gaussian.scopes_out
            == [Scope([0, 1], [3]), Scope([1, 2], [3])]
        )

        multivariate_gaussian = CondMultivariateGaussianLayer.from_signatures(
            [
                FeatureContext(
                    Scope([0, 1], [3]),
                    [
                        FeatureTypes.Gaussian(-1.0, 1.5),
                        FeatureTypes.Gaussian(1.0, 0.5),
                    ],
                ),
                FeatureContext(
                    Scope([1, 2], [3]),
                    [
                        FeatureTypes.Gaussian(1.0, 0.5),
                        FeatureTypes.Gaussian(-1.0, 1.5),
                    ],
                ),
            ]
        )
        self.assertTrue(
            multivariate_gaussian.scopes_out
            == [Scope([0, 1], [3]), Scope([1, 2], [3])]
        )

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondMultivariateGaussianLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Discrete, FeatureTypes.Continuous],
                )
            ],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondMultivariateGaussianLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondMultivariateGaussianLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondMultivariateGaussianLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(
                        Scope([0, 1], [3]),
                        [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                    ),
                    FeatureContext(
                        Scope([1, 2], [3]),
                        [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                    ),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        multivariate_gaussian = AutoLeaf(
            [
                FeatureContext(
                    Scope([0, 1], [3]),
                    [
                        FeatureTypes.Gaussian(mean=-1.0, std=1.5),
                        FeatureTypes.Gaussian(mean=1.0, std=0.5),
                    ],
                ),
                FeatureContext(
                    Scope([1, 2], [3]),
                    [
                        FeatureTypes.Gaussian(1.0, 0.5),
                        FeatureTypes.Gaussian(-1.0, 1.5),
                    ],
                ),
            ]
        )
        self.assertTrue(
            multivariate_gaussian.scopes_out
            == [Scope([0, 1], [3]), Scope([1, 2], [3])]
        )

    def test_layer_structural_marginalization(self):

        # ---------- same scopes -----------

        l = CondMultivariateGaussianLayer(
            scope=[Scope([0, 1], [2]), Scope([0, 1], [2])]
        )

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1]) == None)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])

        self.assertTrue(
            l_marg.scopes_out == [Scope([0, 1], [2]), Scope([0, 1], [2])]
        )

        # ---------- different scopes -----------

        l = CondMultivariateGaussianLayer(
            scope=[Scope([0, 2], [4]), Scope([1, 3], [4])]
        )

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0, 1, 2, 3]) == None)

        # ----- partially marginalize -----
        l_marg = marginalize(l, [0, 2], prune=True)
        self.assertTrue(isinstance(l_marg, CondMultivariateGaussian))
        self.assertEqual(l_marg.scope, Scope([1, 3], [4]))

        l_marg = marginalize(l, [0, 1, 2], prune=True)
        self.assertTrue(isinstance(l_marg, CondGaussian))
        self.assertEqual(l_marg.scope, Scope([3], [4]))

        l_marg = marginalize(l, [0, 2], prune=False)
        self.assertTrue(isinstance(l_marg, CondMultivariateGaussianLayer))
        self.assertEqual(l_marg.scopes_out, [Scope([1, 3], [4])])
        self.assertEqual(len(l_marg.nodes), 1)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [4])

        self.assertTrue(
            l_marg.scopes_out == [Scope([0, 2], [4]), Scope([1, 3], [4])]
        )


if __name__ == "__main__":
    unittest.main()
