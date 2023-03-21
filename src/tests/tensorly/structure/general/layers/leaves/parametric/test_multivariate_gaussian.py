import unittest

import tensorly as tl

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.spn import (
    Gaussian,
    MultivariateGaussian,
    MultivariateGaussianLayer,
    marginalize,
)
from spflow.meta.data import FeatureContext, FeatureTypes, Scope


class TestLayer(unittest.TestCase):
    def test_layer_initialization_1(self):

        # ----- check attributes after correct initialization -----

        l = MultivariateGaussianLayer(scope=Scope([1, 0]), n_nodes=3)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(tl.all(l.scopes_out == [Scope([1, 0]), Scope([1, 0]), Scope([1, 0])]))
        # make sure parameter properties works correctly
        mean_values = l.mean
        cov_values = l.cov
        for node, node_mean, node_cov in zip(l.nodes, mean_values, cov_values):
            self.assertTrue(tl.all(node.mean == node_mean))
            self.assertTrue(tl.all(node.cov == node_cov))

        # ----- single mean/cov list parameter values -----
        mean_value = [0.0, -1.0, 2.3]
        cov_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        l = MultivariateGaussianLayer(scope=Scope([1, 0, 2]), n_nodes=3, mean=mean_value, cov=cov_value)

        for node in l.nodes:
            self.assertTrue(tl.all(node.mean == mean_value))
            self.assertTrue(tl.all(node.cov == cov_value))

        # ----- multiple mean/cov list parameter values -----
        mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
        cov_values = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
            [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]],
        ]
        l = MultivariateGaussianLayer(scope=Scope([0, 1, 2]), n_nodes=3, mean=mean_values, cov=cov_values)

        for node, node_mean, node_cov in zip(l.nodes, mean_values, cov_values):
            self.assertTrue(tl.all(node.mean == node_mean))
            self.assertTrue(tl.all(node.cov == node_cov))

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
            mean=tl.tensor(mean_values),
            cov=tl.tensor(cov_values),
        )

        for node, node_mean, node_cov in zip(l.nodes, mean_values, cov_values):
            self.assertTrue(tl.all(node.mean == node_mean))
            self.assertTrue(tl.all(node.cov == node_cov))

        # wrong number of values
        self.assertRaises(
            ValueError,
            MultivariateGaussianLayer,
            Scope([0, 1, 2]),
            tl.tensor(mean_values[:-1]),
            tl.tensor(cov_values),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            MultivariateGaussianLayer,
            Scope([0, 1, 2]),
            tl.tensor(mean_values),
            tl.tensor(cov_values[:-1]),
            n_nodes=3,
        )
        # wrong number of dimensions (nested list)
        self.assertRaises(
            ValueError,
            MultivariateGaussianLayer,
            Scope([0, 1, 2]),
            mean_values,
            tl.tensor([cov_values for _ in range(3)]),
            n_nodes=3,
        )
        self.assertRaises(
            ValueError,
            MultivariateGaussianLayer,
            Scope([0, 1, 2]),
            tl.tensor([mean_values for _ in range(3)]),
            cov_values,
            n_nodes=3,
        )

        # ---- different scopes -----
        l = MultivariateGaussianLayer(scope=[Scope([0, 1, 2]), Scope([1, 3]), Scope([2])], n_nodes=3)
        for node, node_scope in zip(l.nodes, l.scopes_out):
            self.assertEqual(node.scope, node_scope)

        # ----- invalid number of nodes -----
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0, 1, 2]), n_nodes=0)

        # ----- invalid scope -----
        self.assertRaises(ValueError, MultivariateGaussianLayer, Scope([]), n_nodes=3)
        self.assertRaises(ValueError, MultivariateGaussianLayer, [], n_nodes=3)

        # ----- individual scopes and parameters -----
        scopes = [Scope([1, 2, 3]), Scope([0, 1, 4]), Scope([0, 2, 3])]
        l = MultivariateGaussianLayer(scope=scopes, n_nodes=3)
        for node, node_scope in zip(l.nodes, scopes):
            self.assertEqual(node.scope, node_scope)

    def test_accept(self):

        # continuous meta types
        self.assertTrue(
            MultivariateGaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    ),
                    FeatureContext(
                        Scope([1, 2]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    ),
                ]
            )
        )

        # Gaussian feature type class
        self.assertTrue(
            MultivariateGaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                    ),
                    FeatureContext(
                        Scope([1, 2]),
                        [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                    ),
                ]
            )
        )

        # Gaussian feature type instance
        self.assertTrue(
            MultivariateGaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [
                            FeatureTypes.Gaussian(0.0, 1.0),
                            FeatureTypes.Gaussian(0.0, 1.0),
                        ],
                    ),
                    FeatureContext(
                        Scope([1, 2]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    ),
                ]
            )
        )

        # continuous meta and Gaussian feature types
        self.assertTrue(
            MultivariateGaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Gaussian],
                    )
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            MultivariateGaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Discrete, FeatureTypes.Continuous],
                    )
                ]
            )
        )

        # conditional scope
        self.assertFalse(
            MultivariateGaussianLayer.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        multivariate_gaussian = MultivariateGaussianLayer.from_signatures(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                ),
                FeatureContext(
                    Scope([1, 2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                ),
            ]
        )
        self.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1]), Scope([1, 2])])

        multivariate_gaussian = MultivariateGaussianLayer.from_signatures(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                ),
                FeatureContext(
                    Scope([1, 2]),
                    [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                ),
            ]
        )
        self.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1]), Scope([1, 2])])

        multivariate_gaussian = MultivariateGaussianLayer.from_signatures(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [
                        FeatureTypes.Gaussian(-1.0, 1.5),
                        FeatureTypes.Gaussian(1.0, 0.5),
                    ],
                ),
                FeatureContext(
                    Scope([1, 2]),
                    [
                        FeatureTypes.Gaussian(1.0, 0.5),
                        FeatureTypes.Gaussian(-1.0, 1.5),
                    ],
                ),
            ]
        )
        self.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1]), Scope([1, 2])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            MultivariateGaussianLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Continuous],
                )
            ],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            MultivariateGaussianLayer.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(MultivariateGaussianLayer))

        # make sure leaf is correctly inferred
        self.assertEqual(
            MultivariateGaussianLayer,
            AutoLeaf.infer(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                    ),
                    FeatureContext(
                        Scope([1, 2]),
                        [FeatureTypes.Gaussian, FeatureTypes.Gaussian],
                    ),
                ]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        multivariate_gaussian = AutoLeaf(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [
                        FeatureTypes.Gaussian(mean=-1.0, std=1.5),
                        FeatureTypes.Gaussian(mean=1.0, std=0.5),
                    ],
                ),
                FeatureContext(
                    Scope([1, 2]),
                    [
                        FeatureTypes.Gaussian(1.0, 0.5),
                        FeatureTypes.Gaussian(-1.0, 1.5),
                    ],
                ),
            ]
        )
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
        self.assertTrue(all([tl.all(m1 == m2) for m1, m2 in zip(l.mean, l_marg.mean)]))
        self.assertTrue(all([tl.all(c1 == c2) for c1, c2 in zip(l.cov, l_marg.cov)]))

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
        self.assertTrue(tl.all(l_marg.mean == tl.tensor([3.7, -0.9])))
        self.assertTrue(tl.all(l_marg.cov == tl.tensor([[0.5, 0.0], [0.0, 0.7]])))

        l_marg = marginalize(l, [0, 1, 2], prune=True)
        self.assertTrue(isinstance(l_marg, Gaussian))
        self.assertEqual(l_marg.scope, Scope([3]))
        self.assertTrue(tl.all(l_marg.mean == tl.tensor(-0.9)))
        self.assertTrue(tl.all(l_marg.std == tl.tensor(tl.sqrt(0.7))))

        l_marg = marginalize(l, [0, 2], prune=False)
        self.assertTrue(isinstance(l_marg, MultivariateGaussianLayer))
        self.assertEqual(l_marg.scopes_out, [Scope([1, 3])])
        self.assertEqual(len(l_marg.nodes), 1)
        self.assertTrue(tl.all(l_marg.mean == tl.tensor([3.7, -0.9])))
        self.assertTrue(tl.all(l_marg.cov == tl.tensor([[0.5, 0.0], [0.0, 0.7]])))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [4])

        self.assertTrue(l_marg.scopes_out == [Scope([0, 2]), Scope([1, 3])])
        self.assertTrue(all([tl.all(m1 == m2) for m1, m2 in zip(l.mean, l_marg.mean)]))
        self.assertTrue(all([tl.all(c1 == c2) for c1, c2 in zip(l.cov, l_marg.cov)]))

    def test_get_params(self):

        layer = MultivariateGaussianLayer(
            scope=Scope([0, 1]),
            mean=[[-0.73, 0.29], [0.36, -1.4]],
            cov=[[[1.0, 0.92], [0.92, 1.2]], [[1.0, 0.3], [0.3, 1.4]]],
            n_nodes=2,
        )

        mean, cov, *others = layer.get_params()

        self.assertTrue(len(others) == 0)
        self.assertTrue(tl.allclose(mean, tl.tensor([[-0.73, 0.29], [0.36, -1.4]])))
        self.assertTrue(
            tl.allclose(
                cov,
                tl.tensor([[[1.0, 0.92], [0.92, 1.2]], [[1.0, 0.3], [0.3, 1.4]]]),
            )
        )


if __name__ == "__main__":
    unittest.main()
