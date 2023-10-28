import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian as MultivariateGaussianBase
from spflow.base.structure.general.layers.leaves.parametric.multivariate_gaussian import MultivariateGaussianLayer as MultivariateGaussianLayerBase
from spflow.base.structure.general.nodes.leaves.parametric.gaussian import Gaussian as GaussianBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.tensorly.structure import marginalize
from spflow.torch.structure.general.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian as MultivariateGaussianTorch
from spflow.torch.structure.general.layers.leaves.parametric.multivariate_gaussian import MultivariateGaussianLayer as MultivariateGaussianLayerTorch
from spflow.torch.structure.general.nodes.leaves.parametric.gaussian import Gaussian as GaussianTorch
from spflow.torch.structure.general.layers.leaves.parametric.multivariate_gaussian import updateBackend

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.general.layers.leaves.parametric.general_multivariate_gaussian import MultivariateGaussianLayer
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()


def test_layer_initialization(do_for_all_backends):
    torch.set_default_dtype(torch.float64)

    # ----- check attributes after correct initialization -----

    l = MultivariateGaussianLayer(scope=Scope([1, 0]), n_nodes=3)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1, 0]), Scope([1, 0]), Scope([1, 0])]))
    mean_values = l.mean
    cov_values = l.cov
    # make sure parameter properties works correctly
    for node, node_mean, node_cov in zip(l.nodes, mean_values, cov_values):
        tc.assertTrue(tl.all(node.mean == node_mean))
        tc.assertTrue(tl.all(node.cov == node_cov))

    # ----- single mean/cov list parameter values -----
    mean_value = [0.0, -1.0, 2.3]
    cov_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    l = MultivariateGaussianLayer(scope=Scope([1, 0, 2]), n_nodes=3, mean=mean_value, cov=cov_value)

    for node in l.nodes:
        tc.assertTrue(tl.all(node.mean == tl.tensor(mean_value, dtype=tl.float64)))
        tc.assertTrue(tl.all(node.cov == tl.tensor(cov_value, dtype=tl.float64)))

    # ----- multiple mean/cov list parameter values -----
    mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
    cov_values = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
        [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]],
    ]
    l = MultivariateGaussianLayer(scope=Scope([0, 1, 2]), n_nodes=3, mean=mean_values, cov=cov_values)

    for node, node_mean, node_cov in zip(l.nodes, mean_values, cov_values):
        tc.assertTrue(tl.all(node.mean == tl.tensor(node_mean, dtype=tl.float64)))
        tc.assertTrue(np.allclose(tl_toNumpy(node.cov), tl.tensor(node_cov, dtype=tl.float64)))

    # wrong number of values
    tc.assertRaises(
        ValueError,
        MultivariateGaussianLayer,
        Scope([0, 1, 2]),
        mean_values[:-1],
        cov_values,
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        MultivariateGaussianLayer,
        Scope([0, 1, 2]),
        mean_values,
        cov_values[:-1],
        n_nodes=3,
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        MultivariateGaussianLayer,
        Scope([0, 1, 2]),
        mean_values,
        [cov_values for _ in range(3)],
        n_nodes=3,
    )
    tc.assertRaises(
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
        tc.assertTrue(tl.all(node.mean == tl.tensor(node_mean, dtype=tl.float64)))
        tc.assertTrue(np.allclose(tl_toNumpy(node.cov), tl.tensor(node_cov, dtype=tl.float64)))

    # wrong number of values
    tc.assertRaises(
        ValueError,
        MultivariateGaussianLayer,
        Scope([0, 1, 2]),
        np.array(mean_values[:-1]),
        np.array(cov_values),
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        MultivariateGaussianLayer,
        Scope([0, 1, 2]),
        np.array(mean_values),
        np.array(cov_values[:-1]),
        n_nodes=3,
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        MultivariateGaussianLayer,
        Scope([0, 1, 2]),
        mean_values,
        np.array([cov_values for _ in range(3)]),
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        MultivariateGaussianLayer,
        Scope([0, 1, 2]),
        np.array([mean_values for _ in range(3)]),
        cov_values,
        n_nodes=3,
    )

    # ---- different scopes -----
    l = MultivariateGaussianLayer(scope=[Scope([0, 1, 2]), Scope([1, 3]), Scope([2])], n_nodes=3)
    for node, node_scope in zip(l.nodes, l.scopes_out):
        tc.assertEqual(node.scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, MultivariateGaussianLayer, Scope([0, 1, 2]), n_nodes=0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, MultivariateGaussianLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, MultivariateGaussianLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1, 2, 3]), Scope([0, 1, 4]), Scope([0, 2, 3])]
    l = MultivariateGaussianLayer(scope=scopes, n_nodes=3)
    for node, node_scope in zip(l.nodes, scopes):
        tc.assertEqual(node.scope, node_scope)

def test_accept(do_for_all_backends):

    # continuous meta types
    tc.assertTrue(
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
    tc.assertTrue(
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
    tc.assertTrue(
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
    tc.assertTrue(
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
    tc.assertFalse(
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
    tc.assertFalse(
        MultivariateGaussianLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

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
    tc.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1]), Scope([1, 2])])

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
    tc.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1]), Scope([1, 2])])

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
    tc.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1]), Scope([1, 2])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
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
    tc.assertRaises(
        ValueError,
        MultivariateGaussianLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(MultivariateGaussianLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
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
    tc.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1]), Scope([1, 2])])

def test_layer_structural_marginalization(do_for_all_backends):
    torch.set_default_dtype(torch.float64)

    if tl.get_backend() == "numpy":
        MultivariateGaussianInst = MultivariateGaussianBase
        MultivariateGaussianInstLayer = MultivariateGaussianLayerBase
        GaussianInst = GaussianBase
    elif tl.get_backend() == "pytorch":
        MultivariateGaussianInst = MultivariateGaussianTorch
        MultivariateGaussianInstLayer = MultivariateGaussianLayerTorch
        GaussianInst = GaussianTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # ---------- same scopes -----------

    l = MultivariateGaussianLayer(
        scope=[Scope([0, 1]), Scope([0, 1])],
        mean=[[-0.2, 1.3], [3.7, -0.9]],
        cov=[[[1.3, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 0.7]]],
    )
    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([0, 1]), Scope([0, 1])])
    tc.assertTrue(all([tl.all(m1 == m2) for m1, m2 in zip(l.mean, l_marg.mean)]))
    tc.assertTrue(all([tl.all(c1 == c2) for c1, c2 in zip(l.cov, l_marg.cov)]))

    # ---------- different scopes -----------

    l = MultivariateGaussianLayer(
        scope=[Scope([0, 2]), Scope([1, 3])],
        mean=[[-0.2, 1.3], [3.7, -0.9]],
        cov=[[[1.3, 0.0], [0.0, 1.1]], [[0.5, 0.0], [0.0, 0.7]]],
    )

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1, 2, 3]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [0, 2], prune=True)
    tc.assertTrue(isinstance(l_marg, MultivariateGaussianInst))
    tc.assertEqual(l_marg.scope, Scope([1, 3]))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.mean).astype(np.float64), np.array([3.7, -0.9])))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.cov).astype(np.float64), np.array([[0.5, 0.0], [0.0, 0.7]])))

    l_marg = marginalize(l, [0, 1, 2], prune=True)
    tc.assertTrue(isinstance(l_marg, GaussianInst))
    tc.assertEqual(l_marg.scope, Scope([3]))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.mean).astype(np.float64), np.array(-0.9)))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.std).astype(np.float64), np.array(np.sqrt(0.7))))

    l_marg = marginalize(l, [0, 2], prune=False)
    tc.assertTrue(isinstance(l_marg, MultivariateGaussianInstLayer))
    tc.assertEqual(l_marg.scopes_out, [Scope([1, 3])])
    tc.assertEqual(len(l_marg.nodes), 1)
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.mean[0]).astype(np.float64), np.array([3.7, -0.9])))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.cov[0]).astype(np.float64), np.array([[0.5, 0.0], [0.0, 0.7]])))

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [4])

    tc.assertTrue(l_marg.scopes_out == [Scope([0, 2]), Scope([1, 3])])
    tc.assertTrue(all([tl.all(m1 == m2) for m1, m2 in zip(l.mean, l_marg.mean)]))
    tc.assertTrue(all([tl.all(c1 == c2) for c1, c2 in zip(l.cov, l_marg.cov)]))

def test_layer_dist(do_for_all_backends):

    mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
    cov_values = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
        [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]],
    ]
    l = MultivariateGaussianLayer(scope=Scope([0, 1, 2]), mean=mean_values, cov=cov_values, n_nodes=3)

    # ----- full dist -----
    dist_list = l.dist()

    for mean_value, cov_value, dist in zip(mean_values, cov_values, dist_list):
        if tl.get_backend() == "numpy":
            mean_list = dist.mean
            cov_list = dist.cov_object.covariance
        elif tl.get_backend() == "pytorch":
            mean_list = dist.mean
            cov_list = dist.covariance_matrix
        else:
            raise NotImplementedError("This test is not implemented for this backend")
        tc.assertTrue(np.allclose(tl.tensor(mean_value), tl_toNumpy(mean_list)))
        tc.assertTrue(np.allclose(tl.tensor(cov_value), tl_toNumpy(cov_list)))

    # ----- partial dist -----
    dist_list = l.dist([1, 2])

    for mean_value, cov_value, dist in zip(mean_values[1:], cov_values[1:], dist_list):
        if tl.get_backend() == "numpy":
            mean_list = dist.mean
            cov_list = dist.cov_object.covariance
        elif tl.get_backend() == "pytorch":
            mean_list = dist.mean
            cov_list = dist.covariance_matrix
        else:
            raise NotImplementedError("This test is not implemented for this backend")
        tc.assertTrue(np.allclose(tl.tensor(mean_value), tl_toNumpy(mean_list)))
        tc.assertTrue(np.allclose(tl.tensor(cov_value), tl_toNumpy(cov_list)))

    dist_list = l.dist([1, 0])

    for mean_value, cov_value, dist in zip(reversed(mean_values[:-1]), reversed(cov_values[:-1]), dist_list):
        if tl.get_backend() == "numpy":
            mean_list = dist.mean
            cov_list = dist.cov_object.covariance
        elif tl.get_backend() == "pytorch":
            mean_list = dist.mean
            cov_list = dist.covariance_matrix
        else:
            raise NotImplementedError("This test is not implemented for this backend")
        tc.assertTrue(np.allclose(tl.tensor(mean_value), tl_toNumpy(mean_list)))
        tc.assertTrue(np.allclose(tl.tensor(cov_value), tl_toNumpy(cov_list)))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
    cov_values = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
        [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]],
    ]
    multivariateGaussian = MultivariateGaussianLayer(scope=[Scope([0, 1, 2]), Scope([1, 2, 3]), Scope([0, 1, 2])],
        mean=mean_values,
        cov=cov_values)
    for backend in backends:
        with tl.backend_context(backend):
            multivariateGaussian_updated = updateBackend(multivariateGaussian)
            tc.assertTrue(np.all(multivariateGaussian.scopes_out == multivariateGaussian_updated.scopes_out))
            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*multivariateGaussian.get_params()[0]]),
                    np.array([*multivariateGaussian_updated.get_params()[0]]),
                )
            )

            tc.assertTrue(
                np.allclose(
                    np.array([*multivariateGaussian.get_params()[1]]),
                    np.array([*multivariateGaussian_updated.get_params()[1]]),
                )
            )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
