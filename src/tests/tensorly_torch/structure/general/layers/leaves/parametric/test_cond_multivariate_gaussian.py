import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.tensorly.structure import AutoLeaf
from spflow.base.structure.general.nodes.leaves.parametric.cond_multivariate_gaussian import CondMultivariateGaussian as CondMultivariateGaussianBase
from spflow.base.structure.general.layers.leaves.parametric.cond_multivariate_gaussian import CondMultivariateGaussianLayer as CondMultivariateGaussianLayerBase
from spflow.base.structure.general.nodes.leaves.parametric.cond_gaussian import CondGaussian as CondGaussianBase

from spflow.torch.structure.general.nodes.leaves.parametric.cond_multivariate_gaussian import CondMultivariateGaussian as CondMultivariateGaussianTorch
from spflow.torch.structure.general.layers.leaves.parametric.cond_multivariate_gaussian import CondMultivariateGaussianLayer as CondMultivariateGaussianLayerTorch
from spflow.torch.structure.general.nodes.leaves.parametric.cond_gaussian import CondGaussian as CondGaussianTorch

from spflow.tensorly.structure.general.layers.leaves.parametric.general_cond_multivariate_gaussian import CondMultivariateGaussianLayer
from spflow.torch.structure.general.layers.leaves.parametric.cond_multivariate_gaussian import updateBackend

from spflow.torch.structure import marginalize
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext

tc = unittest.TestCase()

def test_layer_initialization(do_for_all_backends):

    # ----- check attributes after correct initialization -----

    l = CondMultivariateGaussianLayer(scope=Scope([1, 0], [2]), n_nodes=3)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1, 0], [2]), Scope([1, 0], [2]), Scope([1, 0], [2])]))

    # ---- different scopes -----
    l = CondMultivariateGaussianLayer(
        scope=[Scope([0, 1, 2], [4]), Scope([1, 3], [4]), Scope([2], [4])],
        n_nodes=3,
    )
    for node, node_scope in zip(l.nodes, l.scopes_out):
        tc.assertEqual(node.scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(
        ValueError,
        CondMultivariateGaussianLayer,
        Scope([0, 1, 2], [3]),
        n_nodes=0,
    )

    # ----- invalid scope -----
    tc.assertRaises(ValueError, CondMultivariateGaussianLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, CondMultivariateGaussianLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [
        Scope([1, 2, 3], [5]),
        Scope([0, 1, 4], [5]),
        Scope([0, 2, 3], [5]),
    ]
    l = CondMultivariateGaussianLayer(scope=scopes, n_nodes=3)
    for node, node_scope in zip(l.nodes, scopes):
        tc.assertEqual(node.scope, node_scope)

    # -----number of cond_f functions -----
    CondMultivariateGaussianLayer(
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[
            lambda data: {"mean": [0.0], "cov": [[1.0]]},
            lambda data: {"mean": [0.0], "cov": [[1.0]]},
        ],
    )
    tc.assertRaises(
        ValueError,
        CondMultivariateGaussianLayer,
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[lambda data: {"mean": [0.0], "cov": [[1.0]]}],
    )

def test_retrieve_params(do_for_all_backends):

    # ----- single mean/cov list parameter values -----
    mean_value = [0.0, -1.0, 2.3]
    cov_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    l = CondMultivariateGaussianLayer(
        scope=Scope([1, 0, 2], [3]),
        n_nodes=3,
        cond_f=lambda data: {"mean": mean_value, "cov": cov_value},
    )

    for mean_node, cov_node in zip(*l.retrieve_params(tl.tensor([[1]]), DispatchContext())):
        tc.assertTrue(np.allclose(np.array(mean_node, dtype=np.float64), np.array(mean_value)))
        tc.assertTrue(np.allclose(np.array(cov_node, dtype=np.float64), np.array(cov_value)))
    # ----- multiple mean/cov list parameter values -----
    mean_values = [[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]]
    cov_values = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
        [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]],
    ]
    l.set_cond_f(lambda data: {"mean": mean_values, "cov": cov_values})

    for mean_actual, cov_actual, mean_node, cov_node in zip(
        mean_values, cov_values, *l.retrieve_params(tl.tensor([[1]]), DispatchContext())
    ):
        tc.assertTrue(np.allclose(np.array(mean_node, dtype=np.float64), np.array(mean_actual)))
        tc.assertTrue(np.allclose(np.array(cov_node, dtype=np.float64), np.array(cov_actual)))

    # wrong number of values
    l.set_cond_f(lambda data: {"mean": mean_values[:-1], "cov": cov_values})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    l.set_cond_f(lambda data: {"mean": mean_values, "cov": cov_values[:-1]})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    # wrong number of dimensions (nested list)
    l.set_cond_f(
        lambda data: {
            "mean": [mean_values for _ in range(3)],
            "cov": cov_values,
        }
    )
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    l.set_cond_f(
        lambda data: {
            "mean": mean_values,
            "cov": [cov_values for _ in range(3)],
        }
    )
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    # ----- numpy parameter values -----
    l.set_cond_f(
        lambda data: {
            "mean": np.array(mean_values),
            "cov": np.array(cov_values),
        }
    )
    for mean_actual, cov_actual, mean_node, cov_node in zip(
        mean_values, cov_values, *l.retrieve_params(tl.tensor([[1.0]]), DispatchContext())
    ):
        tc.assertTrue(np.allclose(np.array(mean_node, dtype=np.float64), np.array(mean_actual)))
        tc.assertTrue(np.allclose(np.array(cov_node, dtype=np.float64), np.array(cov_actual)))

    # wrong number of values
    l.set_cond_f(
        lambda data: {
            "mean": np.array(mean_values[:-1]),
            "cov": np.array(cov_values),
        }
    )
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    l.set_cond_f(
        lambda data: {
            "mean": np.array(mean_values),
            "cov": np.array(cov_values[:-1]),
        }
    )
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    # wrong number of dimensions (nested list)
    l.set_cond_f(
        lambda data: {
            "mean": np.array([mean_values for _ in range(3)]),
            "cov": np.array(cov_value),
        }
    )
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    l.set_cond_f(
        lambda data: {
            "mean": np.array(mean_values),
            "cov": np.array([cov_values for _ in range(3)]),
        }
    )
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

def test_accept(do_for_all_backends):

    # continuous meta types
    tc.assertTrue(
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
    tc.assertTrue(
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
    tc.assertTrue(
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
    tc.assertTrue(
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
    tc.assertFalse(
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
    tc.assertFalse(
        CondMultivariateGaussianLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

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
    tc.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1], [3]), Scope([1, 2], [3])])

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
    tc.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1], [3]), Scope([1, 2], [3])])

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
    tc.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1], [3]), Scope([1, 2], [3])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
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
    tc.assertRaises(
        ValueError,
        CondMultivariateGaussianLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondMultivariateGaussianLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
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
    tc.assertTrue(multivariate_gaussian.scopes_out == [Scope([0, 1], [3]), Scope([1, 2], [3])])

def test_layer_structural_marginalization(do_for_all_backends):

    # ---------- same scopes -----------

    if tl.get_backend() == "numpy":
        CondMultivariateGaussianInst = CondMultivariateGaussianBase
        CondMultivariateGaussianInstLayer = CondMultivariateGaussianLayerBase
        CondGaussianInst = CondGaussianBase
    elif tl.get_backend() == "pytorch":
        CondMultivariateGaussianInst = CondMultivariateGaussianTorch
        CondMultivariateGaussianInstLayer = CondMultivariateGaussianLayerTorch
        CondGaussianInst = CondGaussianTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")


    l = CondMultivariateGaussianLayer(scope=[Scope([0, 1], [2]), Scope([0, 1], [2])])
    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([0, 1], [2]), Scope([0, 1], [2])])

    # ---------- different scopes -----------

    l = CondMultivariateGaussianLayer(scope=[Scope([0, 2], [4]), Scope([1, 3], [4])])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1, 2, 3]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [0, 2], prune=True)
    tc.assertTrue(isinstance(l_marg, CondMultivariateGaussianInst))
    tc.assertEqual(l_marg.scope, Scope([1, 3], [4]))

    l_marg = marginalize(l, [0, 1, 2], prune=True)
    tc.assertTrue(isinstance(l_marg, CondGaussianInst))
    tc.assertEqual(l_marg.scope, Scope([3], [4]))

    l_marg = marginalize(l, [0, 2], prune=False)
    tc.assertTrue(isinstance(l_marg, CondMultivariateGaussianInstLayer))
    tc.assertEqual(l_marg.scopes_out, [Scope([1, 3], [4])])
    tc.assertEqual(len(l_marg.nodes), 1)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [4])

    tc.assertTrue(l_marg.scopes_out == [Scope([0, 2], [4]), Scope([1, 3], [4])])

def test_layer_dist(do_for_all_backends):

    mean_values = tl.tensor([[0.0, -1.0, 2.3], [1.0, 5.0, -3.0], [-7.1, 3.2, -0.9]])
    cov_values = tl.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.5, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 0.7]],
            [[3.1, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.3]],
        ]
    )
    l = CondMultivariateGaussianLayer(scope=Scope([0, 1, 2], [3]), n_nodes=3)

    # ----- full dist -----
    dist_list = l.dist(mean_values, cov_values)

    for mean_value, cov_value, dist in zip(mean_values, cov_values, dist_list):
        if tl.get_backend() == "numpy":
            mean_list = dist.mean
            cov_list = dist.cov_object.covariance
        elif tl.get_backend() == "pytorch":
            mean_list = dist.mean
            cov_list = dist.covariance_matrix
        else:
            raise NotImplementedError("This test is not implemented for this backend")
        tc.assertTrue(np.allclose(mean_value, mean_list))
        tc.assertTrue(np.allclose(cov_value, cov_list))

    # ----- partial dist -----
    dist_list = l.dist(mean_values, cov_values, [1, 2])

    for mean_value, cov_value, dist in zip(mean_values[1:], cov_values[1:], dist_list):
        if tl.get_backend() == "numpy":
            mean_list = dist.mean
            cov_list = dist.cov_object.covariance
        elif tl.get_backend() == "pytorch":
            mean_list = dist.mean
            cov_list = dist.covariance_matrix
        else:
            raise NotImplementedError("This test is not implemented for this backend")

        tc.assertTrue(np.allclose(mean_value, mean_list))
        tc.assertTrue(np.allclose(cov_value, cov_list))

    dist_list = l.dist(mean_values, cov_values, [1, 0])

    for mean_value, cov_value, dist in zip(reversed(mean_values[:-1]), reversed(cov_values[:-1]), dist_list):
        if tl.get_backend() == "numpy":
            mean_list = dist.mean
            cov_list = dist.cov_object.covariance
        elif tl.get_backend() == "pytorch":
            mean_list = dist.mean
            cov_list = dist.covariance_matrix
        else:
            raise NotImplementedError("This test is not implemented for this backend")

        tc.assertTrue(np.allclose(mean_value, mean_list))
        tc.assertTrue(np.allclose(cov_value, cov_list))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    mutlivariateGaussian = CondMultivariateGaussianLayer(scope=[
            Scope([0, 1, 2], [4]),
            Scope([1, 2, 3], [4]),
            Scope([0, 1, 2], [4]),
        ])
    for backend in backends:
        with tl.backend_context(backend):
            mutlivariateGaussian_updated = updateBackend(mutlivariateGaussian)
            tc.assertTrue(np.all(mutlivariateGaussian.scopes_out == mutlivariateGaussian_updated.scopes_out))
            # check conversion from torch to python

def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    mean_value = [0.0, -1.0, 2.3]
    cov_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    cond_mv_gaussian_default = CondMultivariateGaussianLayer(
        scope=Scope([1, 0, 2], [3]),
        n_nodes=3,
        cond_f=lambda data: {"mean": mean_value, "cov": cov_value},
    )

    cond_mv_gaussian_update = CondMultivariateGaussianLayer(
        scope=Scope([1, 0, 2], [3]),
        n_nodes=3,
        cond_f=lambda data: {"mean": mean_value, "cov": cov_value},
    )
    tc.assertTrue(cond_mv_gaussian_default.dtype == tl.float32)
    params = cond_mv_gaussian_default.retrieve_params(np.array([[1.0]]), DispatchContext())
    mean = params[0]
    cov = params[1]
    tc.assertTrue(mean[0].dtype == tl.float32)
    tc.assertTrue(cov[0].dtype == tl.float32)

    # change to float64 model
    cond_mv_gaussian_update.to_dtype(tl.float64)
    params_up = cond_mv_gaussian_update.retrieve_params(np.array([[1.0]]), DispatchContext())
    mean_up = params_up[0]
    cov_up = params_up[1]

    tc.assertTrue(cond_mv_gaussian_update.dtype == tl.float64)
    tc.assertTrue(mean_up[0].dtype == tl.float64)
    tc.assertTrue(cov_up[0].dtype == tl.float64)
    for m, m_up in zip(mean, mean_up):
        tc.assertTrue(
            np.allclose(
                m,
                m_up,
            )
        )

    for c, c_up in zip(cov, cov_up):
        tc.assertTrue(
            np.allclose(
                c,
                c_up,
            )
        )



def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    torch.set_default_dtype(torch.float32)
    mean_value = [0.0, -1.0, 2.3]
    cov_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    cond_mv_gaussian_default = CondMultivariateGaussianLayer(
        scope=Scope([1, 0, 2], [3]),
        n_nodes=3,
        cond_f=lambda data: {"mean": mean_value, "cov": cov_value},
    )

    cond_mv_gaussian_update = CondMultivariateGaussianLayer(
        scope=Scope([1, 0, 2], [3]),
        n_nodes=3,
        cond_f=lambda data: {"mean": mean_value, "cov": cov_value},
    )
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, cond_mv_gaussian_update.to_device, cuda)
        return

    # put model on gpu
    cond_mv_gaussian_update.to_device(cuda)

    tc.assertTrue(cond_mv_gaussian_default.device.type == "cpu")
    tc.assertTrue(cond_mv_gaussian_update.device.type == "cuda")

    params = cond_mv_gaussian_default.retrieve_params(np.array([[1.0]]), DispatchContext())
    mean = params[0]
    cov = params[1]
    params_up = cond_mv_gaussian_update.retrieve_params(np.array([[1.0]]), DispatchContext())
    mean_up = params_up[0]
    cov_up = params_up[1]


    tc.assertTrue(mean[0].device.type == "cpu")
    tc.assertTrue(mean_up[0].device.type == "cuda")
    tc.assertTrue(cov[0].device.type == "cpu")
    tc.assertTrue(cov_up[0].device.type == "cuda")

    for m, m_up in zip(mean, mean_up):
        tc.assertTrue(
            np.allclose(
                m,
                m_up.cpu(),
            )
        )

    for c, c_up in zip(cov, cov_up):
        tc.assertTrue(
            np.allclose(
                c,
                c_up.cpu(),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
