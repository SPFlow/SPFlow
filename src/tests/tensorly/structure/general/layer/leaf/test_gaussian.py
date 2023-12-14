import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.node.leaf.gaussian import Gaussian as GaussianBase
from spflow.base.structure.general.layer.leaf.gaussian import GaussianLayer as GaussianLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.torch.structure.general.node.leaf.gaussian import Gaussian as GaussianTorch
from spflow.torch.structure.general.layer.leaf.gaussian import GaussianLayer as GaussianLayerTorch
from spflow.torch.structure.general.layer.leaf.gaussian import updateBackend

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.general.layer.leaf.general_gaussian import GaussianLayer
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_initialization(do_for_all_backends):

    # ----- check attributes after correct initialization -----
    mean_values = [0.5, -2.3, 1.0]
    std_values = [1.3, 1.0, 0.2]
    l = GaussianLayer(scope=Scope([1]), n_nodes=3, mean=mean_values, std=std_values)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
    # make sure parameter properties works correctly
    for mean_layer_node, std_layer_node, mean_value, std_value in zip(l.mean, l.std, mean_values, std_values):
        tc.assertTrue(np.allclose(tl_toNumpy(mean_layer_node), mean_value))
        tc.assertTrue(np.allclose(tl_toNumpy(std_layer_node), std_value))

    # ----- float/int parameter values -----
    mean_value = 0.73
    std_value = 1.9
    l = GaussianLayer(scope=Scope([1]), n_nodes=3, mean=mean_value, std=std_value)

    for mean_layer_node, std_layer_node in zip(l.mean, l.std):
        tc.assertTrue(np.allclose(tl_toNumpy(mean_layer_node), mean_value))
        tc.assertTrue(np.allclose(tl_toNumpy(std_layer_node), std_value))

    # ----- list parameter values -----
    mean_values = [0.17, -0.8, 0.53]
    std_values = [0.9, 1.34, 0.98]
    l = GaussianLayer(scope=Scope([1]), n_nodes=3, mean=mean_values, std=std_values)

    for mean_layer_node, std_layer_node, mean_value, std_value in zip(l.mean, l.std, mean_values, std_values):
        tc.assertTrue(np.allclose(tl_toNumpy(mean_layer_node), mean_value))
        tc.assertTrue(np.allclose(tl_toNumpy(std_layer_node), std_value))

    # wrong number of values
    tc.assertRaises(
        ValueError,
        GaussianLayer,
        Scope([0]),
        mean_values,
        std_values[:-1],
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        GaussianLayer,
        Scope([0]),
        mean_values[:-1],
        std_values,
        n_nodes=3,
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        GaussianLayer,
        Scope([0]),
        mean_values,
        [std_values for _ in range(3)],
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        GaussianLayer,
        Scope([0]),
        [mean_values for _ in range(3)],
        std_values,
        n_nodes=3,
    )

    # ----- numpy parameter values -----

    l = GaussianLayer(
        scope=Scope([1]),
        n_nodes=3,
        mean=np.array(mean_values),
        std=np.array(std_values),
    )

    for mean_layer_node, std_layer_node, mean_value, std_value in zip(l.mean, l.std, mean_values, std_values):
        tc.assertTrue(np.allclose(tl_toNumpy(mean_layer_node), mean_value))
        tc.assertTrue(np.allclose(tl_toNumpy(std_layer_node), std_value))

    # wrong number of values
    tc.assertRaises(
        ValueError,
        GaussianLayer,
        Scope([0]),
        np.array(mean_values[:-1]),
        np.array(std_values),
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        GaussianLayer,
        Scope([0]),
        np.array(mean_values),
        np.array(std_values[:-1]),
        n_nodes=3,
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        GaussianLayer,
        Scope([0]),
        np.array(mean_values),
        np.array([std_values for _ in range(3)]),
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        GaussianLayer,
        Scope([0]),
        np.array([mean_values for _ in range(3)]),
        np.array(std_values),
        n_nodes=3,
    )

    # ---- different scopes -----
    l = GaussianLayer(scope=Scope([1]), n_nodes=3)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, GaussianLayer, Scope([0]), n_nodes=0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, GaussianLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, GaussianLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1]), Scope([0]), Scope([0])]
    l = GaussianLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

def test_accept(do_for_all_backends):

    # continuous meta type
    tc.assertTrue(
        GaussianLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # feature type class
    tc.assertTrue(
        GaussianLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Gaussian]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # feature type instance
    tc.assertTrue(
        GaussianLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Gaussian(0.0, 1.0)]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        GaussianLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # conditional scope
    tc.assertFalse(GaussianLayer.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

    # multivariate signature
    tc.assertFalse(
        GaussianLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    gaussian = GaussianLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
            FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
        ]
    )
    tc.assertTrue(gaussian.scopes_out == [Scope([0]), Scope([1])])

    gaussian = GaussianLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Gaussian]),
            FeatureContext(Scope([1]), [FeatureTypes.Gaussian]),
        ]
    )
    tc.assertTrue(gaussian.scopes_out == [Scope([0]), Scope([1])])

    gaussian = GaussianLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Gaussian(0.0, 1.0)]),
            FeatureContext(Scope([1]), [FeatureTypes.Gaussian(0.0, 1.0)]),
        ]
    )
    tc.assertTrue(gaussian.scopes_out == [Scope([0]), Scope([1])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        GaussianLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        GaussianLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        GaussianLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        GaussianInstLayer = GaussianLayerBase
    elif tl.get_backend() == "pytorch":
        GaussianInstLayer = GaussianLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(GaussianLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        GaussianLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Gaussian]),
                FeatureContext(Scope([1]), [FeatureTypes.Gaussian]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    gaussian = AutoLeaf(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Gaussian(mean=-1.0, std=1.5)]),
            FeatureContext(Scope([1]), [FeatureTypes.Gaussian(mean=1.0, std=0.5)]),
        ]
    )
    tc.assertTrue(isinstance(gaussian, GaussianInstLayer))
    tc.assertTrue(gaussian.scopes_out == [Scope([0]), Scope([1])])

def test_layer_structural_marginalization(do_for_all_backends):

    # ---------- same scopes -----------

    if tl.get_backend() == "numpy":
        GaussianInst = GaussianBase
        GaussianInstLayer = GaussianLayerBase
    elif tl.get_backend() == "pytorch":
        GaussianInst = GaussianTorch
        GaussianInstLayer = GaussianLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    l = GaussianLayer(scope=Scope([1]), mean=[0.73, -0.29], std=[0.41, 1.9], n_nodes=2)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.mean), tl_toNumpy(l_marg.mean)))
    tc.assertTrue(np.allclose(tl_toNumpy(l.std), tl_toNumpy(l_marg.std)))

    # ---------- different scopes -----------

    l = GaussianLayer(scope=[Scope([1]), Scope([0])], mean=[0.73, -0.29], std=[0.41, 1.9])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----np
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, GaussianInst))
    tc.assertEqual(l_marg.scope, Scope([0]))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.mean), tl.tensor(-0.29)))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.std), tl.tensor(1.9)))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, GaussianInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.mean), tl.tensor(-0.29)))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.std), tl.tensor(1.9)))

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.mean), tl_toNumpy(l_marg.mean)))
    tc.assertTrue(np.allclose(tl_toNumpy(l.std), tl_toNumpy(l_marg.std)))

def test_layer_dist(do_for_all_backends):

    mean_values = [0.73, -0.29, 0.5]
    std_values = [0.9, 1.34, 0.98]
    l = GaussianLayer(scope=Scope([1]), mean=mean_values, std=std_values, n_nodes=3)

    # ----- full dist -----
    dist = l.dist()

    if tl.get_backend() == "numpy":
        mean_list = [d.kwds.get("loc") for d in dist]
        std_list = [d.kwds.get("scale") for d in dist]
    elif tl.get_backend() == "pytorch":
        mean_list = dist.loc
        std_list = dist.scale
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for mean_value, std_value, mean_dist, std_dist in zip(mean_values, std_values, mean_list, std_list):
        tc.assertTrue(np.allclose(mean_value, tl_toNumpy(mean_dist)))
        tc.assertTrue(np.allclose(std_value, tl_toNumpy(std_dist)))

    # ----- partial dist -----
    dist = l.dist([1, 2])

    if tl.get_backend() == "numpy":
        mean_list = [d.kwds.get("loc") for d in dist]
        std_list = [d.kwds.get("scale") for d in dist]
    elif tl.get_backend() == "pytorch":
        mean_list = dist.loc
        std_list = dist.scale
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for mean_value, std_value, mean_dist, std_dist in zip(mean_values[1:], std_values[1:], mean_list, std_list):
        tc.assertTrue(np.allclose(mean_value, tl_toNumpy(mean_dist)))
        tc.assertTrue(np.allclose(std_value, tl_toNumpy(std_dist)))

    dist = l.dist([1, 0])

    if tl.get_backend() == "numpy":
        mean_list = [d.kwds.get("loc") for d in dist]
        std_list = [d.kwds.get("scale") for d in dist]
    elif tl.get_backend() == "pytorch":
        mean_list = dist.loc
        std_list = dist.scale
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for mean_value, std_value, mean_dist, std_dist in zip(
        reversed(mean_values[:-1]),
        reversed(std_values[:-1]),
        mean_list,
        std_list,
    ):
        tc.assertTrue(np.allclose(mean_value, tl_toNumpy(mean_dist)))
        tc.assertTrue(np.allclose(std_value, tl_toNumpy(std_dist)))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    gaussian = GaussianLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
        mean=[0.2, 0.9, 0.31],
        std=[1.9, 0.3, 0.71])
    for backend in backends:
        with tl.backend_context(backend):
            gaussian_updated = updateBackend(gaussian)
            tc.assertTrue(np.all(gaussian.scopes_out == gaussian_updated.scopes_out))
            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*gaussian.get_params()[0]]),
                    np.array([*gaussian_updated.get_params()[0]]),
                )
            )

            tc.assertTrue(
                np.allclose(
                    np.array([*gaussian.get_params()[1]]),
                    np.array([*gaussian_updated.get_params()[1]]),
                )
            )

def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    gaussian_default = GaussianLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
                             mean=[0.2, 0.9, 0.31],
                             std=[1.9, 0.3, 0.71])
    tc.assertTrue(gaussian_default.dtype == tl.float32)
    tc.assertTrue(gaussian_default.mean.dtype == tl.float32)
    tc.assertTrue(gaussian_default.std.dtype == tl.float32)

    # change to float64 model
    gaussian_updated = GaussianLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
                             mean=[0.2, 0.9, 0.31],
                             std=[1.9, 0.3, 0.71])
    gaussian_updated.to_dtype(tl.float64)
    tc.assertTrue(gaussian_updated.dtype == tl.float64)
    tc.assertTrue(gaussian_updated.mean.dtype == tl.float64)
    tc.assertTrue(gaussian_updated.std.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array([*gaussian_default.get_params()]),
            np.array([*gaussian_updated.get_params()]),
        )
    )

def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    torch.set_default_dtype(torch.float32)
    gaussian_default = GaussianLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
                             mean=[0.2, 0.9, 0.31],
                             std=[1.9, 0.3, 0.71])
    gaussian_updated = GaussianLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
                             mean=[0.2, 0.9, 0.31],
                             std=[1.9, 0.3, 0.71])
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, gaussian_updated.to_device, cuda)
        return

    # put model on gpu
    gaussian_updated.to_device(cuda)

    tc.assertTrue(gaussian_default.device.type == "cpu")
    tc.assertTrue(gaussian_updated.device.type == "cuda")

    tc.assertTrue(gaussian_default.mean.device.type == "cpu")
    tc.assertTrue(gaussian_updated.mean.device.type == "cuda")
    tc.assertTrue(gaussian_default.std.device.type == "cpu")
    tc.assertTrue(gaussian_updated.std.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array([*gaussian_default.get_params()]),
            np.array([*gaussian_updated.get_params()]),
        )
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
