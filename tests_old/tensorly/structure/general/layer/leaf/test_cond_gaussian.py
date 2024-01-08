import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.node.leaf.cond_gaussian import CondGaussian as CondGaussianBase
from spflow.base.structure.general.layer.leaf.cond_gaussian import CondGaussianLayer as CondGaussianLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure import marginalize
from spflow.torch.structure.general.node.leaf.cond_gaussian import CondGaussian as CondGaussianTorch
from spflow.torch.structure.general.layer.leaf.cond_gaussian import (
    CondGaussianLayer as CondGaussianLayerTorch,
)
from spflow.torch.structure.general.layer.leaf.cond_gaussian import updateBackend

from spflow.structure import AutoLeaf
from spflow.modules.layer import CondGaussianLayer

tc = unittest.TestCase()


def test_layer_initialization(do_for_all_backends):
    # ----- check attributes after correct initialization -----

    l = CondGaussianLayer(scope=Scope([1], [0]), n_nodes=3)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])]))

    # ---- different scopes -----
    l = CondGaussianLayer(scope=Scope([1], [0]), n_nodes=3)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, CondGaussianLayer, Scope([0], [1]), n_nodes=0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, CondGaussianLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, CondGaussianLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
    l = CondGaussianLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

    # -----number of cond_f functions -----
    CondGaussianLayer(
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[
            lambda data: {"mean": 0.0, "std": 1.0},
            lambda data: {"mean": 0.0, "std": 1.0},
        ],
    )
    tc.assertRaises(
        ValueError,
        CondGaussianLayer,
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[lambda data: {"mean": 0.0, "std": 1.0}],
    )


def test_retrieve_params(do_for_all_backends):
    # ----- float/int parameter values -----
    mean_value = 0.73
    std_value = 1.9
    l = CondGaussianLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        cond_f=lambda data: {"mean": mean_value, "std": std_value},
    )

    for mean_layer_node, std_layer_node in zip(*l.retrieve_params(tl.tensor([[1]]), DispatchContext())):
        tc.assertTrue(np.allclose(mean_layer_node, mean_value))
        tc.assertTrue(np.allclose(std_layer_node, std_value))

    # ----- list parameter values -----
    mean_values = [0.17, -0.8, 0.53]
    std_values = [0.9, 1.34, 0.98]
    l = CondGaussianLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        cond_f=lambda data: {"mean": mean_values, "std": std_values},
    )

    for mean_value, std_value, mean_layer_node, std_layer_node in zip(
        mean_values, std_values, *l.retrieve_params(tl.tensor([[1]]), DispatchContext())
    ):
        tc.assertTrue(np.allclose(mean_layer_node, mean_value))
        tc.assertTrue(np.allclose(std_layer_node, std_value))

    # wrong number of values
    l.set_cond_f(lambda data: {"mean": mean_values[:-1], "std": std_values})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )
    l.set_cond_f(lambda data: {"mean": mean_values, "std": std_values[:-1]})
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
            "std": std_values,
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
            "std": [std_values for _ in range(3)],
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
            "std": np.array(std_values),
        }
    )
    for mean_actual, std_actual, mean_node, std_node in zip(
        mean_values, std_values, *l.retrieve_params(tl.tensor([[1.0]]), DispatchContext())
    ):
        tc.assertTrue(mean_node == mean_actual)
        tc.assertTrue(std_node == std_actual)

    # wrong number of values
    l.set_cond_f(
        lambda data: {
            "mean": np.array(mean_values[:-1]),
            "std": np.array(std_values),
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
            "std": np.array(std_values[:-1]),
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
            "std": np.array(std_values),
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
            "std": np.array([std_values for _ in range(3)]),
        }
    )
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )


def test_accept(do_for_all_backends):
    # continuous meta type
    tc.assertTrue(
        CondGaussianLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # feature type class
    tc.assertTrue(
        CondGaussianLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # feature type instance
    tc.assertTrue(
        CondGaussianLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian(0.0, 1.0)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        CondGaussianLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # non-conditional scope
    tc.assertFalse(CondGaussianLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # multivariate signature
    tc.assertFalse(
        CondGaussianLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )


def test_initialization_from_signatures(do_for_all_backends):
    gaussian = CondGaussianLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
        ]
    )
    tc.assertTrue(gaussian.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    gaussian = CondGaussianLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Gaussian]),
        ]
    )
    tc.assertTrue(gaussian.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    gaussian = CondGaussianLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Gaussian(0.0, 1.0)]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Gaussian(0.0, 1.0)]),
        ]
    )
    tc.assertTrue(gaussian.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondGaussianLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondGaussianLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondGaussianLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )


def test_autoleaf(do_for_all_backends):
    if tl.get_backend() == "numpy":
        CondGaussianInstLayer = CondGaussianLayerBase
    elif tl.get_backend() == "pytorch":
        CondGaussianInstLayer = CondGaussianLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondGaussianLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
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
    tc.assertTrue(isinstance(gaussian, CondGaussianInstLayer))
    tc.assertTrue(gaussian.scopes_out == [Scope([0], [2]), Scope([1], [2])])


def test_layer_structural_marginalization(do_for_all_backends):
    if tl.get_backend() == "numpy":
        CondGaussianInst = CondGaussianBase
        CondGaussianInstLayer = CondGaussianLayerBase
    elif tl.get_backend() == "pytorch":
        CondGaussianInst = CondGaussianTorch
        CondGaussianInstLayer = CondGaussianLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # ---------- same scopes -----------

    l = CondGaussianLayer(scope=Scope([1], [0]), n_nodes=2)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

    # ---------- different scopes -----------

    l = CondGaussianLayer(scope=[Scope([1], [2]), Scope([0], [2])])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, CondGaussianInst))
    tc.assertEqual(l_marg.scope, Scope([0], [2]))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, CondGaussianInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])


def test_layer_dist(do_for_all_backends):
    mean_values = tl.tensor([0.73, -0.29, 0.5])
    std_values = tl.tensor([0.9, 1.34, 0.98])
    l = CondGaussianLayer(scope=Scope([1], [0]), n_nodes=3)

    # ----- full dist -----
    dist = l.dist(mean_values, std_values)

    if tl.get_backend() == "numpy":
        mean_list = [d.kwds.get("loc") for d in dist]
        std_list = [d.kwds.get("scale") for d in dist]
    elif tl.get_backend() == "pytorch":
        mean_list = dist.loc
        std_list = dist.scale
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for mean_value, std_value, mean_dist, std_dist in zip(mean_values, std_values, mean_list, std_list):
        tc.assertTrue(np.allclose(mean_value, mean_dist))
        tc.assertTrue(np.allclose(std_value, std_dist))

    # ----- partial dist -----
    dist = l.dist(mean_values, std_values, [1, 2])

    if tl.get_backend() == "numpy":
        mean_list = [d.kwds.get("loc") for d in dist]
        std_list = [d.kwds.get("scale") for d in dist]
    elif tl.get_backend() == "pytorch":
        mean_list = dist.loc
        std_list = dist.scale
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for mean_value, std_value, mean_dist, std_dist in zip(
        mean_values[1:], std_values[1:], mean_list, std_list
    ):
        tc.assertTrue(np.allclose(mean_value, mean_dist))
        tc.assertTrue(np.allclose(std_value, std_dist))

    dist = l.dist(mean_values, std_values, [1, 0])

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
        tc.assertTrue(np.allclose(mean_value, mean_dist))
        tc.assertTrue(np.allclose(std_value, std_dist))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    gaussian = CondGaussianLayer(scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])])
    for backend in backends:
        with tl.backend_context(backend):
            gaussian_updated = updateBackend(gaussian)
            tc.assertTrue(np.all(gaussian.scopes_out == gaussian_updated.scopes_out))
            # check conversion from torch to python


def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    mean_value = 0.73
    std_value = 1.9
    cond_gaussian_default = CondGaussianLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        cond_f=lambda data: {"mean": mean_value, "std": std_value},
    )

    cond_gaussian_updated = CondGaussianLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        cond_f=lambda data: {"mean": mean_value, "std": std_value},
    )
    tc.assertTrue(cond_gaussian_default.dtype == tl.float32)
    params = cond_gaussian_default.retrieve_params(np.array([[1.0]]), DispatchContext())
    mean = params[0]
    std = params[1]
    tc.assertTrue(mean.dtype == tl.float32)
    tc.assertTrue(std.dtype == tl.float32)

    # change to float64 model
    cond_gaussian_updated.to_dtype(tl.float64)
    params_up = cond_gaussian_updated.retrieve_params(np.array([[1.0]]), DispatchContext())
    mean_up = params_up[0]
    std_up = params_up[1]

    tc.assertTrue(cond_gaussian_updated.dtype == tl.float64)
    tc.assertTrue(mean_up.dtype == tl.float64)
    tc.assertTrue(std_up.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array(mean),
            np.array(mean_up),
        )
    )

    tc.assertTrue(
        np.allclose(
            np.array(std),
            np.array(std_up),
        )
    )


def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    torch.set_default_dtype(torch.float32)
    mean_value = 0.73
    std_value = 1.9
    cond_gaussian_default = CondGaussianLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        cond_f=lambda data: {"mean": mean_value, "std": std_value},
    )

    cond_gaussian_updated = CondGaussianLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        cond_f=lambda data: {"mean": mean_value, "std": std_value},
    )
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, cond_gaussian_updated.to_device, cuda)
        return

    # put model on gpu
    cond_gaussian_updated.to_device(cuda)

    tc.assertTrue(cond_gaussian_default.device.type == "cpu")
    tc.assertTrue(cond_gaussian_updated.device.type == "cuda")

    params = cond_gaussian_default.retrieve_params(np.array([[1.0]]), DispatchContext())
    mean = params[0]
    std = params[1]
    params_up = cond_gaussian_updated.retrieve_params(np.array([[1.0]]), DispatchContext())
    mean_up = params_up[0]
    std_up = params_up[1]

    tc.assertTrue(mean.device.type == "cpu")
    tc.assertTrue(mean_up.device.type == "cuda")
    tc.assertTrue(std.device.type == "cpu")
    tc.assertTrue(std_up.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array(mean),
            np.array(mean_up.cpu()),
        )
    )

    tc.assertTrue(
        np.allclose(
            np.array(std),
            np.array(std_up.cpu()),
        )
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()