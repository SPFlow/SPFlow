import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.node.leaf.exponential import Exponential as ExponentialBase
from spflow.base.structure.general.layer.leaf.exponential import ExponentialLayer as ExponentialLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import AutoLeaf, marginalize, toBase, toTorch
from spflow.torch.structure.general.node.leaf.exponential import Exponential as ExponentialTorch
from spflow.torch.structure.general.layer.leaf.exponential import ExponentialLayer as ExponentialLayerTorch
from spflow.torch.structure.general.layer.leaf.bernoulli import updateBackend

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.general.layer.leaf.general_exponential import ExponentialLayer
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()


def test_layer_initialization(do_for_all_backends):

    # ----- check attributes after correct initialization -----
    l_values = [0.5, 2.3, 1.0]
    l = ExponentialLayer(scope=Scope([1]), n_nodes=3, l=l_values)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
    # make sure parameter properties works correctly
    for l_layer_node, l_value in zip(l.l, l_values):
        tc.assertTrue(np.allclose(tl_toNumpy(l_layer_node), l_value))

    # ----- float/int parameter values -----
    l_value = 0.73
    l = ExponentialLayer(scope=Scope([1]), n_nodes=3, l=l_value)

    for l_layer_node in l.l:
        tc.assertTrue(np.allclose(tl_toNumpy(l_layer_node), l_value))

    # ----- list parameter values -----
    l_values = [0.17, 0.8, 0.53]
    l = ExponentialLayer(scope=Scope([1]), n_nodes=3, l=l_values)

    for l_layer_node, l_value in zip(l.l, l_values):
        tc.assertTrue(np.allclose(tl_toNumpy(l_layer_node), l_value))

    # wrong number of values
    tc.assertRaises(ValueError, ExponentialLayer, Scope([0]), l_values[:-1], n_nodes=3)
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        ExponentialLayer,
        Scope([0]),
        [l_values for _ in range(3)],
        n_nodes=3,
    )

    # ----- numpy parameter values -----

    l = ExponentialLayer(scope=Scope([1]), n_nodes=3, l=np.array(l_values))

    for l_layer_node, l_value in zip(l.l, l_values):
        tc.assertTrue(np.allclose(tl_toNumpy(l_layer_node), l_value))

    # wrong number of values
    tc.assertRaises(
        ValueError,
        ExponentialLayer,
        Scope([0]),
        np.array(l_values[:-1]),
        n_nodes=3,
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        ExponentialLayer,
        Scope([0]),
        np.array([l_values for _ in range(3)]),
        n_nodes=3,
    )

    # ---- different scopes -----
    l = ExponentialLayer(scope=Scope([1]), n_nodes=3)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, ExponentialLayer, Scope([0]), n_nodes=0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, ExponentialLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, ExponentialLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1]), Scope([0]), Scope([0])]
    l = ExponentialLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

def test_accept(do_for_all_backends):

    # continuous meta type
    tc.assertTrue(
        ExponentialLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # Exponential feature type class
    tc.assertTrue(
        ExponentialLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Exponential]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # Exponential feature type instance
    tc.assertTrue(
        ExponentialLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Exponential(1.0)]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        ExponentialLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # conditional scope
    tc.assertFalse(ExponentialLayer.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

    # multivariate signature
    tc.assertFalse(
        ExponentialLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    exponential = ExponentialLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
            FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
        ]
    )
    tc.assertTrue(np.allclose(tl_toNumpy(exponential.l), np.array([1.0, 1.0])))
    tc.assertTrue(exponential.scopes_out == [Scope([0]), Scope([1])])

    exponential = ExponentialLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Exponential]),
            FeatureContext(Scope([1]), [FeatureTypes.Exponential]),
        ]
    )
    tc.assertTrue(np.allclose(tl_toNumpy(exponential.l), np.array([1.0, 1.0])))
    tc.assertTrue(exponential.scopes_out == [Scope([0]), Scope([1])])

    exponential = ExponentialLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Exponential(l=1.5)]),
            FeatureContext(Scope([1]), [FeatureTypes.Exponential(l=0.5)]),
        ]
    )
    tc.assertTrue(np.allclose(tl_toNumpy(exponential.l), np.array([1.5, 0.5])))
    tc.assertTrue(exponential.scopes_out == [Scope([0]), Scope([1])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        ExponentialLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        ExponentialLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        ExponentialLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(ExponentialLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        ExponentialLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Exponential]),
                FeatureContext(Scope([1]), [FeatureTypes.Exponential]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    exponential = AutoLeaf(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Exponential(l=1.5)]),
            FeatureContext(Scope([1]), [FeatureTypes.Exponential(l=0.5)]),
        ]
    )
    tc.assertTrue(np.allclose(tl_toNumpy(exponential.l), np.array([1.5, 0.5])))
    tc.assertTrue(exponential.scopes_out == [Scope([0]), Scope([1])])

def test_layer_structural_marginalization(do_for_all_backends):

    if tl.get_backend() == "numpy":
        ExponentialInst = ExponentialBase
        ExponentialInstLayer = ExponentialLayerBase
    elif tl.get_backend() == "pytorch":
        ExponentialInst = ExponentialTorch
        ExponentialInstLayer = ExponentialLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # ---------- same scopes -----------

    l = ExponentialLayer(scope=Scope([1]), l=[0.73, 0.29], n_nodes=2)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.l), tl_toNumpy(l_marg.l)))

    # ---------- different scopes -----------

    l = ExponentialLayer(scope=[Scope([1]), Scope([0])], l=[0.73, 0.29])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, ExponentialInst))
    tc.assertEqual(l_marg.scope, Scope([0]))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.l), np.array(0.29)))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, ExponentialInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.l), np.array(0.29)))

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.l), tl_toNumpy(l_marg.l)))

def test_layer_dist(do_for_all_backends):

    l_values = [0.73, 0.29, 0.5]
    l = ExponentialLayer(scope=Scope([1]), l=l_values, n_nodes=3)

    # ----- full dist -----
    dist = l.dist()

    if tl.get_backend() == "numpy":
        l_list = [1 / d.kwds.get("scale") for d in dist]
    elif tl.get_backend() == "pytorch":
        l_list = dist.rate
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for l_value, l_dist in zip(l_values, l_list):
        tc.assertTrue(np.allclose(l_value, tl_toNumpy(l_dist)))

    # ----- partial dist -----
    dist = l.dist([1, 2])

    if tl.get_backend() == "numpy":
        l_list = [1 / d.kwds.get("scale") for d in dist]
    elif tl.get_backend() == "pytorch":
        l_list = dist.rate
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for l_value, l_dist in zip(l_values[1:], l_list):
        tc.assertTrue(np.allclose(l_value, tl_toNumpy(l_dist)))

    dist = l.dist([1, 0])

    if tl.get_backend() == "numpy":
        l_list = [1 / d.kwds.get("scale") for d in dist]
    elif tl.get_backend() == "pytorch":
        l_list = dist.rate
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for l_value, l_dist in zip(reversed(l_values[:-1]), l_list):
        tc.assertTrue(np.allclose(l_value, tl_toNumpy(l_dist)))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    exponential = ExponentialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 0.9, 0.31])
    for backend in backends:
        with tl.backend_context(backend):
            exponential_updated = updateBackend(exponential)
            tc.assertTrue(np.all(exponential.scopes_out == exponential_updated.scopes_out))
            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*exponential.get_params()[0]]),
                    np.array([*exponential_updated.get_params()[0]]),
                )
            )

def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    exponential_default = ExponentialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 0.9, 0.31])
    tc.assertTrue(exponential_default.dtype == tl.float32)
    tc.assertTrue(exponential_default.l.dtype == tl.float32)

    # change to float64 model
    exponential_updated = ExponentialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 0.9, 0.31])
    exponential_updated.to_dtype(tl.float64)
    tc.assertTrue(exponential_updated.dtype == tl.float64)
    tc.assertTrue(exponential_updated.l.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array([*exponential_default.get_params()]),
            np.array([*exponential_updated.get_params()]),
        )
    )

def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    torch.set_default_dtype(torch.float32)
    exponential_default = ExponentialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 0.9, 0.31])
    exponential_updated = ExponentialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 0.9, 0.31])
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, exponential_updated.to_device, cuda)
        return

    # put model on gpu
    exponential_updated.to_device(cuda)

    tc.assertTrue(exponential_default.device.type == "cpu")
    tc.assertTrue(exponential_updated.device.type == "cuda")

    tc.assertTrue(exponential_default.l.device.type == "cpu")
    tc.assertTrue(exponential_updated.l.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array([*exponential_default.get_params()]),
            np.array([*exponential_updated.get_params()]),
        )
    )



if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
