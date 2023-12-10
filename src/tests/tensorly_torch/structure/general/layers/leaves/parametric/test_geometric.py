import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.nodes.leaves.parametric.geometric import Geometric as GeometricBase
from spflow.base.structure.general.layers.leaves.parametric.geometric import GeometricLayer as GeometricLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.torch.structure.general.nodes.leaves.parametric.geometric import Geometric as GeometricTorch
from spflow.torch.structure.general.layers.leaves.parametric.geometric import GeometricLayer as GeometricLayerTorch
from spflow.torch.structure.general.layers.leaves.parametric.geometric import updateBackend

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.general.layers.leaves.parametric.general_geometric import GeometricLayer
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()


def test_layer_initialization(do_for_all_backends):

    # ----- check attributes after correct initialization -----
    p_values = [0.5, 0.3, 0.9]
    l = GeometricLayer(scope=Scope([1]), n_nodes=3, p=p_values)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
    # make sure parameter properties works correctly
    for p_layer_node, p_value in zip(l.p, p_values):
        tc.assertTrue(np.allclose(tl_toNumpy(p_layer_node), p_value))

    # ----- float/int parameter values -----
    p_value = 0.73
    l = GeometricLayer(scope=Scope([1]), n_nodes=3, p=p_value)

    for p_layer_node in l.p:
        tc.assertTrue(np.allclose(tl_toNumpy(p_layer_node), p_value))

    # ----- list parameter values -----
    p_values = [0.17, 0.8, 0.53]
    l = GeometricLayer(scope=Scope([1]), n_nodes=3, p=p_values)

    for p_layer_node, p_value in zip(l.p, p_values):
        tc.assertTrue(np.allclose(tl_toNumpy(p_layer_node), p_value))

    # wrong number of values
    tc.assertRaises(ValueError, GeometricLayer, Scope([0]), p_values[:-1], n_nodes=3)
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        GeometricLayer,
        Scope([0]),
        [p_values for _ in range(3)],
        n_nodes=3,
    )

    # ----- numpy parameter values -----

    p = GeometricLayer(scope=Scope([1]), n_nodes=3, p=np.array(p_values))

    for p_layer_node, p_value in zip(l.p, p_values):
        tc.assertTrue(np.allclose(tl_toNumpy(p_layer_node), p_value))

    # wrong number of values
    tc.assertRaises(
        ValueError,
        GeometricLayer,
        Scope([0]),
        np.array(p_values[:-1]),
        n_nodes=3,
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        GeometricLayer,
        Scope([0]),
        np.array([p_values for _ in range(3)]),
        n_nodes=3,
    )

    # ---- different scopes -----
    l = GeometricLayer(scope=Scope([1]), n_nodes=3)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, GeometricLayer, Scope([0]), n_nodes=0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, GeometricLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, GeometricLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1]), Scope([0]), Scope([0])]
    l = GeometricLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

def test_accept(do_for_all_backends):

    # discrete meta type
    tc.assertTrue(
        GeometricLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
            ]
        )
    )

    # feature type instance
    tc.assertTrue(
        GeometricLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Geometric]),
                FeatureContext(Scope([1]), [FeatureTypes.Geometric]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        GeometricLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1]), [FeatureTypes.Geometric]),
            ]
        )
    )

    # conditional scope
    tc.assertFalse(GeometricLayer.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Geometric])]))

    # multivariate signature
    tc.assertFalse(
        GeometricLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Geometric, FeatureTypes.Geometric],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    geometric = GeometricLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
            FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
        ]
    )
    tc.assertTrue(geometric.scopes_out == [Scope([0]), Scope([1])])

    geometric = GeometricLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Geometric]),
            FeatureContext(Scope([1]), [FeatureTypes.Geometric]),
        ]
    )
    tc.assertTrue(geometric.scopes_out == [Scope([0]), Scope([1])])

    geometric = GeometricLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Geometric(0.5)]),
            FeatureContext(Scope([1]), [FeatureTypes.Geometric(0.5)]),
        ]
    )
    tc.assertTrue(geometric.scopes_out == [Scope([0]), Scope([1])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        GeometricLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        GeometricLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        GeometricLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Discrete, FeatureTypes.Discrete],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(GeometricLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        GeometricLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Geometric]),
                FeatureContext(Scope([1]), [FeatureTypes.Geometric]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    geometric = AutoLeaf(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Geometric(p=0.75)]),
            FeatureContext(Scope([1]), [FeatureTypes.Geometric(p=0.25)]),
        ]
    )
    tc.assertTrue(geometric.scopes_out == [Scope([0]), Scope([1])])

def test_layer_structural_marginalization(do_for_all_backends):

    if tl.get_backend() == "numpy":
        GeometricInst = GeometricBase
        GeometricInstLayer = GeometricLayerBase
    elif tl.get_backend() == "pytorch":
        GeometricInst = GeometricTorch
        GeometricInstLayer = GeometricLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # ---------- same scopes -----------

    l = GeometricLayer(scope=Scope([1]), p=[0.73, 0.29], n_nodes=2)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.p), tl_toNumpy(l_marg.p)))

    # ---------- different scopes -----------

    l = GeometricLayer(scope=[Scope([1]), Scope([0])], p=[0.73, 0.29])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, GeometricInst))
    tc.assertEqual(l_marg.scope, Scope([0]))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.p), tl.tensor(0.29)))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, GeometricInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.p), tl.tensor(0.29)))

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.p), tl_toNumpy(l_marg.p)))

def test_layer_dist(do_for_all_backends):

    p_values = [0.73, 0.29, 0.5]
    l = GeometricLayer(scope=Scope([1]), p=p_values, n_nodes=3)

    # ----- full dist -----
    dist = l.dist()

    if tl.get_backend() == "numpy":
        p_list = [d.kwds.get("p") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for p_value, p_dist in zip(p_values, p_list):
        tc.assertTrue(np.allclose(p_value, tl_toNumpy(p_dist)))

    # ----- partial dist -----
    dist = l.dist([1, 2])

    if tl.get_backend() == "numpy":
        p_list = [d.kwds.get("p") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for p_value, p_dist in zip(p_values[1:], p_list):
        tc.assertTrue(np.allclose(p_value, tl_toNumpy(p_dist)))

    dist = l.dist([1, 0])

    if tl.get_backend() == "numpy":
        p_list = [d.kwds.get("p") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for p_value, p_dist in zip(reversed(p_values[:-1]), p_list):
        tc.assertTrue(np.allclose(p_value, tl_toNumpy(p_dist)))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    geometric = GeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
    for backend in backends:
        with tl.backend_context(backend):
            geometric_updated = updateBackend(geometric)
            tc.assertTrue(np.all(geometric.scopes_out == geometric_updated.scopes_out))
            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*geometric.get_params()[0]]),
                    np.array([*geometric_updated.get_params()[0]]),
                )
            )

def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    geometric_default = GeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
    tc.assertTrue(geometric_default.dtype == tl.float32)
    tc.assertTrue(geometric_default.p.dtype == tl.float32)

    # change to float64 model
    geometric_updated = GeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
    geometric_updated.to_dtype(tl.float64)
    tc.assertTrue(geometric_updated.dtype == tl.float64)
    tc.assertTrue(geometric_updated.p.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array([*geometric_default.get_params()]),
            np.array([*geometric_updated.get_params()]),
        )
    )

def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    torch.set_default_dtype(torch.float32)
    geometric_default = GeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
    geometric_updated = GeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, geometric_updated.to_device, cuda)
        return

    # put model on gpu
    geometric_updated.to_device(cuda)

    tc.assertTrue(geometric_default.device.type == "cpu")
    tc.assertTrue(geometric_updated.device.type == "cuda")

    tc.assertTrue(geometric_default.p.device.type == "cpu")
    tc.assertTrue(geometric_updated.p.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array([*geometric_default.get_params()]),
            np.array([*geometric_updated.get_params()]),
        )
    )




if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
