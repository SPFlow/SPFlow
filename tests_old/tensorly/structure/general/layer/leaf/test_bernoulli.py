import unittest
from typing import Callable

import numpy as np
import torch
import tensorly as tl
from pytest import fixture
import pytest
from spflow.base.structure.general.node.leaf.bernoulli import Bernoulli as BernoulliBase
from spflow.base.structure.general.layer.leaf.bernoulli import BernoulliLayer as BernoulliLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.base.structure import marginalize
from spflow.torch.structure.general.node.leaf.bernoulli import Bernoulli as BernoulliTorch
from spflow.torch.structure.general.layer.leaf.bernoulli import BernoulliLayer as BernoulliLayerTorch
from spflow.torch.structure.general.layer.leaf.bernoulli import updateBackend
from spflow.structure import AutoLeaf
from spflow.structure.spn.layer.leaf import BernoulliLayer
from spflow.utils import Tensor
from spflow.tensor import ops as tle
import unittest


tc = unittest.TestCase()


def test_layer_initialization(do_for_all_backends):
    # ----- check attributes after correct initialization -----
    p_values = [0.3, 0.7, 0.5]
    l = BernoulliLayer(scope=Scope([1]), n_nodes=3, p=p_values)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
    # make sure parameter properties works correctly
    for p_layer_node, p_value in zip(l.p, p_values):
        tc.assertTrue(np.allclose(tle.toNumpy(p_layer_node), np.array(p_value)))

    # ----- float/int parameter values -----
    p_value = 0.13
    l = BernoulliLayer(scope=Scope([1]), n_nodes=3, p=p_value)

    for p_layer_node in l.p:
        tc.assertTrue(np.allclose(tle.toNumpy(p_layer_node), p_value))

    # ----- list parameter values -----
    p_values = [0.17, 0.8, 0.53]
    l = BernoulliLayer(scope=Scope([1]), n_nodes=3, p=p_values)

    for p_layer_node, p_value in zip(l.p, p_values):
        tc.assertTrue(np.allclose(tle.toNumpy(p_layer_node), np.array(p_value)))

    # wrong number of values
    tc.assertRaises(ValueError, BernoulliLayer, Scope([0]), p_values[:-1], n_nodes=3)
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        BernoulliLayer,
        Scope([0]),
        [p_values for _ in range(3)],
        n_nodes=3,
    )

    # ----- numpy parameter values -----

    l = BernoulliLayer(scope=Scope([1]), n_nodes=3, p=np.array(p_values))

    for p_layer_node, p_value in zip(l.p, p_values):
        tc.assertTrue(np.allclose(tle.toNumpy(p_layer_node), np.array(p_value)))

    # wrong number of values
    tc.assertRaises(
        ValueError,
        BernoulliLayer,
        Scope([0]),
        np.array(p_values[:-1]),
        n_nodes=3,
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        BernoulliLayer,
        Scope([0]),
        np.array([p_values for _ in range(3)]),
        n_nodes=3,
    )

    # ---- different scopes -----
    l = BernoulliLayer(scope=Scope([1]), n_nodes=3)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, BernoulliLayer, Scope([0]), n_nodes=0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, BernoulliLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, BernoulliLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1]), Scope([0]), Scope([0])]
    l = BernoulliLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)


def test_accept(do_for_all_backends):
    # discrete meta type
    tc.assertTrue(
        BernoulliLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
            ]
        )
    )

    # feature type class
    tc.assertTrue(
        BernoulliLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Bernoulli]),
                FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
            ]
        )
    )

    # feature type instance
    tc.assertTrue(
        BernoulliLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(0.5)]),
                FeatureContext(Scope([1]), [FeatureTypes.Bernoulli(0.5)]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        BernoulliLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # conditional scope
    tc.assertFalse(BernoulliLayer.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

    # multivariate signature
    tc.assertFalse(
        BernoulliLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ]
        )
    )


def test_initialization_from_signatures(do_for_all_backends):
    bernoulli = BernoulliLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
            FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
        ]
    )
    tc.assertTrue(np.allclose(tle.toNumpy(bernoulli.p), np.array([0.5, 0.5])))
    tc.assertTrue(bernoulli.scopes_out == [Scope([0]), Scope([1])])

    bernoulli = BernoulliLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Bernoulli]),
            FeatureContext(Scope([1]), [FeatureTypes.Bernoulli]),
        ]
    )
    tc.assertTrue(np.allclose(tle.toNumpy(bernoulli.p), np.array([0.5, 0.5])))
    tc.assertTrue(bernoulli.scopes_out == [Scope([0]), Scope([1])])

    bernoulli = BernoulliLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(p=0.75)]),
            FeatureContext(Scope([1]), [FeatureTypes.Bernoulli(p=0.25)]),
        ]
    )
    tc.assertTrue(np.allclose(tle.toNumpy(bernoulli.p), np.array([0.75, 0.25])))
    tc.assertTrue(bernoulli.scopes_out == [Scope([0]), Scope([1])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        BernoulliLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        BernoulliLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        BernoulliLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Discrete, FeatureTypes.Discrete],
            )
        ],
    )


def test_autoleaf(do_for_all_backends):
    tc.assertTrue(AutoLeaf.is_registered(BernoulliLayer))

    feature_ctx_1 = FeatureContext(Scope([0]), [FeatureTypes.Bernoulli(p=0.75)])
    feature_ctx_2 = FeatureContext(Scope([1]), [FeatureTypes.Bernoulli(p=0.25)])

    # make sure leaf is correctly inferred
    tc.assertEqual(BernoulliLayer, AutoLeaf.infer([feature_ctx_1, feature_ctx_2]))

    # make sure AutoLeaf can return correctly instantiated object
    bernoulli = AutoLeaf([feature_ctx_1, feature_ctx_2])
    tc.assertTrue(np.allclose(tle.toNumpy(bernoulli.p), np.array([0.75, 0.25])))
    tc.assertTrue(bernoulli.scopes_out == [Scope([0]), Scope([1])])


def test_layer_structural_marginalization(do_for_all_backends):
    # ---------- same scopes -----------

    l = BernoulliLayer(scope=Scope([1]), p=[0.73, 0.29], n_nodes=2)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
    tc.assertTrue(np.allclose(tle.toNumpy(l.p), tle.toNumpy(l_marg.p)))

    # ---------- different scopes -----------

    l = BernoulliLayer(scope=[Scope([1]), Scope([0])], p=[0.73, 0.29])

    if tl.get_backend() == "numpy":
        BernoulliInst = BernoulliBase
        BernoulliInstLayer = BernoulliLayerBase
    elif tl.get_backend() == "pytorch":
        BernoulliInst = BernoulliTorch
        BernoulliInstLayer = BernoulliLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, BernoulliInst))
    tc.assertEqual(l_marg.scope, Scope([0]))
    tc.assertTrue(np.allclose(tle.toNumpy(l_marg.p), np.array(0.29)))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, BernoulliInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)
    tc.assertTrue(np.allclose(tle.toNumpy(l_marg.p), np.array(0.29)))

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
    tc.assertTrue(np.allclose(tle.toNumpy(l.p), tle.toNumpy(l_marg.p)))


def test_layer_dist(do_for_all_backends):
    p_values = [0.73, 0.29, 0.5]
    l = BernoulliLayer(scope=Scope([1]), p=p_values, n_nodes=3)
    dist = l.dist()
    # ----- full dist -----

    if tl.get_backend() == "numpy":
        p_list = [d.kwds.get("p") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for p_value, p_dist in zip(p_values, p_list):
        tc.assertTrue(np.allclose(np.array(p_value), tle.toNumpy(p_dist)))

    # ----- partial dist -----
    dist = l.dist([1, 2])

    if tl.get_backend() == "numpy":
        p_list = [d.kwds.get("p") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for p_value, p_dist in zip(p_values[1:], p_list):
        tc.assertTrue(np.allclose(np.array(p_value), tle.toNumpy(p_dist)))

    dist = l.dist([1, 0])

    if tl.get_backend() == "numpy":
        p_list = [d.kwds.get("p") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for p_value, p_dist in zip(reversed(p_values[:-1]), p_list):
        tc.assertTrue(np.allclose(np.array(p_value), tle.toNumpy(p_dist)))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    bernoulli = BernoulliLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
    for backend in backends:
        with tl.backend_context(backend):
            bernoulli_updated = updateBackend(bernoulli)
            tc.assertTrue(np.all(bernoulli.scopes_out == bernoulli_updated.scopes_out))
            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*bernoulli.get_params()]),
                    np.array([*bernoulli_updated.get_params()]),
                )
            )


def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    bernoulli_default = BernoulliLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
    tc.assertTrue(bernoulli_default.dtype == tl.float32)
    tc.assertTrue(bernoulli_default.p.dtype == tl.float32)

    # change to float64 model
    bernoulli_updated = BernoulliLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
    bernoulli_updated.to_dtype(tl.float64)
    tc.assertTrue(bernoulli_updated.dtype == tl.float64)
    tc.assertTrue(bernoulli_updated.p.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array([*bernoulli_default.get_params()]),
            np.array([*bernoulli_updated.get_params()]),
        )
    )


def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    torch.set_default_dtype(torch.float32)
    bernoulli_default = BernoulliLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
    bernoulli_updated = BernoulliLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 0.9, 0.31])
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, bernoulli_updated.to_device, cuda)
        return

    # put model on gpu
    bernoulli_updated.to_device(cuda)

    tc.assertTrue(bernoulli_default.device.type == "cpu")
    tc.assertTrue(bernoulli_updated.device.type == "cuda")

    tc.assertTrue(bernoulli_default.p.device.type == "cpu")
    tc.assertTrue(bernoulli_updated.p.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array([*bernoulli_default.get_params()]),
            np.array([*bernoulli_updated.get_params()]),
        )
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
