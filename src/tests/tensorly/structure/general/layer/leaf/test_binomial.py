import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.node.leaf.binomial import Binomial as BinomialBase
from spflow.base.structure.general.layer.leaf.binomial import BinomialLayer as BinomialLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import marginalize
from spflow.torch.structure.general.node.leaf.binomial import Binomial as BinomialTorch
from spflow.torch.structure.general.layer.leaf.binomial import BinomialLayer as BinomialLayerTorch
from spflow.torch.structure.general.layer.leaf.binomial import updateBackend

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.general.layer.leaf.general_binomial import BinomialLayer
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_initialization(do_for_all_backends):

    # ----- check attributes after correct initialization -----
    n_values = [3, 2, 7]
    p_values = [0.3, 0.7, 0.5]
    l = BinomialLayer(scope=[Scope([1]), Scope([0]), Scope([2])], n=n_values, p=p_values)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([0]), Scope([2])]))
    # make sure parameter properties works correctly
    for n_layer_node, p_layer_node, n_value, p_value in zip(l.n, l.p, n_values, p_values):
        tc.assertTrue(np.allclose(tl_toNumpy(n_layer_node), np.array(n_value)))
        tc.assertTrue(np.allclose(tl_toNumpy(p_layer_node), np.array(p_value)))

    # ----- float/int parameter values -----
    n_value = 5
    p_value = 0.13
    l = BinomialLayer(scope=Scope([1]), n_nodes=3, n=n_value, p=p_value)

    for n_layer_node, p_layer_node in zip(l.n, l.p):
        tc.assertTrue(np.allclose(tl_toNumpy(n_layer_node), n_value))
        tc.assertTrue(np.allclose(tl_toNumpy(p_layer_node), p_value))

    # ----- list parameter values -----
    n_values = [3, 2, 7]
    p_values = [0.17, 0.8, 0.53]
    l = BinomialLayer(scope=[Scope([0]), Scope([1]), Scope([2])], n=n_values, p=p_values)

    for n_layer_node, p_layer_node, n_value, p_value in zip(l.n, l.p, n_values, p_values):
        tc.assertTrue(np.allclose(tl_toNumpy(n_layer_node), np.array(n_value)))
        tc.assertTrue(np.allclose(tl_toNumpy(p_layer_node), np.array(p_value)))

    # wrong number of values
    tc.assertRaises(
        ValueError,
        BinomialLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        n_values,
        p_values[:-1],
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        BinomialLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        n_values[:-1],
        p_values,
        n_nodes=3,
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        BinomialLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        [n_values for _ in range(3)],
        p_values,
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        BinomialLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        n_values,
        [p_values for _ in range(3)],
        n_nodes=3,
    )

    # ----- numpy parameter values -----

    l = BinomialLayer(
        scope=[Scope([0]), Scope([1]), Scope([2])],
        n=np.array(n_values),
        p=np.array(p_values),
    )

    for n_layer_node, p_layer_node, n_value, p_value in zip(l.n, l.p, n_values, p_values):
        tc.assertTrue(np.allclose(tl_toNumpy(n_layer_node), np.array(n_value)))
        tc.assertTrue(np.allclose(tl_toNumpy(p_layer_node), np.array(p_value)))

    # wrong number of values
    tc.assertRaises(
        ValueError,
        BinomialLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        np.array(n_values[:-1]),
        np.array(p_values),
    )
    tc.assertRaises(
        ValueError,
        BinomialLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        np.array(n_values),
        np.array(p_values[:-1]),
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        BinomialLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        n_values,
        np.array([p_values for _ in range(3)]),
    )
    tc.assertRaises(
        ValueError,
        BinomialLayer,
        [Scope([0]), Scope([1]), Scope([2])],
        np.array([n_values for _ in range(3)]),
        p_values,
    )

    # ---- different scopes -----
    l = BinomialLayer(scope=Scope([1]), n_nodes=3, n=2)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, BinomialLayer, Scope([0]), n_nodes=0, n=2)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, BinomialLayer, Scope([]), n_nodes=3, n=2)
    tc.assertRaises(ValueError, BinomialLayer, [], n_nodes=3, n=2)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1]), Scope([0]), Scope([0])]
    l = BinomialLayer(scope=[Scope([1]), Scope([0])], n_nodes=3, n=2)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

def test_accept(do_for_all_backends):

    # discrete meta type (should reject)
    tc.assertFalse(
        BinomialLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
            ]
        )
    )

    # feature type instance
    tc.assertTrue(
        BinomialLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Binomial(n=3)]),
                FeatureContext(Scope([1]), [FeatureTypes.Binomial(n=3)]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        BinomialLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # conditional scope
    tc.assertFalse(BinomialLayer.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Binomial(n=3)])]))

    # multivariate signature
    tc.assertFalse(
        BinomialLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [
                        FeatureTypes.Binomial(n=3),
                        FeatureTypes.Binomial(n=3),
                    ],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    binomial = BinomialLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Binomial(n=3)]),
            FeatureContext(Scope([1]), [FeatureTypes.Binomial(n=5)]),
        ]
    )
    tc.assertTrue(np.all(tl_toNumpy(binomial.n) == np.array([3, 5])))
    tc.assertTrue(np.allclose(tl_toNumpy(binomial.p), np.array([0.5, 0.5])))
    tc.assertTrue(binomial.scopes_out == [Scope([0]), Scope([1])])

    binomial = BinomialLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Binomial(n=3, p=0.75)]),
            FeatureContext(Scope([1]), [FeatureTypes.Binomial(n=5, p=0.25)]),
        ]
    )
    tc.assertTrue(np.all(tl_toNumpy(binomial.n) == np.array([3, 5])))
    tc.assertTrue(np.allclose(tl_toNumpy(binomial.p), np.array([0.75, 0.25])))
    tc.assertTrue(binomial.scopes_out == [Scope([0]), Scope([1])])

    # ----- invalid arguments -----

    # discrete meta type
    tc.assertFalse(
        BinomialLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
            ]
        )
    )

    # invalid feature type
    tc.assertRaises(
        ValueError,
        BinomialLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        BinomialLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Binomial(3)])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        BinomialLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Binomial(3), FeatureTypes.Binomial(5)],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        BinomialInstLayer = BinomialLayerBase
    elif tl.get_backend() == "pytorch":
        BinomialInstLayer = BinomialLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(BinomialLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        BinomialLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Binomial(n=3)]),
                FeatureContext(Scope([1]), [FeatureTypes.Binomial(n=5)]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    binomial = AutoLeaf(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Binomial(n=3, p=0.75)]),
            FeatureContext(Scope([1]), [FeatureTypes.Binomial(n=5, p=0.25)]),
        ]
    )
    tc.assertTrue(isinstance(binomial, BinomialInstLayer))
    tc.assertTrue(np.all(tl_toNumpy(binomial.n) == np.array([3, 5])))
    tc.assertTrue(np.allclose(tl_toNumpy(binomial.p), np.array([0.75, 0.25])))
    tc.assertTrue(binomial.scopes_out == [Scope([0]), Scope([1])])

def test_layer_structural_marginalization(do_for_all_backends):

    # ---------- same scopes -----------

    l = BinomialLayer(scope=Scope([1]), p=[0.73, 0.29], n_nodes=2, n=2)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.p), tl_toNumpy(l_marg.p)))

    # ---------- different scopes -----------

    l = BinomialLayer(scope=[Scope([1]), Scope([0])], n=[3, 2], p=[0.73, 0.29])

    if tl.get_backend() == "numpy":
        BinomialInst = BinomialBase
        BinomialInstLayer = BinomialLayerBase
    elif tl.get_backend() == "pytorch":
        BinomialInst = BinomialTorch
        BinomialInstLayer = BinomialLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, BinomialInst))
    tc.assertEqual(l_marg.scope, Scope([0]))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.n), np.array(2)))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.p), np.array(0.29)))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, BinomialInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.n), np.array(2)))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.p), np.array(0.29)))

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.n), tl_toNumpy(l_marg.n)))
    tc.assertTrue(np.allclose(tl_toNumpy(l.p), tl_toNumpy(l_marg.p)))

def test_layer_dist(do_for_all_backends):

    n_values = [3, 2, 7]
    p_values = [0.73, 0.29, 0.5]
    l = BinomialLayer(
        scope=[Scope([1]), Scope([0]), Scope([2])],
        n=n_values,
        p=p_values,
        n_nodes=3,
    )

    # ----- full dist -----
    dist = l.dist()

    if tl.get_backend() == "numpy":
        p_list = [d.kwds.get("p") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    if tl.get_backend() == "numpy":
        n_list = [d.kwds.get("n") for d in dist]
    elif tl.get_backend() == "pytorch":
        n_list = dist.total_count
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for n_value, p_value, n_dist, p_dist in zip(n_values, p_values, n_list, p_list):
        tc.assertTrue(np.allclose(np.array(n_value), tl_toNumpy(n_dist)))
        tc.assertTrue(np.allclose(np.array(p_value), tl_toNumpy(p_dist)))

    # ----- partial dist -----
    dist = l.dist([1, 2])

    if tl.get_backend() == "numpy":
        p_list = [d.kwds.get("p") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    if tl.get_backend() == "numpy":
        n_list = [d.kwds.get("n") for d in dist]
    elif tl.get_backend() == "pytorch":
        n_list = dist.total_count
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for n_value, p_value, n_dist, p_dist in zip(n_values[1:], p_values[1:], n_list, p_list):
        tc.assertTrue(np.allclose(np.array(n_value), tl_toNumpy(n_dist)))
        tc.assertTrue(np.allclose(np.array(p_value), tl_toNumpy(p_dist)))

    dist = l.dist([1, 0])

    if tl.get_backend() == "numpy":
        p_list = [d.kwds.get("p") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    if tl.get_backend() == "numpy":
        n_list = [d.kwds.get("n") for d in dist]
    elif tl.get_backend() == "pytorch":
        n_list = dist.total_count
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for n_value, p_value, n_dist, p_dist in zip(
        reversed(n_values[:-1]),
        reversed(p_values[:-1]),
        n_list,
        p_list,
    ):
        tc.assertTrue(np.allclose(np.array(n_value), tl_toNumpy(n_dist)))
        tc.assertTrue(np.allclose(np.array(p_value), tl_toNumpy(p_dist)))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    binomial = BinomialLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
        n=[2, 5, 2],
        p=[0.2, 0.9, 0.31],)
    for backend in backends:
        with tl.backend_context(backend):
            binomial_updated = updateBackend(binomial)
            tc.assertTrue(np.all(binomial.scopes_out == binomial_updated.scopes_out))
            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*binomial.get_params()[0]]),
                    np.array([*binomial_updated.get_params()[0]]),
                )
            )

            tc.assertTrue(
                np.allclose(
                    np.array([*binomial.get_params()[1]]),
                    np.array([*binomial_updated.get_params()[1]]),
                )
            )

def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    binomial_default = BinomialLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
                             n=[2, 5, 2],
                             p=[0.2, 0.9, 0.31], )
    tc.assertTrue(binomial_default.dtype == tl.float32)
    tc.assertTrue(binomial_default.p.dtype == tl.float32)

    # change to float64 model
    binomial_updated = BinomialLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
                                     n=[2, 5, 2],
                                     p=[0.2, 0.9, 0.31], )
    binomial_updated.to_dtype(tl.float64)
    tc.assertTrue(binomial_updated.dtype == tl.float64)
    tc.assertTrue(binomial_updated.p.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array([*binomial_default.get_params()]),
            np.array([*binomial_updated.get_params()]),
        )
    )

def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    torch.set_default_dtype(torch.float32)
    binomial_default = BinomialLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
                                     n=[2, 5, 2],
                                     p=[0.2, 0.9, 0.31], )
    binomial_updated = BinomialLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
                                     n=[2, 5, 2],
                                     p=[0.2, 0.9, 0.31], )
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, binomial_updated.to_device, cuda)
        return

    # put model on gpu
    binomial_updated.to_device(cuda)

    tc.assertTrue(binomial_default.device.type == "cpu")
    tc.assertTrue(binomial_updated.device.type == "cuda")

    tc.assertTrue(binomial_default.p.device.type == "cpu")
    tc.assertTrue(binomial_updated.p.device.type == "cuda")

    tc.assertTrue(binomial_default.n.device.type == "cpu")
    tc.assertTrue(binomial_updated.n.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array([*binomial_default.get_params()]),
            np.array([*binomial_updated.get_params()]),
        )
    )



if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
