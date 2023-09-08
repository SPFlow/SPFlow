import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.nodes.leaves.parametric.cond_exponential import CondExponential as CondExponentialBase
from spflow.base.structure.general.layers.leaves.parametric.cond_exponential import CondExponentialLayer as CondExponentialLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.torch.structure.general.nodes.leaves.parametric.cond_exponential import CondExponential as CondExponentialTorch
from spflow.torch.structure.general.layers.leaves.parametric.cond_exponential import CondExponentialLayer as CondExponentialLayerTorch
from spflow.torch.structure.general.layers.leaves.parametric.cond_exponential import updateBackend

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.general.layers.leaves.parametric.general_cond_exponential import CondExponentialLayer
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()


def test_layer_initialization(do_for_all_backends):

    # ----- check attributes after correct initialization -----

    l = CondExponentialLayer(scope=Scope([1], [0]), n_nodes=3)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])]))

    # ---- different scopes -----
    l = CondExponentialLayer(scope=Scope([1], [0]), n_nodes=3)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, CondExponentialLayer, Scope([0], [1]), n_nodes=0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, CondExponentialLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, CondExponentialLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
    l = CondExponentialLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

    # -----number of cond_f functions -----
    CondExponentialLayer(
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[lambda data: {"l": 1.5}, lambda data: {"l": 1.5}],
    )
    tc.assertRaises(
        ValueError,
        CondExponentialLayer,
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[lambda data: {"l": 0.5}],
    )

def test_retrieve_params(do_for_all_backends):

    # ----- float/int parameter values -----
    l_value = 0.73
    l = CondExponentialLayer(scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"l": l_value})

    for l_layer_node in l.retrieve_params(torch.tensor([[1]]), DispatchContext()):
        tc.assertTrue(np.allclose(tl_toNumpy(l_layer_node), np.array(l_value)))

    # ----- list parameter values -----
    l_values = [0.17, 0.8, 0.53]
    l = CondExponentialLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        cond_f=lambda data: {"l": l_values},
    )

    for l_layer_node, l_value in zip(l.retrieve_params(torch.tensor([[1]]), DispatchContext()), l_values):
        tc.assertTrue(np.allclose(tl_toNumpy(l_layer_node), np.array(l_value)))

    # wrong number of values
    l.set_cond_f(lambda data: {"l": l_values[:-1]})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    # wrong number of dimensions (nested list)
    l.set_cond_f(lambda data: {"l": [l_values for _ in range(3)]})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    # ----- numpy parameter values -----
    l.set_cond_f(lambda data: {"l": np.array(l_values)})
    for l_node, l_actual in zip(
        l.retrieve_params(tl.tensor([[1.0]]), DispatchContext()),
        l_values,
    ):
        tc.assertTrue(l_node == l_actual)

    # wrong number of values
    l.set_cond_f(lambda data: {"l": np.array(l_values[:-1])})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    # wrong number of dimensions (nested list)
    l.set_cond_f(lambda data: {"l": np.array([l_values for _ in range(3)])})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

def test_accept(do_for_all_backends):

    # continuous meta type
    tc.assertTrue(
        CondExponentialLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # Exponential feature type class
    tc.assertTrue(
        CondExponentialLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Exponential]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # Exponential feature type instance
    tc.assertTrue(
        CondExponentialLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Exponential(1.0)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        CondExponentialLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # non-conditional scope
    tc.assertFalse(CondExponentialLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # multivariate signature
    tc.assertFalse(
        CondExponentialLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    exponential = CondExponentialLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
        ]
    )
    tc.assertTrue(exponential.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    exponential = CondExponentialLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Exponential]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Exponential]),
        ]
    )
    tc.assertTrue(exponential.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    exponential = CondExponentialLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Exponential(1.5)]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Exponential(0.5)]),
        ]
    )
    tc.assertTrue(exponential.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondExponentialLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondExponentialLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondExponentialLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondExponentialLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondExponentialLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Exponential]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Exponential]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    exponential = AutoLeaf(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Exponential(l=1.5)]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Exponential(l=0.5)]),
        ]
    )
    tc.assertTrue(exponential.scopes_out == [Scope([0], [2]), Scope([1], [2])])

def test_layer_structural_marginalization(do_for_all_backends):

    # ---------- same scopes -----------

    l = CondExponentialLayer(scope=Scope([1], [0]), n_nodes=2)

    if tl.get_backend() == "numpy":
        CondExponentialInst = CondExponentialBase
        CondExponentialInstLayer = CondExponentialLayerBase
    elif tl.get_backend() == "pytorch":
        CondExponentialInst = CondExponentialTorch
        CondExponentialInstLayer = CondExponentialLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

    # ---------- different scopes -----------

    l = CondExponentialLayer(scope=[Scope([1], [2]), Scope([0], [2])])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, CondExponentialInst))
    tc.assertEqual(l_marg.scope, Scope([0], [2]))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, CondExponentialInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])

def test_layer_dist(do_for_all_backends):

    l_values = tl.tensor([0.73, 0.29, 0.5])
    l = CondExponentialLayer(scope=Scope([1], [0]), n_nodes=3)

    # ----- full dist -----
    dist = l.dist(l_values)

    if tl.get_backend() == "numpy":
        l_list = [1/d.kwds.get("scale") for d in dist]
    elif tl.get_backend() == "pytorch":
        l_list = dist.rate
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for l_value, l_dist in zip(l_values, l_list):
        tc.assertTrue(np.allclose(tl_toNumpy(l_value), tl_toNumpy(l_dist)))

    # ----- partial dist -----
    dist = l.dist(l_values, [1, 2])

    if tl.get_backend() == "numpy":
        l_list = [1/d.kwds.get("scale") for d in dist]
    elif tl.get_backend() == "pytorch":
        l_list = dist.rate
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for l_value, l_dist in zip(l_values[1:], l_list):
        tc.assertTrue(np.allclose(l_value, l_dist))

    dist = l.dist(l_values, [1, 0])

    if tl.get_backend() == "numpy":
        l_list = [1/d.kwds.get("scale") for d in dist]
    elif tl.get_backend() == "pytorch":
        l_list = dist.rate
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for l_value, l_dist in zip(reversed(l_values[:-1]), l_list):
        tc.assertTrue(np.allclose(l_value, l_dist))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    exponential = CondExponentialLayer(scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])])
    for backend in backends:
        with tl.backend_context(backend):
            exponential_updated = updateBackend(exponential)
            tc.assertTrue(np.all(exponential.scopes_out == exponential_updated.scopes_out))
        # check conversion from torch to python



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
