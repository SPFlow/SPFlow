import unittest

import numpy as np
import torch
import tensorly as tl
from spflow.base.structure.general.nodes.leaves.parametric.cond_bernoulli import CondBernoulli as CondBernoulliBase
from spflow.base.structure.general.layers.leaves.parametric.cond_bernoulli import CondBernoulliLayer as CondBernoulliLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure import marginalize
from spflow.torch.structure.general.nodes.leaves.parametric.cond_bernoulli import CondBernoulli as CondBernoulliTorch
from spflow.torch.structure.general.layers.leaves.parametric.cond_bernoulli import CondBernoulliLayer as CondBernoulliLayerTorch

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.general.layers.leaves.parametric.general_cond_bernoulli import CondBernoulliLayer
from spflow.torch.structure.general.layers.leaves.parametric.cond_bernoulli import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_initialization(do_for_all_backends):

    # ----- check attributes after correct initialization -----
    l = CondBernoulliLayer(scope=Scope([1], [0]), n_nodes=3)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])]))

    # ---- different scopes -----
    l = CondBernoulliLayer(scope=Scope([1], [0]), n_nodes=3)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, CondBernoulliLayer, Scope([0], [1]), n_nodes=0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, CondBernoulliLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, CondBernoulliLayer, Scope([0]), n_nodes=3)
    tc.assertRaises(ValueError, CondBernoulliLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
    l = CondBernoulliLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

    # -----number of cond_f functions -----
    CondBernoulliLayer(
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[lambda data: {"p": 0.5}, lambda data: {"p": 0.5}],
    )
    tc.assertRaises(
        ValueError,
        CondBernoulliLayer,
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[lambda data: {"p": 0.5}],
    )

def test_retrieve_param(do_for_all_backends):

    # ----- float/int parameter values -----
    p_value = 0.13
    l = CondBernoulliLayer(scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"p": p_value})

    for p in l.retrieve_params(np.array([[1.0]]), DispatchContext()):
        tc.assertTrue(np.all(tl_toNumpy(p) == p_value))

    # ----- list parameter values -----
    p_values = [0.17, 0.8, 0.53]
    l = CondBernoulliLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        cond_f=lambda data: {"p": p_values},
    )

    for p_layer_node, p_actual in zip(
        l.retrieve_params(np.array([[1.0]]), DispatchContext()),
        p_values,
    ):
        tc.assertTrue(np.all(tl_toNumpy(p_layer_node) == p_actual))

    # wrong number of values
    l.set_cond_f(lambda data: {"p": p_values[:-1]})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        torch.tensor([[1]]),
        DispatchContext(),
    )

    # wrong number of dimensions (nested list)
    l.set_cond_f(lambda data: {"p": [p_values for _ in range(3)]})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        torch.tensor([[1]]),
        DispatchContext(),
    )

    # ----- numpy parameter values -----
    l.set_cond_f(lambda data: {"p": np.array(p_values)})

    for p_layer_node, p_actual in zip(
        l.retrieve_params(np.array([[1.0]]), DispatchContext()),
        p_values,
    ):
        tc.assertTrue(p_layer_node == p_actual)

    # wrong number of values
    l.set_cond_f(lambda data: {"p": np.array(p_values[:-1])})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        torch.tensor([[1]]),
        DispatchContext(),
    )

    # wrong number of dimensions (nested list)
    l.set_cond_f(lambda data: {"p": np.array([p_values for _ in range(3)])})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        torch.tensor([[1]]),
        DispatchContext(),
    )

def test_accept(do_for_all_backends):

    # discrete meta type
    tc.assertTrue(
        CondBernoulliLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Discrete]),
            ]
        )
    )

    # Bernoulli feature type class
    tc.assertTrue(
        CondBernoulliLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Bernoulli]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Discrete]),
            ]
        )
    )

    # Bernoulli feature type instance
    tc.assertTrue(
        CondBernoulliLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Bernoulli(0.5)]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Bernoulli(0.5)]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        CondBernoulliLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # non-conditional scope
    tc.assertFalse(CondBernoulliLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # multivariate signature
    tc.assertFalse(
        CondBernoulliLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    bernoulli = CondBernoulliLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
            FeatureContext(Scope([1], [3]), [FeatureTypes.Discrete]),
        ]
    )
    tc.assertTrue(bernoulli.scopes_out == [Scope([0], [2]), Scope([1], [3])])

    bernoulli = CondBernoulliLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Bernoulli]),
            FeatureContext(Scope([1], [3]), [FeatureTypes.Bernoulli]),
        ]
    )
    tc.assertTrue(bernoulli.scopes_out == [Scope([0], [2]), Scope([1], [3])])

    bernoulli = CondBernoulliLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Bernoulli(p=0.75)]),
            FeatureContext(Scope([1], [3]), [FeatureTypes.Bernoulli(p=0.25)]),
        ]
    )
    tc.assertTrue(bernoulli.scopes_out == [Scope([0], [2]), Scope([1], [3])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondBernoulliLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondBernoulliLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondBernoulliLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Discrete, FeatureTypes.Discrete],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondBernoulliLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondBernoulliLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Bernoulli()]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Bernoulli()]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    bernoulli = AutoLeaf(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Bernoulli(p=0.75)]),
            FeatureContext(Scope([1], [3]), [FeatureTypes.Bernoulli(p=0.25)]),
        ]
    )
    tc.assertTrue(bernoulli.scopes_out == [Scope([0], [2]), Scope([1], [3])])

def test_layer_structural_marginalization(do_for_all_backends):

    if tl.get_backend() == "numpy":
        CondBernoulliInst = CondBernoulliBase
        CondBernoulliInstLayer = CondBernoulliLayerBase
    elif tl.get_backend() == "pytorch":
        CondBernoulliInst = CondBernoulliTorch
        CondBernoulliInstLayer = CondBernoulliLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")


    # ---------- same scopes -----------

    l = CondBernoulliLayer(scope=Scope([1], [0]), n_nodes=2)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

    # ---------- different scopes -----------

    l = CondBernoulliLayer(scope=[Scope([1], [2]), Scope([0], [2])])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, CondBernoulliInst))
    tc.assertEqual(l_marg.scope, Scope([0], [2]))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, CondBernoulliInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])

def test_layer_dist(do_for_all_backends):

    p_values = torch.tensor([0.73, 0.29, 0.5])
    l = CondBernoulliLayer(scope=Scope([1], [0]), n_nodes=3)

    # ----- full dist -----
    dist = l.dist(p_values)

    if tl.get_backend() == "numpy":
        p_list = [d.kwds.get("p") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for p_value, p_dist in zip(p_values, p_list):
        tc.assertTrue(torch.allclose(p_value, p_dist))

    # ----- partial dist -----
    dist = l.dist(p_values, [1, 2])

    if tl.get_backend() == "numpy":
        p_list = [d.kwds.get("p") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for p_value, p_dist in zip(p_values[1:], p_list):
        tc.assertTrue(torch.allclose(p_value, p_dist))

    dist = l.dist(p_values, [1, 0])

    if tl.get_backend() == "numpy":
        p_list = [d.kwds.get("p") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for p_value, p_dist in zip(reversed(p_values[:-1]), p_list):
        tc.assertTrue(torch.allclose(p_value, p_dist))
"""
def test_layer_backend_conversion_1(self):

    torch_layer = CondBernoulliLayer(scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])])
    base_layer = toBase(torch_layer)

    self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
    self.assertEqual(base_layer.n_out, torch_layer.n_out)

def test_layer_backend_conversion_2(self):

    base_layer = BaseCondBernoulliLayer(scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])])
    torch_layer = toTorch(base_layer)

    self.assertTrue(np.all(base_layer.scopes_out == torch_layer.scopes_out))
    self.assertEqual(base_layer.n_out, torch_layer.n_out)
"""
def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    bernoulli = CondBernoulliLayer(scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])])
    for backend in backends:
        with tl.backend_context(backend):
            bernoulli_updated = updateBackend(bernoulli)
            tc.assertTrue(np.all(bernoulli.scopes_out == bernoulli_updated.scopes_out))
            


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
