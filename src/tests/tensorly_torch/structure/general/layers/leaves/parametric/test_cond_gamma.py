import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.nodes.leaves.parametric.cond_gamma import CondGamma as CondGammaBase
from spflow.base.structure.general.layers.leaves.parametric.cond_gamma import CondGammaLayer as CondGammaLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure import marginalize, toBase, toTorch
from spflow.torch.structure.general.nodes.leaves.parametric.cond_gamma import CondGamma as CondGammaTorch
from spflow.torch.structure.general.layers.leaves.parametric.cond_gamma import CondGammaLayer as CondGammaLayerTorch
from spflow.torch.structure.general.layers.leaves.parametric.cond_gamma import updateBackend

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.general.layers.leaves.parametric.general_cond_gamma import CondGammaLayer
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()


def test_layer_initialization(do_for_all_backends):

    # ----- check attributes after correct initialization -----
    l = CondGammaLayer(scope=Scope([1], [0]), n_nodes=3)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])]))

    # ---- different scopes -----
    l = CondGammaLayer(scope=Scope([1], [0]), n_nodes=3)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, CondGammaLayer, Scope([0], [1]), n_nodes=0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, CondGammaLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, CondGammaLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
    l = CondGammaLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

    # -----number of cond_f functions -----
    CondGammaLayer(
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[
            lambda data: {"alpha": 0.5, "beta": 0.5},
            lambda data: {"alpha": 0.5, "beta": 0.5},
        ],
    )
    tc.assertRaises(
        ValueError,
        CondGammaLayer,
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[lambda data: {"alpha": 0.5, "beta": 0.5}],
    )

def test_accept(do_for_all_backends):

    # continuous meta type
    tc.assertTrue(
        CondGammaLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # feature type class
    tc.assertTrue(
        CondGammaLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Gamma]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # feature type instance
    tc.assertTrue(
        CondGammaLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Gamma(1.0, 1.0)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        CondGammaLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # non-conditional scope
    tc.assertFalse(CondGammaLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # multivariate signature
    tc.assertFalse(
        CondGammaLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    gamma = CondGammaLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
        ]
    )
    tc.assertTrue(gamma.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    gamma = CondGammaLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Gamma]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Gamma]),
        ]
    )
    tc.assertTrue(gamma.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    gamma = CondGammaLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Gamma(1.0, 1.0)]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Gamma(1.0, 1.0)]),
        ]
    )
    tc.assertTrue(gamma.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondGammaLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondGammaLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondGammaLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondGammaLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondGammaLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Gamma]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Gamma]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    gamma = AutoLeaf(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Gamma(alpha=1.5, beta=0.5)]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Gamma(alpha=0.5, beta=1.5)]),
        ]
    )
    tc.assertTrue(gamma.scopes_out == [Scope([0], [2]), Scope([1], [2])])

def test_retrieve_params(do_for_all_backends):

    # ----- float/int parameter values -----
    alpha_value = 0.73
    beta_value = 1.9
    l = CondGammaLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        cond_f=lambda data: {"alpha": alpha_value, "beta": beta_value},
    )

    for alpha_layer_node, beta_layer_node in zip(*l.retrieve_params(tl.tensor([[1.0]]), DispatchContext())):
        tc.assertTrue(np.allclose(tl_toNumpy(alpha_layer_node), alpha_value))
        tc.assertTrue(np.allclose(tl_toNumpy(beta_layer_node), beta_value))

    # ----- list parameter values -----
    alpha_values = [0.17, 0.8, 0.53]
    beta_values = [0.9, 1.34, 0.98]
    l = CondGammaLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        cond_f=lambda data: {"alpha": alpha_values, "beta": beta_values},
    )

    for alpha_value, beta_value, alpha_layer_node, beta_layer_node in zip(
        alpha_values, beta_values, *l.retrieve_params(tl.tensor([[1.0]]), DispatchContext())
    ):
        tc.assertTrue(np.allclose(tl_toNumpy(alpha_layer_node), alpha_value))
        tc.assertTrue(np.allclose(tl_toNumpy(beta_layer_node), beta_value))

    # wrong number of values
    l.set_cond_f(lambda data: {"alpha": alpha_values[:-1], "beta": beta_values})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )
    l.set_cond_f(lambda data: {"alpha": alpha_values, "beta": beta_values[:-1]})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    # wrong number of dimensions (nested list)
    l.set_cond_f(
        lambda data: {
            "alpha": [alpha_values for _ in range(3)],
            "beta": beta_values,
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
            "alpha": alpha_values,
            "beta": [beta_values for _ in range(3)],
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
            "alpha": np.array(alpha_values),
            "beta": np.array(beta_values),
        }
    )
    for alpha_actual, beta_actual, alpha_node, beta_node in zip(
        alpha_values, beta_values, *l.retrieve_params(tl.tensor([[1.0]]), DispatchContext())
    ):
        tc.assertTrue(alpha_node == alpha_actual)
        tc.assertTrue(beta_node == beta_actual)

    # wrong number of values
    l.set_cond_f(
        lambda data: {
            "alpha": np.array(alpha_values[:-1]),
            "beta": np.array(beta_values),
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
            "alpha": np.array(alpha_values),
            "beta": np.array(beta_values[:-1]),
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
            "alpha": np.array([alpha_values for _ in range(3)]),
            "beta": np.array(beta_values),
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
            "alpha": np.array(alpha_values),
            "beta": np.array([beta_values for _ in range(3)]),
        }
    )
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

def test_layer_structural_marginalization(do_for_all_backends):

    if tl.get_backend() == "numpy":
        CondGammaInst = CondGammaBase
        CondGammaInstLayer = CondGammaLayerBase
    elif tl.get_backend() == "pytorch":
        CondGammaInst = CondGammaTorch
        CondGammaInstLayer = CondGammaLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # ---------- same scopes -----------

    l = CondGammaLayer(scope=Scope([1], [0]), n_nodes=2)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

    # ---------- different scopes -----------

    l = CondGammaLayer(scope=[Scope([1], [2]), Scope([0], [2])])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, CondGammaInst))
    tc.assertEqual(l_marg.scope, Scope([0], [2]))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, CondGammaInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])

def test_layer_dist(do_for_all_backends):

    alpha_values = tl.tensor([0.73, 0.29, 0.5])
    beta_values = tl.tensor([0.9, 1.34, 0.98])
    l = CondGammaLayer(scope=Scope([1], [0]), n_nodes=3)

    # ----- full dist -----
    dist = l.dist(alpha=alpha_values, beta=beta_values)

    if tl.get_backend() == "numpy":
        beta_list = [1/d.kwds.get("scale") for d in dist]
        alpha_list = [d.kwds.get("a") for d in dist]
    elif tl.get_backend() == "pytorch":
        beta_list = dist.rate
        alpha_list = dist.concentration
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for alpha_value, beta_value, alpha_dist, beta_dist in zip(
        alpha_values, beta_values, alpha_list, beta_list
    ):
        tc.assertTrue(np.allclose(alpha_value, alpha_dist))
        tc.assertTrue(np.allclose(beta_value, beta_dist))

    # ----- partial dist -----
    dist = l.dist(alpha=alpha_values, beta=beta_values, node_ids=[1, 2])

    if tl.get_backend() == "numpy":
        beta_list = [1/d.kwds.get("scale") for d in dist]
        alpha_list = [d.kwds.get("a") for d in dist]
    elif tl.get_backend() == "pytorch":
        beta_list = dist.rate
        alpha_list = dist.concentration
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for alpha_value, beta_value, alpha_dist, beta_dist in zip(
        alpha_values[1:], beta_values[1:], alpha_list, beta_list
    ):
        tc.assertTrue(np.allclose(alpha_value, alpha_dist))
        tc.assertTrue(np.allclose(beta_value, beta_dist))

    dist = l.dist(alpha=alpha_values, beta=beta_values, node_ids=[1, 0])

    if tl.get_backend() == "numpy":
        beta_list = [1/d.kwds.get("scale") for d in dist]
        alpha_list = [d.kwds.get("a") for d in dist]
    elif tl.get_backend() == "pytorch":
        beta_list = dist.rate
        alpha_list = dist.concentration
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for alpha_value, beta_value, alpha_dist, beta_dist in zip(
        reversed(alpha_values[:-1]),
        reversed(beta_values[:-1]),
        alpha_list,
        beta_list,
    ):
        tc.assertTrue(np.allclose(alpha_value, alpha_dist))
        tc.assertTrue(np.allclose(beta_value, beta_dist))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    gamma = CondGammaLayer(scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])])
    for backend in backends:
        with tl.backend_context(backend):
            gamma_updated = updateBackend(gamma)
            tc.assertTrue(np.all(gamma.scopes_out == gamma_updated.scopes_out))
        # check conversion from torch to python



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
