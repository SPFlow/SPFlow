import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.nodes.leaves.parametric.gamma import Gamma as GammaBase
from spflow.base.structure.general.layers.leaves.parametric.gamma import GammaLayer as GammaLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import marginalize
from spflow.torch.structure.general.nodes.leaves.parametric.gamma import Gamma as GammaTorch
from spflow.torch.structure.general.layers.leaves.parametric.gamma import GammaLayer as GammaLayerTorch
from spflow.torch.structure.general.layers.leaves.parametric.gamma import updateBackend

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.general.layers.leaves.parametric.general_gamma import GammaLayer
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()


def test_layer_initialization(do_for_all_backends):

    # ----- check attributes after correct initialization -----
    alpha_values = [0.5, 2.3, 1.0]
    beta_values = [1.3, 1.0, 0.2]
    l = GammaLayer(scope=Scope([1]), n_nodes=3, alpha=alpha_values, beta=beta_values)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
    # make sure parameter properties works correctly
    for alpha_layer_node, beta_layer_node, alpha_value, beta_value in zip(
        l.alpha, l.beta, alpha_values, beta_values
    ):
        tc.assertTrue(np.allclose(tl_toNumpy(alpha_layer_node), alpha_value))
        tc.assertTrue(np.allclose(tl_toNumpy(beta_layer_node), beta_value))

    # ----- float/int parameter values -----
    alpha_value = 0.73
    beta_value = 1.9
    l = GammaLayer(scope=Scope([1]), n_nodes=3, alpha=alpha_value, beta=beta_value)

    for alpha_layer_node, beta_layer_node in zip(l.alpha, l.beta):
        tc.assertTrue(np.allclose(tl_toNumpy(alpha_layer_node), alpha_value))
        tc.assertTrue(np.allclose(tl_toNumpy(beta_layer_node), beta_value))

    # ----- list parameter values -----
    alpha_values = [0.17, 0.8, 0.53]
    beta_values = [0.9, 1.34, 0.98]
    l = GammaLayer(scope=Scope([1]), n_nodes=3, alpha=alpha_values, beta=beta_values)

    for alpha_layer_node, beta_layer_node, alpha_value, beta_value in zip(
        l.alpha, l.beta, alpha_values, beta_values
    ):
        tc.assertTrue(np.allclose(tl_toNumpy(alpha_layer_node), alpha_value))
        tc.assertTrue(np.allclose(tl_toNumpy(beta_layer_node), beta_value))

    # wrong number of values
    tc.assertRaises(
        ValueError,
        GammaLayer,
        Scope([0]),
        alpha_values[:-1],
        beta_values,
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        GammaLayer,
        Scope([0]),
        alpha_values,
        beta_values[:-1],
        n_nodes=3,
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        GammaLayer,
        Scope([0]),
        alpha_values,
        [beta_values for _ in range(3)],
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        GammaLayer,
        Scope([0]),
        [alpha_values for _ in range(3)],
        beta_values,
        n_nodes=3,
    )

    # ----- numpy parameter values -----

    l = GammaLayer(
        scope=Scope([1]),
        n_nodes=3,
        alpha=np.array(alpha_values),
        beta=np.array(beta_values),
    )

    for alpha_layer_node, beta_layer_node, alpha_value, beta_value in zip(
        l.alpha, l.beta, alpha_values, beta_values
    ):
        tc.assertTrue(np.allclose(tl_toNumpy(alpha_layer_node), alpha_value))
        tc.assertTrue(np.allclose(tl_toNumpy(beta_layer_node), beta_value))

    # wrong number of values
    tc.assertRaises(
        ValueError,
        GammaLayer,
        Scope([0]),
        np.array(alpha_values[:-1]),
        np.array(beta_values),
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        GammaLayer,
        Scope([0]),
        np.array(alpha_values),
        np.array(beta_values[:-1]),
        n_nodes=3,
    )
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        GammaLayer,
        Scope([0]),
        np.array(alpha_values),
        np.array([beta_values for _ in range(3)]),
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        GammaLayer,
        Scope([0]),
        np.array([alpha_values for _ in range(3)]),
        np.array(beta_values),
        n_nodes=3,
    )

    # ---- different scopes -----
    l = GammaLayer(scope=Scope([1]), n_nodes=3)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, GammaLayer, Scope([0]), n_nodes=0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, GammaLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, GammaLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1]), Scope([0]), Scope([0])]
    l = GammaLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

def test_accept(do_for_all_backends):

    # continuous meta type
    tc.assertTrue(
        GammaLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # feature type class
    tc.assertTrue(
        GammaLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Gamma]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # feature type instance
    tc.assertTrue(
        GammaLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Gamma(1.0, 1.0)]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        GammaLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # conditional scope
    tc.assertFalse(GammaLayer.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

    # multivariate signature
    tc.assertFalse(
        GammaLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    gamma = GammaLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
            FeatureContext(Scope([1]), [FeatureTypes.Continuous]),
        ]
    )
    tc.assertTrue(gamma.scopes_out == [Scope([0]), Scope([1])])

    gamma = GammaLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Gamma]),
            FeatureContext(Scope([1]), [FeatureTypes.Gamma]),
        ]
    )
    tc.assertTrue(gamma.scopes_out == [Scope([0]), Scope([1])])

    gamma = GammaLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Gamma(1.0, 1.0)]),
            FeatureContext(Scope([1]), [FeatureTypes.Gamma(1.0, 1.0)]),
        ]
    )
    tc.assertTrue(gamma.scopes_out == [Scope([0]), Scope([1])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        GammaLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # conditional scope
    tc.assertRaises(
        ValueError,
        GammaLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        GammaLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(GammaLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        GammaLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Gamma]),
                FeatureContext(Scope([1]), [FeatureTypes.Gamma]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    gamma = AutoLeaf(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Gamma(alpha=1.5, beta=0.5)]),
            FeatureContext(Scope([1]), [FeatureTypes.Gamma(alpha=0.5, beta=1.5)]),
        ]
    )
    tc.assertTrue(gamma.scopes_out == [Scope([0]), Scope([1])])

def test_layer_structural_marginalization(do_for_all_backends):

    # ---------- same scopes -----------

    if tl.get_backend() == "numpy":
        GammaInst = GammaBase
        GammaInstLayer = GammaLayerBase
    elif tl.get_backend() == "pytorch":
        GammaInst = GammaTorch
        GammaInstLayer = GammaLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    l = GammaLayer(scope=Scope([1]), alpha=[0.73, 0.29], beta=[0.41, 1.9], n_nodes=2)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.alpha), tl_toNumpy(l_marg.alpha)))
    tc.assertTrue(np.allclose(tl_toNumpy(l.beta), tl_toNumpy(l_marg.beta)))

    # ---------- different scopes -----------

    l = GammaLayer(scope=[Scope([1]), Scope([0])], alpha=[0.73, 0.29], beta=[0.41, 1.9])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, GammaInst))
    tc.assertEqual(l_marg.scope, Scope([0]))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.alpha), tl.tensor(0.29)))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.beta), tl.tensor(1.9)))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, GammaInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.alpha), tl.tensor(0.29)))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.beta), tl.tensor(1.9)))

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.alpha), tl_toNumpy(l_marg.alpha)))
    tc.assertTrue(np.allclose(tl_toNumpy(l.beta), tl_toNumpy(l_marg.beta)))

def test_layer_dist(do_for_all_backends):

    alpha_values = [0.73, 0.29, 0.5]
    beta_values = [0.9, 1.34, 0.98]
    l = GammaLayer(scope=Scope([1]), alpha=alpha_values, beta=beta_values, n_nodes=3)

    # ----- full dist -----
    dist = l.dist()

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
        tc.assertTrue(np.allclose(tl.tensor(alpha_value), tl_toNumpy(alpha_dist)))
        tc.assertTrue(np.allclose(tl.tensor(beta_value), tl_toNumpy(beta_dist)))

    # ----- partial dist -----
    dist = l.dist([1, 2])

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
        tc.assertTrue(np.allclose(tl.tensor(alpha_value), tl_toNumpy(alpha_dist)))
        tc.assertTrue(np.allclose(tl.tensor(beta_value), tl_toNumpy(beta_dist)))

    dist = l.dist([1, 0])

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
        tc.assertTrue(np.allclose(tl.tensor(alpha_value), tl_toNumpy(alpha_dist)))
        tc.assertTrue(np.allclose(tl.tensor(beta_value), tl_toNumpy(beta_dist)))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    gamma = GammaLayer(scope=[Scope([0]), Scope([1]), Scope([0])],
        alpha=[0.2, 0.9, 0.31],
        beta=[1.9, 0.3, 0.71])
    for backend in backends:
        with tl.backend_context(backend):
            gamma_updated = updateBackend(gamma)
            tc.assertTrue(np.all(gamma.scopes_out == gamma_updated.scopes_out))
            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*gamma.get_params()[0]]),
                    np.array([*gamma_updated.get_params()[0]]),
                )
            )

            tc.assertTrue(
                np.allclose(
                    np.array([*gamma.get_params()[1]]),
                    np.array([*gamma_updated.get_params()[1]]),
                )
            )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
