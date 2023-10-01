import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.nodes.leaves.parametric.poisson import Poisson as PoissonBase
from spflow.base.structure.general.layers.leaves.parametric.poisson import PoissonLayer as PoissonLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.torch.structure import marginalize
from spflow.torch.structure.general.nodes.leaves.parametric.poisson import Poisson as PoissonTorch
from spflow.torch.structure.general.layers.leaves.parametric.poisson import PoissonLayer as PoissonLayerTorch
from spflow.torch.structure.general.layers.leaves.parametric.poisson import updateBackend

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.general.layers.leaves.parametric.general_poisson import PoissonLayer
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_initialization(do_for_all_backends):

    # ----- check attributes after correct initialization -----
    l_values = [0.5, 2.3, 1.0]
    l = PoissonLayer(scope=Scope([1]), n_nodes=3, l=l_values)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1]), Scope([1]), Scope([1])]))
    # make sure parameter properties works correctly
    for l_layer_node, l_value in zip(l.l, l_values):
        tc.assertTrue(np.allclose(tl_toNumpy(l_layer_node), tl.tensor(l_value)))

    # ----- float/int parameter values -----
    l_value = 0.73
    l = PoissonLayer(scope=Scope([1]), n_nodes=3, l=l_value)

    for l_layer_node in l.l:
        tc.assertTrue(np.allclose(tl_toNumpy(l_layer_node), tl.tensor(l_value)))

    # ----- list parameter values -----
    l_values = [0.17, 0.8, 0.53]
    l = PoissonLayer(scope=Scope([1]), n_nodes=3, l=l_values)

    for l_layer_node, l_value in zip(l.l, l_values):
        tc.assertTrue(np.allclose(tl_toNumpy(l_layer_node), tl.tensor(l_value)))

    # wrong number of values
    tc.assertRaises(ValueError, PoissonLayer, Scope([0]), l_values[:-1], n_nodes=3)
    # wrong number of dimensions (nested list)
    tc.assertRaises(
        ValueError,
        PoissonLayer,
        Scope([0]),
        [l_values for _ in range(3)],
        n_nodes=3,
    )

    # ----- numpy parameter values -----

    l = PoissonLayer(scope=Scope([1]), n_nodes=3, l=np.array(l_values))

    for l_layer_node, l_value in zip(l.l, l_values):
        tc.assertTrue(np.allclose(tl_toNumpy(l_layer_node), tl.tensor(l_value)))

    # wrong number of values
    tc.assertRaises(
        ValueError,
        PoissonLayer,
        Scope([0]),
        np.array(l_values[:-1]),
        n_nodes=3,
    )
    tc.assertRaises(
        ValueError,
        PoissonLayer,
        Scope([0]),
        np.array([l_values for _ in range(3)]),
        n_nodes=3,
    )

    # ---- different scopes -----
    l = PoissonLayer(scope=Scope([1]), n_nodes=3)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, PoissonLayer, Scope([0]), n_nodes=0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, PoissonLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, PoissonLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1]), Scope([0]), Scope([0])]
    l = PoissonLayer(scope=[Scope([1]), Scope([0])], n_nodes=3)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

def test_accept(do_for_all_backends):

    # continuous meta type
    tc.assertTrue(
        PoissonLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
            ]
        )
    )

    # Poisson feature type class
    tc.assertTrue(
        PoissonLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Poisson]),
                FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
            ]
        )
    )

    # Poisson feature type instance
    tc.assertTrue(
        PoissonLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Poisson(1.0)]),
                FeatureContext(Scope([1]), [FeatureTypes.Poisson(1.0)]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        PoissonLayer.accepts(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1]), [FeatureTypes.Poisson(1.0)]),
            ]
        )
    )

    # conditional scope
    tc.assertFalse(PoissonLayer.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

    # multivariate signature
    tc.assertFalse(
        PoissonLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    poisson = PoissonLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Discrete]),
            FeatureContext(Scope([1]), [FeatureTypes.Discrete]),
        ]
    )
    tc.assertTrue(poisson.scopes_out == [Scope([0]), Scope([1])])

    poisson = PoissonLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Poisson]),
            FeatureContext(Scope([1]), [FeatureTypes.Poisson]),
        ]
    )
    tc.assertTrue(poisson.scopes_out == [Scope([0]), Scope([1])])

    poisson = PoissonLayer.from_signatures(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Poisson(l=1.5)]),
            FeatureContext(Scope([1]), [FeatureTypes.Poisson(l=2.0)]),
        ]
    )
    tc.assertTrue(poisson.scopes_out == [Scope([0]), Scope([1])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        PoissonLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        PoissonLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        PoissonLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    if tl.get_backend() == "numpy":
        PoissonInstLayer = PoissonLayerBase
    elif tl.get_backend() == "pytorch":
        PoissonInstLayer = PoissonLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(PoissonLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        PoissonLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0]), [FeatureTypes.Poisson]),
                FeatureContext(Scope([1]), [FeatureTypes.Poisson]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    poisson = AutoLeaf(
        [
            FeatureContext(Scope([0]), [FeatureTypes.Poisson(l=1.5)]),
            FeatureContext(Scope([1]), [FeatureTypes.Poisson(l=2.0)]),
        ]
    )
    tc.assertTrue(isinstance(poisson, PoissonInstLayer))
    tc.assertTrue(poisson.scopes_out == [Scope([0]), Scope([1])])

def test_layer_structural_marginalization(do_for_all_backends):

    if tl.get_backend() == "numpy":
        PoissonInst = PoissonBase
        PoissonInstLayer = PoissonLayerBase
    elif tl.get_backend() == "pytorch":
        PoissonInst = PoissonTorch
        PoissonInstLayer = PoissonLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # ---------- same scopes -----------

    l = PoissonLayer(scope=Scope([1]), l=[0.73, 0.29], n_nodes=2)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.l), tl_toNumpy(l_marg.l)))

    # ---------- different scopes -----------

    l = PoissonLayer(scope=[Scope([1]), Scope([0])], l=[0.73, 0.29])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, PoissonInst))
    tc.assertEqual(l_marg.scope, Scope([0]))
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.l), tl.tensor(0.29)))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, PoissonInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)
    tc.assertTrue(np.allclose(tl_toNumpy(l_marg.l), tl.tensor(0.29)))

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([0])])
    tc.assertTrue(np.allclose(tl_toNumpy(l.l), tl_toNumpy(l_marg.l)))

def test_layer_dist(do_for_all_backends):

    l_values = [0.73, 0.29, 0.5]
    l = PoissonLayer(scope=Scope([1]), l=l_values, n_nodes=3)

    # ----- full dist -----
    dist = l.dist()

    if tl.get_backend() == "numpy":
        mu_list = [d.kwds.get("mu") for d in dist]
    elif tl.get_backend() == "pytorch":
        mu_list = dist.rate
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for l_value, l_dist in zip(l_values, mu_list):
        tc.assertTrue(np.allclose(tl.tensor(l_value), tl_toNumpy(l_dist)))

    # ----- partial dist -----
    dist = l.dist([1, 2])

    if tl.get_backend() == "numpy":
        mu_list = [d.kwds.get("mu") for d in dist]
    elif tl.get_backend() == "pytorch":
        mu_list = dist.rate
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for l_value, l_dist in zip(l_values[1:], mu_list):
        tc.assertTrue(np.allclose(tl.tensor(l_value), tl_toNumpy(l_dist)))

    dist = l.dist([1, 0])

    if tl.get_backend() == "numpy":
        mu_list = [d.kwds.get("mu") for d in dist]
    elif tl.get_backend() == "pytorch":
        mu_list = dist.rate
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for l_value, l_dist in zip(reversed(l_values[:-1]), mu_list):
        tc.assertTrue(np.allclose(tl.tensor(l_value), tl_toNumpy(l_dist)))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    poisson = PoissonLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 0.9, 0.31])
    for backend in backends:
        with tl.backend_context(backend):
            poisson_updated = updateBackend(poisson)
            tc.assertTrue(np.all(poisson.scopes_out == poisson_updated.scopes_out))
            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*poisson.get_params()[0]]),
                    np.array([*poisson_updated.get_params()[0]]),
                )
            )




if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
