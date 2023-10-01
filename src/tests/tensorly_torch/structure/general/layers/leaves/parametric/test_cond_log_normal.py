import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.nodes.leaves.parametric.cond_log_normal import CondLogNormal as CondLogNormalBase
from spflow.base.structure.general.layers.leaves.parametric.cond_log_normal import CondLogNormalLayer as CondLogNormalLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure import marginalize
from spflow.torch.structure.general.nodes.leaves.parametric.cond_log_normal import CondLogNormal as CondLogNormalTorch
from spflow.torch.structure.general.layers.leaves.parametric.cond_log_normal import CondLogNormalLayer as CondLogNormalLayerTorch
from spflow.torch.structure.general.layers.leaves.parametric.cond_log_normal import updateBackend

from spflow.tensorly.structure import AutoLeaf
from spflow.tensorly.structure.general.layers.leaves.parametric.general_cond_log_normal import CondLogNormalLayer

tc = unittest.TestCase()


def test_layer_initialization(do_for_all_backends):

    # ----- check attributes after correct initialization -----
    l = CondLogNormalLayer(scope=Scope([1], [0]), n_nodes=3)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])]))

    # ---- different scopes -----
    l = CondLogNormalLayer(scope=Scope([1], [0]), n_nodes=3)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, CondLogNormalLayer, Scope([0], [1]), n_nodes=0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, CondLogNormalLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, CondLogNormalLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
    l = CondLogNormalLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

    # -----number of cond_f functions -----
    CondLogNormalLayer(
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[
            lambda data: {"mean": 0.0, "std": 1.0},
            lambda data: {"mean": 0.0, "std": 1.0},
        ],
    )
    tc.assertRaises(
        ValueError,
        CondLogNormalLayer,
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[lambda data: {"mean": 0.0, "std": 1.0}],
    )

def test_retrieve_params(do_for_all_backends):

    # ----- float/int parameter values -----
    mean_value = 0.73
    std_value = 1.9
    l = CondLogNormalLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        cond_f=lambda data: {"mean": mean_value, "std": std_value},
    )

    for mean_layer_node, std_layer_node in zip(*l.retrieve_params(torch.tensor([[1]]), DispatchContext())):
        tc.assertTrue(np.allclose(mean_layer_node, mean_value))
        tc.assertTrue(np.allclose(std_layer_node, std_value))

    # ----- list parameter values -----
    mean_values = [0.17, -0.8, 0.53]
    std_values = [0.9, 1.34, 0.98]
    l = CondLogNormalLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        cond_f=lambda data: {"mean": mean_values, "std": std_values},
    )

    for mean_value, std_value, mean_layer_node, std_layer_node in zip(
        mean_values, std_values, *l.retrieve_params(torch.tensor([[1]]), DispatchContext())
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
        torch.tensor([[1]]),
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
        CondLogNormalLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # feature type class
    tc.assertTrue(
        CondLogNormalLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.LogNormal]),
                FeatureContext(Scope([1], [3]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # feature type instance
    tc.assertTrue(
        CondLogNormalLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.LogNormal(0.0, 1.0)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        CondLogNormalLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
            ]
        )
    )

    # non-conditional scope
    tc.assertFalse(CondLogNormalLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

    # multivariate signature
    tc.assertFalse(
        CondLogNormalLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ]
        )
    )

def test_initialization_from_signatures(do_for_all_backends):

    log_normal = CondLogNormalLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Continuous]),
        ]
    )
    tc.assertTrue(log_normal.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    log_normal = CondLogNormalLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.LogNormal]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.LogNormal]),
        ]
    )
    tc.assertTrue(log_normal.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    log_normal = CondLogNormalLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.LogNormal(0.0, 1.0)]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.LogNormal(0.0, 1.0)]),
        ]
    )
    tc.assertTrue(log_normal.scopes_out == [Scope([0], [2]), Scope([1], [2])])
    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondLogNormalLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondLogNormalLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondLogNormalLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )

def test_autoleaf(do_for_all_backends):

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondLogNormalLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondLogNormalLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.LogNormal]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.LogNormal]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    log_normal = AutoLeaf(
        [
            FeatureContext(
                Scope([0], [2]),
                [FeatureTypes.LogNormal(mean=-1.0, std=1.5)],
            ),
            FeatureContext(Scope([1], [2]), [FeatureTypes.LogNormal(mean=1.0, std=0.5)]),
        ]
    )
    tc.assertTrue(log_normal.scopes_out == [Scope([0], [2]), Scope([1], [2])])

def test_layer_structural_marginalization(do_for_all_backends):

    # ---------- same scopes -----------

    if tl.get_backend() == "numpy":
        CondLogNormalInst = CondLogNormalBase
        CondLogNormalInstLayer = CondLogNormalLayerBase
    elif tl.get_backend() == "pytorch":
        CondLogNormalInst = CondLogNormalTorch
        CondLogNormalInstLayer = CondLogNormalLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    l = CondLogNormalLayer(scope=Scope([1], [0]), n_nodes=2)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

    # ---------- different scopes -----------

    l = CondLogNormalLayer(scope=[Scope([1], [2]), Scope([0], [2])])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, CondLogNormalInst))
    tc.assertEqual(l_marg.scope, Scope([0], [2]))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, CondLogNormalInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])

def test_layer_dist(do_for_all_backends):

    mean_values = tl.tensor([0.73, -0.29, 0.5])
    std_values = tl.tensor([0.9, 1.34, 0.98])
    l = CondLogNormalLayer(scope=Scope([1], [0]), n_nodes=3)

    # ----- full dist -----
    dist = l.dist(mean_values, std_values)

    if tl.get_backend() == "numpy":
        loc_list = [np.log(d.kwds.get("scale")) for d in dist]
        scale_list = [d.kwds.get("s") for d in dist]
    elif tl.get_backend() == "pytorch":
        loc_list = dist.loc
        scale_list = dist.scale
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for mean_value, std_value, mean_dist, std_dist in zip(mean_values, std_values, loc_list, scale_list):
        tc.assertTrue(np.allclose(mean_value, mean_dist))
        tc.assertTrue(np.allclose(std_value, std_dist))

    # ----- partial dist -----
    dist = l.dist(mean_values, std_values, [1, 2])

    if tl.get_backend() == "numpy":
        loc_list = [np.log(d.kwds.get("scale")) for d in dist]
        scale_list = [d.kwds.get("s") for d in dist]
    elif tl.get_backend() == "pytorch":
        loc_list = dist.loc
        scale_list = dist.scale
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for mean_value, std_value, mean_dist, std_dist in zip(mean_values[1:], std_values[1:], loc_list, scale_list):
        tc.assertTrue(np.allclose(mean_value, mean_dist))
        tc.assertTrue(np.allclose(std_value, std_dist))

    dist = l.dist(mean_values, std_values, [1, 0])

    if tl.get_backend() == "numpy":
        loc_list = [np.log(d.kwds.get("scale")) for d in dist]
        scale_list = [d.kwds.get("s") for d in dist]
    elif tl.get_backend() == "pytorch":
        loc_list = dist.loc
        scale_list = dist.scale
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for mean_value, std_value, mean_dist, std_dist in zip(
        reversed(mean_values[:-1]),
        reversed(std_values[:-1]),
        loc_list,
        scale_list,
    ):
        tc.assertTrue(np.allclose(mean_value, mean_dist))
        tc.assertTrue(np.allclose(std_value, std_dist))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    logNormal = CondLogNormalLayer(scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])])
    for backend in backends:
        with tl.backend_context(backend):
            logNormal_updated = updateBackend(logNormal)
            tc.assertTrue(np.all(logNormal.scopes_out == logNormal_updated.scopes_out))
            # check conversion from torch to python



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
