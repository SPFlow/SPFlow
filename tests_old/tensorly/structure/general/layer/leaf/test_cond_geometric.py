import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.node.leaf.cond_geometric import CondGeometric as CondGeometricBase
from spflow.base.structure.general.layer.leaf.cond_geometric import (
    CondGeometricLayer as CondGeometricLayerBase,
)
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure import marginalize
from spflow.torch.structure.general.node.leaf.cond_geometric import CondGeometric as CondGeometricTorch
from spflow.torch.structure.general.layer.leaf.cond_geometric import (
    CondGeometricLayer as CondGeometricLayerTorch,
)
from spflow.torch.structure.general.layer.leaf.cond_geometric import updateBackend

from spflow.structure import AutoLeaf
from spflow.structure.general.layer.leaf import CondGeometricLayer

tc = unittest.TestCase()


def test_layer_initialization(do_for_all_backends):
    # ----- check attributes after correct initialization -----
    l = CondGeometricLayer(scope=Scope([1], [0]), n_nodes=3)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])]))

    # ---- different scopes -----
    l = CondGeometricLayer(scope=Scope([1], [0]), n_nodes=3)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, CondGeometricLayer, Scope([0], [1]), n_nodes=0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, CondGeometricLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, CondGeometricLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
    l = CondGeometricLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

    # -----number of cond_f functions -----
    CondGeometricLayer(
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[lambda data: {"p": 0.5}, lambda data: {"p": 0.5}],
    )
    tc.assertRaises(
        ValueError,
        CondGeometricLayer,
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[lambda data: {"p": 0.5}],
    )


def test_retrieve_params(do_for_all_backends):
    # ----- float/int parameter values -----
    p_value = 0.73
    l = CondGeometricLayer(scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"p": p_value})

    for p_layer_node in l.retrieve_params(tl.tensor([[1]]), DispatchContext()):
        tc.assertTrue(np.allclose(p_layer_node, p_value))

    # ----- list parameter values -----
    p_values = [0.17, 0.8, 0.53]
    l = CondGeometricLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        cond_f=lambda data: {"p": p_values},
    )

    for p_layer_node, p_value in zip(l.retrieve_params(tl.tensor([[1]]), DispatchContext()), p_values):
        tc.assertTrue(np.allclose(p_layer_node, p_value))

    # wrong number of values
    l.set_cond_f(lambda data: {"p": p_values[:-1]})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    # wrong number of dimensions (nested list)
    l.set_cond_f(lambda data: {"p": [p_values for _ in range(3)]})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    # ----- numpy parameter values -----
    l.set_cond_f(lambda data: {"p": np.array(p_values)})
    for p_node, p_actual in zip(
        l.retrieve_params(torch.tensor([[1.0]]), DispatchContext()),
        p_values,
    ):
        tc.assertTrue(p_node == p_actual)

    # wrong number of values
    l.set_cond_f(lambda data: {"p": np.array(p_values[:-1])})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    # wrong number of dimensions (nested list)
    l.set_cond_f(lambda data: {"p": np.array([p_values for _ in range(3)])})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )


def test_accept(do_for_all_backends):
    # discrete meta type
    tc.assertTrue(
        CondGeometricLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Discrete]),
            ]
        )
    )

    # feature type instance
    tc.assertTrue(
        CondGeometricLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Geometric]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Geometric]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        CondGeometricLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Geometric]),
            ]
        )
    )

    # non-conditional scope
    tc.assertFalse(CondGeometricLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Geometric])]))

    # multivariate signature
    tc.assertFalse(
        CondGeometricLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Geometric, FeatureTypes.Geometric],
                )
            ]
        )
    )


def test_initialization_from_signatures(do_for_all_backends):
    geometric = CondGeometricLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Discrete]),
        ]
    )
    tc.assertTrue(geometric.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    geometric = CondGeometricLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Geometric]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Geometric]),
        ]
    )
    tc.assertTrue(geometric.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    geometric = CondGeometricLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Geometric(0.5)]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Geometric(0.5)]),
        ]
    )
    tc.assertTrue(geometric.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondGeometricLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondGeometricLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondGeometricLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Discrete, FeatureTypes.Discrete],
            )
        ],
    )


def test_autoleaf(do_for_all_backends):
    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondGeometricLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondGeometricLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Geometric]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Geometric]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    geometric = AutoLeaf(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Geometric(p=0.75)]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Geometric(p=0.25)]),
        ]
    )
    tc.assertTrue(geometric.scopes_out == [Scope([0], [2]), Scope([1], [2])])


def test_layer_structural_marginalization(do_for_all_backends):
    # ---------- same scopes -----------

    if tl.get_backend() == "numpy":
        CondGeometricInst = CondGeometricBase
        CondGeometricInstLayer = CondGeometricLayerBase
    elif tl.get_backend() == "pytorch":
        CondGeometricInst = CondGeometricTorch
        CondGeometricInstLayer = CondGeometricLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    l = CondGeometricLayer(scope=Scope([1], [0]), n_nodes=2)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

    # ---------- different scopes -----------

    l = CondGeometricLayer(scope=[Scope([1], [2]), Scope([0], [2])])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, CondGeometricInst))
    tc.assertEqual(l_marg.scope, Scope([0], [2]))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, CondGeometricInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])


def test_layer_dist(do_for_all_backends):
    p_values = torch.tensor([0.73, 0.29, 0.5])
    l = CondGeometricLayer(scope=Scope([1], [0]), n_nodes=3)

    # ----- full dist -----
    dist = l.dist(p_values)

    if tl.get_backend() == "numpy":
        p_list = [d.kwds.get("p") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for p_value, p_dist in zip(p_values, p_list):
        tc.assertTrue(np.allclose(p_value, p_dist))

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


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    geometric = CondGeometricLayer(scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])])
    for backend in backends:
        with tl.backend_context(backend):
            geometric_updated = updateBackend(geometric)
            tc.assertTrue(np.all(geometric.scopes_out == geometric_updated.scopes_out))
        # check conversion from torch to python


def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    p_value = 0.73
    cond_geometric_default = CondGeometricLayer(
        scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"p": p_value}
    )
    tc.assertTrue(cond_geometric_default.dtype == tl.float32)
    p = cond_geometric_default.retrieve_params(np.array([[1.0]]), DispatchContext())
    tc.assertTrue(p.dtype == tl.float32)

    # change to float64 model
    cond_geometric_updated = CondGeometricLayer(
        scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"p": p_value}
    )
    cond_geometric_updated.to_dtype(tl.float64)
    p_up = cond_geometric_updated.retrieve_params(np.array([[1.0]]), DispatchContext())
    tc.assertTrue(cond_geometric_updated.dtype == tl.float64)
    tc.assertTrue(p_up.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array(p),
            np.array(p_up),
        )
    )


def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    p_value = 0.13
    torch.set_default_dtype(torch.float32)
    cond_geometric_default = CondGeometricLayer(
        scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"p": p_value}
    )
    cond_geometric_updated = CondGeometricLayer(
        scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"p": p_value}
    )
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, cond_geometric_updated.to_device, cuda)
        return

    # put model on gpu
    cond_geometric_updated.to_device(cuda)

    tc.assertTrue(cond_geometric_default.device.type == "cpu")
    tc.assertTrue(cond_geometric_updated.device.type == "cuda")

    p = cond_geometric_default.retrieve_params(np.array([[1.0]]), DispatchContext())
    p_up = cond_geometric_updated.retrieve_params(np.array([[1.0]]), DispatchContext())

    tc.assertTrue(p.device.type == "cpu")
    tc.assertTrue(p_up.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array(p),
            np.array(p_up.cpu()),
        )
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
