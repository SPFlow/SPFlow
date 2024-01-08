import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.node.leaf.cond_poisson import CondPoisson as CondPoissonBase
from spflow.base.structure.general.layer.leaf.cond_poisson import CondPoissonLayer as CondPoissonLayerBase
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure import marginalize
from spflow.torch.structure.general.node.leaf.cond_poisson import CondPoisson as CondPoissonTorch
from spflow.torch.structure.general.layer.leaf.cond_poisson import CondPoissonLayer as CondPoissonLayerTorch
from spflow.torch.structure.general.layer.leaf.cond_poisson import updateBackend

from spflow.structure import AutoLeaf
from spflow.modules.layer import CondPoissonLayer

tc = unittest.TestCase()


def test_layer_initialization(do_for_all_backends):
    torch.set_default_dtype(torch.float64)
    # ----- check attributes after correct initialization -----
    l = CondPoissonLayer(scope=Scope([1], [0]), n_nodes=3)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1], [0]), Scope([1], [0]), Scope([1], [0])]))

    # ---- different scopes -----
    l = CondPoissonLayer(scope=Scope([1], [0]), n_nodes=3)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(ValueError, CondPoissonLayer, Scope([0], [1]), n_nodes=0)

    # ----- invalid scope -----
    tc.assertRaises(ValueError, CondPoissonLayer, Scope([]), n_nodes=3)
    tc.assertRaises(ValueError, CondPoissonLayer, [], n_nodes=3)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
    l = CondPoissonLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

    # -----number of cond_f functions -----
    CondPoissonLayer(
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[lambda data: {"l": 1}, lambda data: {"l": 1}],
    )
    tc.assertRaises(
        ValueError,
        CondPoissonLayer,
        Scope([0], [1]),
        n_nodes=2,
        cond_f=[lambda data: {"l": 1}],
    )


def test_retrieve_params(do_for_all_backends):
    # ----- float/int parameter values -----
    l_value = 0.73
    l = CondPoissonLayer(scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"l": l_value})

    for l_layer_node in l.retrieve_params(tl.tensor([[1]]), DispatchContext()):
        tc.assertTrue(np.allclose(l_layer_node, tl.tensor(l_value)))

    # ----- list parameter values -----
    l_values = [0.17, 0.8, 0.53]
    l = CondPoissonLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        cond_f=lambda data: {"l": l_values},
    )

    for l_layer_node, l_value in zip(l.retrieve_params(tl.tensor([[1]]), DispatchContext()), l_values):
        tc.assertTrue(np.allclose(l_layer_node, l_value))

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
        CondPoissonLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Discrete]),
            ]
        )
    )

    # Poisson feature type class
    tc.assertTrue(
        CondPoissonLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Poisson]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Discrete]),
            ]
        )
    )

    # Poisson feature type instance
    tc.assertTrue(
        CondPoissonLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Poisson(1.0)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Poisson(1.0)]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        CondPoissonLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Poisson(1.0)]),
            ]
        )
    )

    # non-conditional scope
    tc.assertFalse(CondPoissonLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.Discrete])]))

    # multivariate signature
    tc.assertFalse(
        CondPoissonLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Discrete, FeatureTypes.Discrete],
                )
            ]
        )
    )


def test_initialization_from_signatures(do_for_all_backends):
    poisson = CondPoissonLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Discrete]),
        ]
    )
    tc.assertTrue(poisson.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    poisson = CondPoissonLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Poisson]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Poisson]),
        ]
    )
    tc.assertTrue(poisson.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    poisson = CondPoissonLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Poisson(l=1.5)]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Poisson(l=2.0)]),
        ]
    )
    tc.assertTrue(poisson.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    # ----- invalid arguments -----

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondPoissonLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondPoissonLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondPoissonLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [FeatureTypes.Continuous, FeatureTypes.Continuous],
            )
        ],
    )


def test_autoleaf(do_for_all_backends):
    if tl.get_backend() == "numpy":
        CondPoissonInstLayer = CondPoissonLayerBase
    elif tl.get_backend() == "pytorch":
        CondPoissonInstLayer = CondPoissonLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondPoissonLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondPoissonLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Poisson]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Poisson]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    poisson = AutoLeaf(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.Poisson(l=1.5)]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.Poisson(l=2.0)]),
        ]
    )
    tc.assertTrue(isinstance(poisson, CondPoissonInstLayer))
    tc.assertTrue(poisson.scopes_out == [Scope([0], [2]), Scope([1], [2])])


def test_layer_structural_marginalization(do_for_all_backends):
    if tl.get_backend() == "numpy":
        CondPoissonInst = CondPoissonBase
        CondPoissonInstLayer = CondPoissonLayerBase
    elif tl.get_backend() == "pytorch":
        CondPoissonInst = CondPoissonTorch
        CondPoissonInstLayer = CondPoissonLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    # ---------- same scopes -----------

    l = CondPoissonLayer(scope=Scope([1], [0]), n_nodes=2)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

    # ---------- different scopes -----------

    l = CondPoissonLayer(scope=[Scope([1], [2]), Scope([0], [2])])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, CondPoissonInst))
    tc.assertEqual(l_marg.scope, Scope([0], [2]))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, CondPoissonInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])


def test_layer_dist(do_for_all_backends):
    l_values = torch.tensor([0.73, 0.29, 0.5])
    l = CondPoissonLayer(scope=Scope([1], [0]), n_nodes=3)

    # ----- full dist -----
    dist = l.dist(l_values)

    if tl.get_backend() == "numpy":
        mu_list = [d.kwds.get("mu") for d in dist]
    elif tl.get_backend() == "pytorch":
        mu_list = dist.rate
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for l_value, l_dist in zip(l_values, mu_list):
        tc.assertTrue(torch.allclose(l_value, l_dist))

    # ----- partial dist -----
    dist = l.dist(l_values, [1, 2])

    if tl.get_backend() == "numpy":
        mu_list = [d.kwds.get("mu") for d in dist]
    elif tl.get_backend() == "pytorch":
        mu_list = dist.rate
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for l_value, l_dist in zip(l_values[1:], mu_list):
        tc.assertTrue(torch.allclose(l_value, l_dist))

    dist = l.dist(l_values, [1, 0])

    if tl.get_backend() == "numpy":
        mu_list = [d.kwds.get("mu") for d in dist]
    elif tl.get_backend() == "pytorch":
        mu_list = dist.rate
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for l_value, l_dist in zip(reversed(l_values[:-1]), mu_list):
        tc.assertTrue(torch.allclose(l_value, l_dist))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    poisson = CondPoissonLayer(scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])])
    for backend in backends:
        with tl.backend_context(backend):
            poisson_updated = updateBackend(poisson)
            tc.assertTrue(np.all(poisson.scopes_out == poisson_updated.scopes_out))
            # check conversion from torch to python


def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    l_value = 0.73
    cond_poisson_default = CondPoissonLayer(
        scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"l": l_value}
    )

    cond_poisson_updated = CondPoissonLayer(
        scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"l": l_value}
    )
    tc.assertTrue(cond_poisson_default.dtype == tl.float32)
    l = cond_poisson_default.retrieve_params(np.array([[1.0]]), DispatchContext())
    tc.assertTrue(l.dtype == tl.float32)

    # change to float64 model
    cond_poisson_updated.to_dtype(tl.float64)
    l_up = cond_poisson_updated.retrieve_params(np.array([[1.0]]), DispatchContext())
    tc.assertTrue(cond_poisson_updated.dtype == tl.float64)
    tc.assertTrue(l_up.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array(l),
            np.array(l_up),
        )
    )


def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    torch.set_default_dtype(torch.float32)
    l_value = 0.73
    cond_poisson_default = CondPoissonLayer(
        scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"l": l_value}
    )

    cond_poisson_updated = CondPoissonLayer(
        scope=Scope([1], [0]), n_nodes=3, cond_f=lambda data: {"l": l_value}
    )
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, cond_poisson_updated.to_device, cuda)
        return

    # put model on gpu
    cond_poisson_updated.to_device(cuda)

    tc.assertTrue(cond_poisson_default.device.type == "cpu")
    tc.assertTrue(cond_poisson_updated.device.type == "cuda")

    l = cond_poisson_default.retrieve_params(np.array([[1.0]]), DispatchContext())
    l_up = cond_poisson_updated.retrieve_params(np.array([[1.0]]), DispatchContext())

    tc.assertTrue(l.device.type == "cpu")
    tc.assertTrue(l_up.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array(l),
            np.array(l_up.cpu()),
        )
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
