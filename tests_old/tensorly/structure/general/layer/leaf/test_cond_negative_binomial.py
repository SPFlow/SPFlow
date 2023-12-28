import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.base.structure.general.node.leaf.cond_negative_binomial import (
    CondNegativeBinomial as CondNegativeBinomialBase,
)
from spflow.base.structure.general.layer.leaf.cond_negative_binomial import (
    CondNegativeBinomialLayer as CondNegativeBinomialLayerBase,
)
from spflow.meta.data import FeatureContext, FeatureTypes, Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.structure import marginalize
from spflow.torch.structure.general.node.leaf.cond_negative_binomial import (
    CondNegativeBinomial as CondNegativeBinomialTorch,
)
from spflow.torch.structure.general.layer.leaf.cond_negative_binomial import (
    CondNegativeBinomialLayer as CondNegativeBinomialLayerTorch,
)
from spflow.torch.structure.general.layer.leaf.cond_negative_binomial import updateBackend

from spflow.structure import AutoLeaf
from spflow.modules.layer import CondNegativeBinomialLayer

tc = unittest.TestCase()


def test_layer_initialization(do_for_all_backends):
    # ----- check attributes after correct initialization -----
    l = CondNegativeBinomialLayer(scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])], n=2)
    # make sure number of creates nodes is correct
    tc.assertEqual(len(l.scopes_out), 3)
    # make sure scopes are correct
    tc.assertTrue(np.all(l.scopes_out == [Scope([1], [3]), Scope([0], [3]), Scope([2], [3])]))

    # ----- n initialization -----
    l = CondNegativeBinomialLayer(
        scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
        n=[3, 5, 2],
    )
    # wrong number of n values
    tc.assertRaises(
        ValueError,
        CondNegativeBinomialLayer,
        [Scope([1]), Scope([0]), Scope([2])],
        n=[3, 5],
        n_nodes=3,
    )
    # wrong shape of n values
    tc.assertRaises(
        ValueError,
        CondNegativeBinomialLayer,
        [Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
        [[3, 5] for _ in range(3)],
        n_nodes=3,
    )

    # n numpy array
    l = CondNegativeBinomialLayer(
        scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
        n=np.array([3, 5, 2]),
    )
    # wrong number of n values
    tc.assertRaises(
        ValueError,
        CondNegativeBinomialLayer,
        scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
        n=np.array([3, 5]),
    )
    # wrong shape of n values
    tc.assertRaises(
        ValueError,
        CondNegativeBinomialLayer,
        scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
        n=np.array([[3, 5, 2]]),
    )

    # ---- different scopes -----
    l = CondNegativeBinomialLayer(scope=Scope([1], [0]), n_nodes=3, n=2)
    for layer_scope, node_scope in zip(l.scopes_out, l.scopes_out):
        tc.assertEqual(layer_scope, node_scope)

    # ----- invalid number of nodes -----
    tc.assertRaises(
        ValueError,
        CondNegativeBinomialLayer,
        Scope([0], [1]),
        n_nodes=0,
        n=2,
    )

    # ----- invalid scope -----
    tc.assertRaises(ValueError, CondNegativeBinomialLayer, Scope([]), n_nodes=3, n=2)
    tc.assertRaises(ValueError, CondNegativeBinomialLayer, [], n_nodes=3, n=2)

    # ----- individual scopes and parameters -----
    scopes = [Scope([1], [2]), Scope([0], [2]), Scope([0], [2])]
    l = CondNegativeBinomialLayer(scope=[Scope([1], [2]), Scope([0], [2])], n_nodes=3, n=2)

    for layer_scope, node_scope in zip(l.scopes_out, scopes):
        tc.assertEqual(layer_scope, node_scope)

    # -----number of cond_f functions -----
    CondNegativeBinomialLayer(
        Scope([0], [1]),
        n=3,
        n_nodes=2,
        cond_f=[lambda data: {"p": 0.5}, lambda data: {"p": 0.5}],
    )
    tc.assertRaises(
        ValueError,
        CondNegativeBinomialLayer,
        Scope([0], [1]),
        n=3,
        n_nodes=2,
        cond_f=[lambda data: {"p": 0.5}],
    )


def test_retrieve_params(do_for_all_backends):
    # ----- float/int parameter values -----
    n_value = 5
    p_value = 0.13
    l = CondNegativeBinomialLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        n=n_value,
        cond_f=lambda data: {"p": p_value},
    )

    for n_layer_node, p_layer_node in zip(l.n, l.retrieve_params(tl.tensor([[1]]), DispatchContext())):
        tc.assertTrue(tl.all(n_layer_node == n_value))
        tc.assertTrue(np.allclose(p_layer_node, p_value))

    # ----- list parameter values -----
    n_values = [3, 2, 7]
    p_values = [0.17, 0.8, 0.53]
    l = CondNegativeBinomialLayer(
        scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
        n=n_values,
        cond_f=lambda data: {"p": p_values},
    )

    for n_value, p_value, n_layer_node, p_layer_node in zip(
        n_values,
        p_values,
        l.n,
        l.retrieve_params(tl.tensor([[1]]), DispatchContext()),
    ):
        tc.assertTrue(np.allclose(n_layer_node, n_value))
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
        l.retrieve_params(tl.tensor([[1.0]]), DispatchContext()),
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

    l.set_cond_f(lambda data: {"p": np.expand_dims(np.array(p_values), 0)})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )

    l.set_cond_f(lambda data: {"p": np.expand_dims(np.array(p_values), 1)})
    tc.assertRaises(
        ValueError,
        l.retrieve_params,
        tl.tensor([[1]]),
        DispatchContext(),
    )


def test_accept(do_for_all_backends):
    # discrete meta type (should reject)
    tc.assertFalse(
        CondNegativeBinomialLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Discrete]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.Discrete]),
            ]
        )
    )

    # feature type instance
    tc.assertTrue(
        CondNegativeBinomialLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.NegativeBinomial(n=3)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.NegativeBinomial(n=3)]),
            ]
        )
    )

    # invalid feature type
    tc.assertFalse(
        CondNegativeBinomialLayer.accepts(
            [
                FeatureContext(Scope([0], [2]), [FeatureTypes.Continuous]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.NegativeBinomial(n=3)]),
            ]
        )
    )

    # non-conditional scope
    tc.assertFalse(
        CondNegativeBinomialLayer.accepts([FeatureContext(Scope([0]), [FeatureTypes.NegativeBinomial(n=3)])])
    )

    # multivariate signature
    tc.assertFalse(
        CondNegativeBinomialLayer.accepts(
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [
                        FeatureTypes.NegativeBinomial(n=3),
                        FeatureTypes.Binomial(n=3),
                    ],
                )
            ]
        )
    )


def test_initialization_from_signatures(do_for_all_backends):
    negative_binomial = CondNegativeBinomialLayer.from_signatures(
        [
            FeatureContext(Scope([0], [2]), [FeatureTypes.NegativeBinomial(n=3)]),
            FeatureContext(Scope([1], [2]), [FeatureTypes.NegativeBinomial(n=5)]),
        ]
    )
    tc.assertTrue(negative_binomial.scopes_out == [Scope([0], [2]), Scope([1], [2])])

    # ----- invalid arguments -----

    # discrete meta type
    tc.assertRaises(
        ValueError,
        CondNegativeBinomialLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
    )

    # invalid feature type
    tc.assertRaises(
        ValueError,
        CondNegativeBinomialLayer.from_signatures,
        [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
    )

    # non-conditional scope
    tc.assertRaises(
        ValueError,
        CondNegativeBinomialLayer.from_signatures,
        [FeatureContext(Scope([0]), [FeatureTypes.NegativeBinomial(3)])],
    )

    # multivariate signature
    tc.assertRaises(
        ValueError,
        CondNegativeBinomialLayer.from_signatures,
        [
            FeatureContext(
                Scope([0, 1], [2]),
                [
                    FeatureTypes.NegativeBinomial(3),
                    FeatureTypes.NegativeBinomial(5),
                ],
            )
        ],
    )


def test_autoleaf(do_for_all_backends):
    # make sure leaf is registered
    tc.assertTrue(AutoLeaf.is_registered(CondNegativeBinomialLayer))

    # make sure leaf is correctly inferred
    tc.assertEqual(
        CondNegativeBinomialLayer,
        AutoLeaf.infer(
            [
                FeatureContext(Scope([0], [1]), [FeatureTypes.NegativeBinomial(n=3)]),
                FeatureContext(Scope([1], [2]), [FeatureTypes.NegativeBinomial(n=5)]),
            ]
        ),
    )

    # make sure AutoLeaf can return correctly instantiated object
    negative_binomial = AutoLeaf(
        [
            FeatureContext(
                Scope([0], [2]),
                [FeatureTypes.NegativeBinomial(n=3, p=0.75)],
            ),
            FeatureContext(
                Scope([1], [2]),
                [FeatureTypes.NegativeBinomial(n=5, p=0.25)],
            ),
        ]
    )
    tc.assertTrue(negative_binomial.scopes_out == [Scope([0], [2]), Scope([1], [2])])


def test_layer_structural_marginalization(do_for_all_backends):
    # ---------- same scopes -----------

    if tl.get_backend() == "numpy":
        CondNegativeBinomialInst = CondNegativeBinomialBase
        CondNegativeBinomialInstLayer = CondNegativeBinomialLayerBase
    elif tl.get_backend() == "pytorch":
        CondNegativeBinomialInst = CondNegativeBinomialTorch
        CondNegativeBinomialInstLayer = CondNegativeBinomialLayerTorch
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    l = CondNegativeBinomialLayer(scope=Scope([1], [0]), n_nodes=2, n=2)

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [1]) == None)

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [0]), Scope([1], [0])])

    # ---------- different scopes -----------

    l = CondNegativeBinomialLayer(scope=[Scope([1], [2]), Scope([0], [2])], n=[3, 2])

    # ----- marginalize over entire scope -----
    tc.assertTrue(marginalize(l, [0, 1]) == None)

    # ----- partially marginalize -----
    l_marg = marginalize(l, [1], prune=True)
    tc.assertTrue(isinstance(l_marg, CondNegativeBinomialInst))
    tc.assertEqual(l_marg.scope, Scope([0], [2]))
    tc.assertTrue(np.allclose(l_marg.n, tl.tensor(2)))

    l_marg = marginalize(l, [1], prune=False)
    tc.assertTrue(isinstance(l_marg, CondNegativeBinomialInstLayer))
    tc.assertEqual(len(l_marg.scopes_out), 1)
    tc.assertTrue(np.allclose(l_marg.n, tl.tensor(2)))

    # ----- marginalize over non-scope rvs -----
    l_marg = marginalize(l, [2])

    tc.assertTrue(l_marg.scopes_out == [Scope([1], [2]), Scope([0], [2])])
    tc.assertTrue(np.allclose(l.n, l_marg.n))


def test_layer_dist(do_for_all_backends):
    n_values = [3, 2, 7]
    p_values = tl.tensor([0.73, 0.29, 0.5], dtype=tl.float64)
    l = CondNegativeBinomialLayer(
        scope=[Scope([1], [3]), Scope([0], [3]), Scope([2], [3])],
        n=n_values,
    )

    # ----- full dist -----
    dist = l.dist(p_values)

    if tl.get_backend() == "numpy":
        p_list = [1 - d.kwds.get("p") for d in dist]
        n_list = [d.kwds.get("n") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
        n_list = dist.total_count
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for n_value, p_value, n_dist, p_dist in zip(n_values, p_values, n_list, p_list):
        tc.assertTrue(np.allclose(n_value, n_dist))
        tc.assertTrue(np.allclose(1 - p_value, p_dist))

    # ----- partial dist -----
    dist = l.dist(p_values, [1, 2])

    if tl.get_backend() == "numpy":
        p_list = [1 - d.kwds.get("p") for d in dist]
        n_list = [d.kwds.get("n") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
        n_list = dist.total_count
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for n_value, p_value, n_dist, p_dist in zip(n_values[1:], p_values[1:], n_list, p_list):
        tc.assertTrue(np.allclose(n_value, n_dist))
        tc.assertTrue(np.allclose(1 - p_value, p_dist))

    dist = l.dist(p_values, [1, 0])

    if tl.get_backend() == "numpy":
        p_list = [1 - d.kwds.get("p") for d in dist]
        n_list = [d.kwds.get("n") for d in dist]
    elif tl.get_backend() == "pytorch":
        p_list = dist.probs
        n_list = dist.total_count
    else:
        raise NotImplementedError("This test is not implemented for this backend")

    for n_value, p_value, n_dist, p_dist in zip(
        reversed(n_values[:-1]),
        reversed(p_values[:-1]),
        n_list,
        p_list,
    ):
        tc.assertTrue(np.allclose(n_value, n_dist))
        tc.assertTrue(np.allclose(1 - p_value, p_dist))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    negativeBinomial = CondNegativeBinomialLayer(
        scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])], n=[2, 5, 2]
    )
    for backend in backends:
        with tl.backend_context(backend):
            negativeBinomial_updated = updateBackend(negativeBinomial)
            tc.assertTrue(np.all(negativeBinomial.scopes_out == negativeBinomial_updated.scopes_out))
            # check conversion from torch to python
            tc.assertTrue(
                np.allclose(
                    np.array([*negativeBinomial.get_params()[0]]),
                    np.array([*negativeBinomial_updated.get_params()[0]]),
                )
            )


def test_change_dtype(do_for_all_backends):
    # create float32 model
    torch.set_default_dtype(torch.float32)
    n_value = 5
    p_value = 0.13
    cond_binomial_default = CondNegativeBinomialLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        n=n_value,
        cond_f=lambda data: {"p": p_value},
    )

    cond_binomial_updated = CondNegativeBinomialLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        n=n_value,
        cond_f=lambda data: {"p": p_value},
    )
    tc.assertTrue(cond_binomial_default.dtype == tl.float32)
    p = cond_binomial_default.retrieve_params(np.array([[1.0]]), DispatchContext())
    tc.assertTrue(p.dtype == tl.float32)
    tc.assertTrue(cond_binomial_default.n.dtype == tl.float32)

    # change to float64 model
    cond_binomial_updated.to_dtype(tl.float64)
    p_up = cond_binomial_updated.retrieve_params(np.array([[1.0]]), DispatchContext())
    tc.assertTrue(cond_binomial_updated.dtype == tl.float64)
    tc.assertTrue(p_up.dtype == tl.float64)
    tc.assertTrue(cond_binomial_updated.n.dtype == tl.float64)
    tc.assertTrue(
        np.allclose(
            np.array(p),
            np.array(p_up),
        )
    )
    tc.assertTrue(
        np.allclose(
            np.array(cond_binomial_default.n),
            np.array(cond_binomial_updated.n),
        )
    )


def test_change_device(do_for_all_backends):
    cuda = torch.device("cuda")
    # create model on cpu
    torch.set_default_dtype(torch.float32)
    n_value = 5
    p_value = 0.13
    cond_binomial_default = CondNegativeBinomialLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        n=n_value,
        cond_f=lambda data: {"p": p_value},
    )

    cond_binomial_updated = CondNegativeBinomialLayer(
        scope=Scope([1], [0]),
        n_nodes=3,
        n=n_value,
        cond_f=lambda data: {"p": p_value},
    )
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, cond_binomial_updated.to_device, cuda)
        return

    # put model on gpu
    cond_binomial_updated.to_device(cuda)

    tc.assertTrue(cond_binomial_default.device.type == "cpu")
    tc.assertTrue(cond_binomial_updated.device.type == "cuda")

    p = cond_binomial_default.retrieve_params(np.array([[1.0]]), DispatchContext())
    p_up = cond_binomial_updated.retrieve_params(np.array([[1.0]]), DispatchContext())

    tc.assertTrue(p.device.type == "cpu")
    tc.assertTrue(p_up.device.type == "cuda")

    tc.assertTrue(cond_binomial_default.n.device.type == "cpu")
    tc.assertTrue(cond_binomial_updated.n.device.type == "cuda")

    tc.assertTrue(
        np.allclose(
            np.array(p),
            np.array(p_up.cpu()),
        )
    )
    tc.assertTrue(
        np.allclose(
            np.array(cond_binomial_default.n),
            np.array(cond_binomial_updated.n.cpu()),
        )
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
