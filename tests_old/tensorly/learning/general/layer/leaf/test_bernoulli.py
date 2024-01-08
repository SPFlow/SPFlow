import random
import unittest

import numpy as np
import pytest
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.modules.module import log_likelihood
from spflow.torch.learning import (
    em,
    expectation_maximization,
)
from spflow.learning import maximum_likelihood_estimation
from spflow.structure.spn import BernoulliLayer
from spflow.structure.spn import ProductNode, SumNode
from spflow.tensor import ops as tle
from spflow.torch.structure.general.layer.leaf.bernoulli import updateBackend

tc = unittest.TestCase()


def test_mle(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = BernoulliLayer(scope=[Scope([0]), Scope([1])])

    # simulate data
    data = np.hstack(
        [
            np.random.binomial(n=1, p=0.3, size=(10000, 1)),
            np.random.binomial(n=1, p=0.7, size=(10000, 1)),
        ]
    )

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data))

    tc.assertTrue(np.allclose(tle.toNumpy(layer.p), tl.tensor([0.3, 0.7]), atol=1e-2, rtol=1e-3))


def test_mle_edge_0(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = BernoulliLayer(Scope([0]), n_nodes=1)

    # simulate data
    data = np.random.binomial(n=1, p=0.0, size=(100, 1))

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data))

    tc.assertTrue(tl.all(layer.p > 0.0))


def test_mle_edge_1(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = BernoulliLayer(Scope([0]), n_nodes=1)

    # simulate data
    data = np.random.binomial(n=1, p=1.0, size=(100, 1))

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data))

    tc.assertTrue(tl.all(layer.p <= 1.0))


def test_mle_only_nans(do_for_all_backends):
    layer = BernoulliLayer(scope=[Scope([0]), Scope([1])], n_nodes=1)

    # simulate data
    data = tl.tensor([[float("nan"), float("nan")], [float("nan"), 1.0]])

    # check if exception is raised
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        data,
        nan_strategy="ignore",
        bias_correction=False,
    )


def test_mle_invalid_support(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = BernoulliLayer(Scope([0]))

    # perform MLE (should raise exceptions)
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        tl.tensor([[float("inf")]]),
        bias_correction=True,
    )
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        tl.tensor([[-0.1]]),
        bias_correction=True,
    )
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        tl.tensor([[2]]),
        bias_correction=True,
    )


def test_mle_nan_strategy_none(do_for_all_backends):
    layer = BernoulliLayer(Scope([0]))
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        tl.tensor([[float("nan")], [1], [0], [1]]),
        nan_strategy=None,
    )


def test_mle_nan_strategy_ignore(do_for_all_backends):
    layer = BernoulliLayer(Scope([0]))
    maximum_likelihood_estimation(
        layer,
        tl.tensor([[float("nan")], [1], [0], [1]]),
        nan_strategy="ignore",
    )
    tc.assertTrue(np.allclose(tle.toNumpy(layer.p), tl.tensor(2.0 / 3.0)))


def test_mle_nan_strategy_callable(do_for_all_backends):
    layer = BernoulliLayer(Scope([0]))
    # should not raise an issue
    maximum_likelihood_estimation(layer, tl.tensor([[1], [0], [1]]), nan_strategy=lambda x: x)


def test_mle_nan_strategy_invalid(do_for_all_backends):
    layer = BernoulliLayer(Scope([0]))
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        tl.tensor([[float("nan")], [1], [0], [1]]),
        nan_strategy="invalid_string",
    )
    tc.assertRaises(
        ValueError,
        maximum_likelihood_estimation,
        layer,
        tl.tensor([[float("nan")], [1], [0], [1]]),
        nan_strategy=1,
    )


def test_weighted_mle(do_for_all_backends):
    leaf = BernoulliLayer([Scope([0]), Scope([1])], n_nodes=3)

    data = tl.tensor(
        np.hstack(
            [
                np.vstack(
                    [
                        np.random.binomial(n=1, p=0.8, size=(10000, 1)),
                        np.random.binomial(n=1, p=0.2, size=(10000, 1)),
                    ]
                ),
                np.vstack(
                    [
                        np.random.binomial(n=1, p=0.3, size=(10000, 1)),
                        np.random.binomial(n=1, p=0.7, size=(10000, 1)),
                    ]
                ),
            ]
        )
    )

    weights = tl.concatenate([tl.zeros(10000), tl.ones(10000)])

    maximum_likelihood_estimation(leaf, data, weights)

    tc.assertTrue(np.allclose(tle.toNumpy(leaf.p), tl.tensor([0.2, 0.7]), atol=1e-2, rtol=1e-2))


def test_em_step(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = BernoulliLayer([Scope([0]), Scope([1])])
    data = tl.tensor(
        np.hstack(
            [
                np.random.binomial(n=1, p=0.2, size=(10000, 1)),
                np.random.binomial(n=1, p=0.7, size=(10000, 1)),
            ]
        )
    )
    dispatch_ctx = DispatchContext()

    # compute gradients of log-likelihoods w.r.t. module log-likelihoods
    ll = log_likelihood(layer, data, dispatch_ctx=dispatch_ctx)
    ll.retain_grad()
    ll.sum().backward()

    # perform an em step
    em(layer, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tle.toNumpy(layer.p), tl.tensor([0.2, 0.7]), atol=1e-2, rtol=1e-3))


def test_em_product_of_bernoullis(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = BernoulliLayer([Scope([0]), Scope([1])])
    prod_node = ProductNode([layer])

    data = tl.tensor(
        np.hstack(
            [
                np.random.binomial(n=1, p=0.8, size=(10000, 1)),
                np.random.binomial(n=1, p=0.2, size=(10000, 1)),
            ]
        )
    )

    expectation_maximization(prod_node, data, max_steps=10)

    tc.assertTrue(np.allclose(tle.toNumpy(layer.p), tl.tensor([0.8, 0.2]), atol=1e-3, rtol=1e-2))


def test_em_sum_of_bernoullis(do_for_all_backends):
    # em is only implemented for pytorch backend
    if do_for_all_backends == "numpy":
        return

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    leaf = BernoulliLayer(Scope([0]), n_nodes=2, p=[0.4, 0.6])
    sum_node = SumNode([leaf], weights=[0.5, 0.5])

    data = tl.tensor(
        np.vstack(
            [
                np.random.binomial(n=1, p=0.8, size=(1000, 1)),
                np.random.binomial(n=1, p=0.3, size=(1000, 1)),
            ]
        )
    )

    expectation_maximization(sum_node, data, max_steps=10)

    # optimal p
    p_opt = tl.sum(data, axis=0) / data.shape[0]
    # total p represented by mixture
    p_em = tl.sum(sum_node.weights * leaf.p)

    tc.assertTrue(np.allclose(p_opt, tle.toNumpy(p_em)))


def test_update_backend(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = BernoulliLayer(scope=[Scope([0]), Scope([1])])

    # simulate data
    data = np.hstack(
        [
            np.random.binomial(n=1, p=0.3, size=(10000, 1)),
            np.random.binomial(n=1, p=0.7, size=(10000, 1)),
        ]
    )

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data))

    params = tle.toNumpy(layer.p)

    layer = BernoulliLayer(scope=[Scope([0]), Scope([1])])
    prod_node = ProductNode([layer])
    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
    params_em = tle.toNumpy(layer.p)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer = BernoulliLayer(scope=[Scope([0]), Scope([1])])
            layer_updated = updateBackend(layer)
            maximum_likelihood_estimation(layer_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tle.toNumpy(layer_updated.p), params, atol=1e-2, rtol=1e-3))

            layer = BernoulliLayer(scope=[Scope([0]), Scope([1])])
            layer_updated = updateBackend(layer)
            prod_node = ProductNode([layer_updated])
            if tl.get_backend() != "pytorch":
                with pytest.raises(NotImplementedError):
                    expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
            else:
                expectation_maximization(prod_node, tl.tensor(data), max_steps=10)
                tc.assertTrue(np.allclose(tle.toNumpy(layer_updated.p), params_em, atol=1e-3, rtol=1e-2))


def test_change_dtype(do_for_all_backends):
    np.random.seed(0)
    random.seed(0)

    layer = BernoulliLayer(scope=[Scope([0]), Scope([1])])
    prod_node = ProductNode([layer])

    # simulate data
    data = np.hstack(
        [
            np.random.binomial(n=1, p=0.3, size=(10000, 1)),
            np.random.binomial(n=1, p=0.7, size=(10000, 1)),
        ]
    )

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data, dtype=tl.float32))
    tc.assertTrue(layer.p.dtype == tl.float32)

    layer.to_dtype(tl.float64)

    dummy_data = tl.tensor(data, dtype=tl.float64)
    maximum_likelihood_estimation(layer, dummy_data)
    tc.assertTrue(layer.p.dtype == tl.float64)

    if do_for_all_backends == "numpy":
        tc.assertRaises(
            NotImplementedError,
            expectation_maximization,
            prod_node,
            tl.tensor(data, dtype=tl.float64),
            max_steps=10,
        )
    else:
        # test if em runs without error after dype change
        expectation_maximization(prod_node, tl.tensor(data, dtype=tl.float64), max_steps=10)


def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    np.random.seed(0)
    random.seed(0)

    layer = BernoulliLayer(scope=[Scope([0]), Scope([1])])
    prod_node = ProductNode([layer])

    # simulate data
    data = np.hstack(
        [
            np.random.binomial(n=1, p=0.3, size=(10000, 1)),
            np.random.binomial(n=1, p=0.7, size=(10000, 1)),
        ]
    )

    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer.to_device, cuda)
        return

    # perform MLE
    maximum_likelihood_estimation(layer, tl.tensor(data, dtype=tl.float32))

    tc.assertTrue(layer.p.device.type == "cpu")

    layer.to_device(cuda)

    dummy_data = tl.tensor(data, dtype=tl.float32, device=cuda)

    # perform MLE
    maximum_likelihood_estimation(layer, dummy_data)
    tc.assertTrue(layer.p.device.type == "cuda")

    # test if em runs without error after device change
    expectation_maximization(prod_node, tl.tensor(data, dtype=tl.float32, device=cuda), max_steps=10)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()