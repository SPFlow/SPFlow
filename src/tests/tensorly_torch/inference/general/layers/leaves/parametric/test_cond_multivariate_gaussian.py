import random
import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.general.layers.leaves.parametric.general_cond_multivariate_gaussian import CondMultivariateGaussianLayer
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_multivariate_gaussian import CondMultivariateGaussian
from spflow.torch.structure.general.layers.leaves.parametric.cond_multivariate_gaussian import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_likelihood_no_mean(do_for_all_backends):

    multivariate_gaussian = CondMultivariateGaussianLayer(
        Scope([0, 1], [2]),
        cond_f=lambda data: {"cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]},
        n_nodes=2,
    )
    tc.assertRaises(
        KeyError,
        log_likelihood,
        multivariate_gaussian,
        tl.tensor([[0], [1]]),
    )

def test_likelihood_no_cov(do_for_all_backends):

    multivariate_gaussian = CondMultivariateGaussianLayer(
        Scope([0, 1], [2]),
        cond_f=lambda data: {"mean": [[0.0, 0.0], [0.0, 0.0]]},
        n_nodes=2,
    )
    tc.assertRaises(
        KeyError,
        log_likelihood,
        multivariate_gaussian,
        tl.tensor([[0], [1]]),
    )

def test_likelihood_no_mean_cov(do_for_all_backends):

    multivariate_gaussian = CondMultivariateGaussianLayer(Scope([0], [1]), n_nodes=2)
    tc.assertRaises(
        ValueError,
        log_likelihood,
        multivariate_gaussian,
        tl.tensor([[0], [1]]),
    )

def test_likelihood_module_cond_f(do_for_all_backends):

    cond_f = lambda data: {
        "mean": [[0.0, 0.0], [0.0, 0.0]],
        "cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
    }

    multivariate_gaussian = CondMultivariateGaussianLayer(Scope([0, 1], [2]), n_nodes=2, cond_f=cond_f)

    # create test inputs/outputs
    data = tl.stack([tl.zeros(2), tl.ones(2)], axis=0)
    targets = tl.tensor([[0.1591549, 0.1591549], [0.0585498, 0.0585498]])

    probs = likelihood(multivariate_gaussian, data)
    log_probs = log_likelihood(multivariate_gaussian, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args(do_for_all_backends):

    multivariate_gaussian = CondMultivariateGaussianLayer(Scope([0, 1], [2]), n_nodes=2)

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[multivariate_gaussian] = {
        "mean": [[0.0, 0.0], [0.0, 0.0]],
        "cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
    }

    # create test inputs/outputs
    data = tl.stack([tl.zeros(2), tl.ones(2)], axis=0)
    targets = tl.tensor([[0.1591549, 0.1591549], [0.0585498, 0.0585498]])

    probs = likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_cond_f(do_for_all_backends):

    multivariate_gaussian = CondMultivariateGaussianLayer(Scope([0, 1], [2]), n_nodes=2)

    cond_f = lambda data: {
        "mean": [[0.0, 0.0], [0.0, 0.0]],
        "cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
    }

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[multivariate_gaussian] = {"cond_f": cond_f}

    # create test inputs/outputs
    data = tl.stack([tl.zeros(2), tl.ones(2)], axis=0)
    targets = tl.tensor([[0.1591549, 0.1591549], [0.0585498, 0.0585498]])

    probs = likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_layer_likelihood(do_for_all_backends):

    mean_values = [
        tl.zeros(2),
        tl.arange(3, dtype=tl.float64),
    ]
    cov_values = [
        tl.eye(2),
        tl.tensor(
            [
                [2, 2, 1],
                [2, 3, 2],
                [1, 2, 3],
            ],
            dtype=tl.float64,
        ),
    ]

    layer = CondMultivariateGaussianLayer(
        scope=[Scope([0, 1], [5]), Scope([2, 3, 4], [5])],
        cond_f=lambda data: {"mean": mean_values, "cov": cov_values},
    )

    nodes = [
        CondMultivariateGaussian(
            Scope([0, 1], [5]),
            cond_f=lambda data: {
                "mean": mean_values[0],
                "cov": cov_values[0],
            },
        ),
        CondMultivariateGaussian(
            Scope([2, 3, 4], [5]),
            cond_f=lambda data: {
                "mean": mean_values[1],
                "cov": cov_values[1],
            },
        ),
    ]

    dummy_data = tl.tensor(np.vstack([tl.zeros(5), tl.ones(5)]), dtype=tl.float64)

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(nodes_ll)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    mean = [
        torch.zeros(2, dtype=tl.float64, requires_grad=True),
        torch.arange(3, dtype=tl.float64, requires_grad=True),
    ]
    cov = [
        torch.eye(2, requires_grad=True),
        torch.tensor(
            [[2, 2, 1], [2, 3, 2], [1, 2, 3]],
            dtype=tl.float64,
            requires_grad=True,
        ),
    ]

    torch_multivariate_gaussian = CondMultivariateGaussianLayer(
        scope=[Scope([0, 1], [5]), Scope([2, 3, 4], [5])],
        cond_f=lambda data: {"mean": mean, "cov": cov},
    )

    # create dummy input data (batch size x random variables)
    data = tl.randn((3, 5))

    log_probs_torch = log_likelihood(torch_multivariate_gaussian, data)

    # create dummy targets
    targets_torch = tl.ones((3, 2))

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(all([m.grad is not None for m in mean]))
    tc.assertTrue(all([c.grad is not None for c in cov]))

def test_likelihood_marginalization(do_for_all_backends):

    gaussian = CondMultivariateGaussianLayer(
        scope=[Scope([0, 1], [3]), Scope([1, 2], [3])],
        cond_f=lambda data: {
            "mean": tl.zeros((2, 2)),
            "cov": tl.stack([tl.eye(2), tl.eye(2)]),
        },
    )
    data = tl.tensor([[float("nan"), float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(gaussian, data))

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor([1.0, 1.0])))

def test_support(do_for_all_backends):
    # TODO
    pass

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    cond_f = lambda data: {
        "mean": [[0.0, 0.0], [0.0, 0.0]],
        "cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
    }

    multivariate_gaussian = CondMultivariateGaussianLayer(Scope([0, 1], [2]), n_nodes=2, cond_f=cond_f)

    # create test inputs/outputs
    data = tl.stack([tl.zeros(2), tl.ones(2)], axis=0)
    log_probs = log_likelihood(multivariate_gaussian, data)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer_updated = updateBackend(multivariate_gaussian)
            log_probs_updated = log_likelihood(layer_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))

def test_change_dtype(do_for_all_backends):
    cond_f = lambda data: {
        "mean": [[0.0, 0.0], [0.0, 0.0]],
        "cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
    }

    layer = CondMultivariateGaussianLayer(Scope([0, 1], [2]), n_nodes=2, cond_f=cond_f)
    dummy_data = tl.tensor(tl.stack([tl.zeros(2), tl.ones(2)], axis=0), dtype=tl.float32)
    layer_ll = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    layer.to_dtype(tl.float64)
    dummy_data = tl.tensor(tl.stack([tl.zeros(2), tl.ones(2)], axis=0), dtype=tl.float64)
    layer_ll_up = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    cond_f = lambda data: {
        "mean": [[0.0, 0.0], [0.0, 0.0]],
        "cov": [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
    }

    layer = CondMultivariateGaussianLayer(Scope([0, 1], [2]), n_nodes=2, cond_f=cond_f)
    dummy_data = tl.tensor(tl.stack([tl.zeros(2), tl.ones(2)], axis=0))
    layer_ll = log_likelihood(layer, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    layer.to_device(cuda)
    dummy_data = tl.tensor(tl.stack([tl.zeros(2), tl.ones(2)], axis=0), device=cuda)
    layer_ll = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
