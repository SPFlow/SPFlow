import math
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.tensorly.structure.spn import (
    CondMultivariateGaussian
)
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.torch.structure.general.node.leaf.cond_multivariate_gaussian import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy
from spflow.tensorly.structure.spn import ProductNode
from spflow.tensorly.structure.general.node.leaf.general_cond_gaussian import CondGaussian
from spflow.base.structure.general.node.leaf.cond_multivariate_gaussian import CondMultivariateGaussian as BaseCondMultivariateGaussian

tc = unittest.TestCase()

def test_likelihood_module_cond_f(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    cond_f = lambda data: {"mean": tl.zeros(2), "cov": tl.eye(2)}

    multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [2]), cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor(np.stack([np.zeros(2), np.ones(2)], axis=0))
    targets = tl.tensor([[0.1591549], [0.0585498]])

    probs = likelihood(multivariate_gaussian, data)
    log_probs = log_likelihood(multivariate_gaussian, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_p(do_for_all_backends):

    multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [2]))

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[multivariate_gaussian] = {
        "mean": tl.zeros(2),
        "cov": tl.eye(2),
    }

    # create test inputs/outputs
    data = tl.tensor(np.stack([np.zeros(2), np.ones(2)], axis=0))
    targets = tl.tensor([[0.1591549], [0.0585498]])

    probs = likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_cond_f(do_for_all_backends):

    multivariate_gaussian = CondMultivariateGaussian(Scope([0, 1], [2]))

    cond_f = lambda data: {"mean": tl.zeros(2), "cov": tl.eye(2)}

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[multivariate_gaussian] = {"cond_f": cond_f}

    # create test inputs/outputs
    data = tl.tensor(np.stack([np.zeros(2), np.ones(2)], axis=0))
    targets = tl.tensor([[0.1591549], [0.0585498]])

    probs = likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(multivariate_gaussian, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_inference(do_for_all_backends):

    mean = np.arange(3)
    cov = np.array([[2, 2, 1], [2, 3, 2], [1, 2, 3]])

    torch_multivariate_gaussian = CondMultivariateGaussian(
        Scope([0, 1, 2], [3]),
        cond_f=lambda data: {"mean": mean, "cov": cov},
    )
    node_multivariate_gaussian = BaseCondMultivariateGaussian(
        Scope([0, 1, 2], [3]),
        cond_f=lambda data: {"mean": mean, "cov": cov},
    )

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 3)

    log_probs = log_likelihood(node_multivariate_gaussian, data)
    log_probs_torch = log_likelihood(torch_multivariate_gaussian, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tl_toNumpy(log_probs_torch)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    mean = tl.tensor(np.arange(3), dtype=torch.get_default_dtype(), requires_grad=True)
    cov = tl.tensor(
        [[2.0, 2.0, 1.0], [2.0, 3.0, 2.0], [1.0, 2.0, 3.0]],
        requires_grad=True,
    )

    torch_multivariate_gaussian = CondMultivariateGaussian(
        Scope([0, 1, 2], [3]),
        cond_f=lambda data: {"mean": mean, "cov": cov},
    )

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 3)

    log_probs_torch = log_likelihood(torch_multivariate_gaussian, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(mean.grad is not None)
    tc.assertTrue(cov.grad is not None)

def test_likelihood_marginalization(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    # ----- full marginalization -----

    multivariate_gaussian = CondMultivariateGaussian(
        Scope([0, 1], [2]),
        cond_f=lambda data: {
            "mean": tl.zeros(2, dtype=tl.float64),
            "cov": tl.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=tl.float64),
        },
    )
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise an error and should return 1
    probs = likelihood(multivariate_gaussian, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0, dtype=tl.float64)))

    # ----- partial marginalization -----

    data = tl.tensor([[0.0, float("nan")], [float("nan"), 0.0]], dtype=tl.float64)
    targets = tl.tensor([[0.282095], [0.398942]], dtype=tl.float64)

    # inference using multivariate gaussian and partial marginalization
    # partial marginalization is not implemented for base backend
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, likelihood, multivariate_gaussian, data)
    else:
        mv_probs = likelihood(multivariate_gaussian, data)
        tc.assertTrue(np.allclose(tl_toNumpy(mv_probs), targets))

        # inference using univariate gaussians for each random variable (combined via product node for convenience)
        univariate_gaussians = ProductNode(
            children=[
                CondGaussian(
                    Scope([0], [2]),
                    cond_f=lambda data: {"mean": 0.0, "std": math.sqrt(2.0)},
                ),  # requires standard deviation instead of variance
                CondGaussian(
                    Scope([1], [2]),
                    cond_f=lambda data: {"mean": 0.0, "std": 1.0},
                ),
            ],
        )

        uv_probs = likelihood(univariate_gaussians, data)

        # compare
        tc.assertTrue(np.allclose(tl_toNumpy(mv_probs), tl_toNumpy(uv_probs)))

        # higher-dimensional example
        multivariate_gaussian = CondMultivariateGaussian(
            Scope([0, 1, 2, 3], [4]),
            cond_f=lambda data: {
                "mean": tl.zeros(4, dtype=tl.float64),
                "cov": tl.tensor(
                    [
                        [2.0, 0.5, 0.5, 0.25],
                        [0.5, 1.0, 0.75, 0.5],
                        [0.5, 0.75, 1.5, 0.5],
                        [0.25, 0.5, 0.5, 1.25],
                    ]
                , dtype=tl.float64),
            },
        )

        data = tl.tensor(
            [
                [0.0] * 4,
                [0.0, float("nan"), float("nan"), 0.0],
                [float("nan"), 0.0, 0.0, 0.0],
                [float("nan")] * 4,
            ]
        , dtype=tl.float64)
        targets = tl.tensor([[0.02004004], [0.10194075], [0.06612934], [1.0]], dtype=tl.float64)

        # inference using multivariate gaussian and partial marginalization
        mv_probs = likelihood(multivariate_gaussian, data)

        tc.assertTrue(np.allclose(tl_toNumpy(mv_probs), targets, atol=1e-6))

def test_support(do_for_all_backends):

    # Support for Multivariate Gaussian distribution: floats (inf,+inf)^k

    multivariate_gaussian = CondMultivariateGaussian(
        Scope([0, 1], [2]),
        cond_f=lambda data: {"mean": np.zeros(2), "cov": np.eye(2)},
    )

    # check infinite values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        multivariate_gaussian,
        tl.tensor([[-float("inf"), 0.0]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        multivariate_gaussian,
        tl.tensor([[0.0, float("inf")]]),
    )

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    mean = np.arange(3)
    cov = np.array([[2, 2, 1], [2, 3, 2], [1, 2, 3]])

    multivariate_gaussian = CondMultivariateGaussian(
        Scope([0, 1, 2], [3]),
        cond_f=lambda data: {"mean": mean, "cov": cov},
    )

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 3)

    log_probs = log_likelihood(multivariate_gaussian, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            multivariate_gaussian_updated = updateBackend(multivariate_gaussian)
            log_probs_updated = log_likelihood(multivariate_gaussian_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))

def test_change_dtype(do_for_all_backends):
    mean = np.arange(3)
    cov = np.array([[2, 2, 1], [2, 3, 2], [1, 2, 3]])

    node = CondMultivariateGaussian(
        Scope([0, 1, 2], [3]),
        cond_f=lambda data: {"mean": mean, "cov": cov},
    )
    dummy_data = tl.tensor(np.random.rand(3, 3), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    node.to_dtype(tl.float64)
    dummy_data = tl.tensor(np.random.rand(3, 3), dtype=tl.float64)
    layer_ll_up = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    mean = np.arange(3)
    cov = np.array([[2, 2, 1], [2, 3, 2], [1, 2, 3]])

    node = CondMultivariateGaussian(
        Scope([0, 1, 2], [3]),
        cond_f=lambda data: {"mean": mean, "cov": cov},
    )
    dummy_data = tl.tensor(np.random.rand(3, 3), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, node.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    node.to_device(cuda)
    dummy_data = tl.tensor(np.random.rand(3, 3), device=cuda)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
