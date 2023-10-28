import math
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.tensorly.structure.spn import MultivariateGaussian
from spflow.meta.data import Scope
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure import marginalize
from spflow.tensorly.structure.spn import ProductNode
from spflow.tensorly.structure.spn import Gaussian
from spflow.base.structure.general.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian as BaseMultivariateGaussian
from spflow.torch.structure.general.nodes.leaves.parametric.multivariate_gaussian import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_inference(do_for_all_backends):
    torch.set_default_dtype(torch.float64)

    mean = np.arange(3)
    cov = np.array([[2, 2, 1], [2, 3, 2], [1, 2, 3]])

    torch_multivariate_gaussian = MultivariateGaussian(Scope([0, 1, 2]), mean, cov)
    node_multivariate_gaussian = BaseMultivariateGaussian(Scope([0, 1, 2]), mean.tolist(), cov.tolist())

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 3)

    log_probs = log_likelihood(node_multivariate_gaussian, data)
    log_probs_torch = log_likelihood(torch_multivariate_gaussian, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tl_toNumpy(log_probs_torch)))

def test_gradient_computation(do_for_all_backends):
    torch.set_default_dtype(torch.float64)

    if do_for_all_backends == "numpy":
        return

    mean = np.arange(3)
    cov = np.array([[2, 2, 1], [2, 3, 2], [1, 2, 3]])

    torch_multivariate_gaussian = MultivariateGaussian(Scope([0, 1, 2]), mean, cov)

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 3)

    log_probs_torch = log_likelihood(torch_multivariate_gaussian, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_multivariate_gaussian.mean.grad is not None)
    tc.assertTrue(torch_multivariate_gaussian.tril_diag_aux.grad is not None)
    tc.assertTrue(torch_multivariate_gaussian.tril_nondiag.grad is not None)

    mean_orig = torch_multivariate_gaussian.mean.detach().clone()
    tril_diag_aux_orig = torch_multivariate_gaussian.tril_diag_aux.detach().clone()
    tril_nondiag_orig = torch_multivariate_gaussian.tril_nondiag.detach().clone()

    optimizer = torch.optim.SGD(torch_multivariate_gaussian.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(
        torch.allclose(
            mean_orig - torch_multivariate_gaussian.mean.grad,
            torch_multivariate_gaussian.mean,
        )
    )
    tc.assertTrue(
        torch.allclose(
            tril_diag_aux_orig - torch_multivariate_gaussian.tril_diag_aux.grad,
            torch_multivariate_gaussian.tril_diag_aux,
        )
    )
    tc.assertTrue(
        torch.allclose(
             tril_nondiag_orig - torch_multivariate_gaussian.tril_nondiag.grad,
            torch_multivariate_gaussian.tril_nondiag,
        )
    )

    # verify that distribution parameters match parameters
    tc.assertTrue(
        torch.allclose(
            torch_multivariate_gaussian.mean,
            torch_multivariate_gaussian.dist.loc,
        )
    )
    tc.assertTrue(
        torch.allclose(
            torch_multivariate_gaussian.cov,
            torch_multivariate_gaussian.dist.covariance_matrix,
        )
    )

def test_gradient_optimization(do_for_all_backends):
    torch.set_default_dtype(torch.float64)

    if do_for_all_backends == "numpy":
        return

    # initialize distribution
    torch_multivariate_gaussian = MultivariateGaussian(
        Scope([0, 1]),
        mean=tl.tensor([1.0, -1.0], dtype=tl.float64),
        cov=tl.tensor(
            [
                [2.0, 0.5],
                [0.5, 1.5],
            ]
        , dtype=tl.float64),
    )

    torch.manual_seed(0)

    # create dummy data (unit variance Gaussian)
    data = torch.distributions.MultivariateNormal(loc=tl.zeros(2), covariance_matrix=tl.eye(2)).sample(
        (100000,)
    )

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_multivariate_gaussian.parameters(), lr=0.5)

    # perform optimization (possibly overfitting)
    for i in range(20):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_multivariate_gaussian, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(
        torch.allclose(
            torch_multivariate_gaussian.mean,
            tl.zeros(2),
            atol=1e-2,
            rtol=0.3,
        )
    )
    tc.assertTrue(
        torch.allclose(
            torch_multivariate_gaussian.cov,
            tl.eye(2),
            atol=1e-2,
            rtol=0.3,
        )
    )

def test_likelihood_marginalization(do_for_all_backends):
    torch.set_default_dtype(torch.float64)

    # ----- full marginalization -----

    multivariate_gaussian = MultivariateGaussian(
        Scope([0, 1]),
        tl.zeros(2),
        tl.tensor([[2.0, 0.0], [0.0, 1.0]]),
    )
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise an error and should return 1
    probs = likelihood(multivariate_gaussian, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0)))

    # ----- partial marginalization -----

    data = tl.tensor([[0.0, float("nan")], [float("nan"), 0.0]])
    targets = tl.tensor([[0.282095], [0.398942]])

    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, likelihood, multivariate_gaussian, data)
    else:

        # inference using multivariate gaussian and partial marginalization
        mv_probs = likelihood(multivariate_gaussian, data)

        tc.assertTrue(np.allclose(tl_toNumpy(mv_probs), targets))

        # inference using univariate gaussians for each random variable (combined via product node for convenience)
        univariate_gaussians = ProductNode(
            children=[
                Gaussian(Scope([0]), 0.0, math.sqrt(2.0)),  # requires standard deviation instead of variance
                Gaussian(Scope([1]), 0.0, 1.0),
            ],
        )

        uv_probs = likelihood(univariate_gaussians, data)

        # compare
        tc.assertTrue(np.allclose(tl_toNumpy(mv_probs), tl_toNumpy(uv_probs)))

        # inference using "structurally" marginalized multivariate gaussians for each random variable (combined via product node for convenience)
        marginalized_mv_gaussians = ProductNode(
            children=[
                marginalize(multivariate_gaussian, [1]),
                marginalize(multivariate_gaussian, [0]),
            ],
        )

        marg_mv_probs = likelihood(marginalized_mv_gaussians, data)

        # compare
        tc.assertTrue(np.allclose(tl_toNumpy(marg_mv_probs), tl_toNumpy(uv_probs)))

        # higher-dimensional example
        multivariate_gaussian = MultivariateGaussian(
            Scope([0, 1, 2, 3]),
            tl.zeros(4),
            tl.tensor(
                [
                    [2.0, 0.5, 0.5, 0.25],
                    [0.5, 1.0, 0.75, 0.5],
                    [0.5, 0.75, 1.5, 0.5],
                    [0.25, 0.5, 0.5, 1.25],
                ]
            , dtype=tl.float64),
        )

        data = tl.tensor(
            [
                [0.0] * 4,
                [0.0, float("nan"), float("nan"), 0.0],
                [float("nan"), 0.0, 0.0, 0.0],
                [float("nan")] * 4,
            ]
        ,tl.float64)
        targets = tl.tensor([[0.02004004], [0.10194075], [0.06612934], [1.0]], dtype=tl.float64)

        # inference using multivariate gaussian and partial marginalization
        mv_probs = likelihood(multivariate_gaussian, data)

        tc.assertTrue(np.allclose(tl_toNumpy(mv_probs), targets, atol=1e-6))

def test_support(do_for_all_backends):

    # Support for Multivariate Gaussian distribution: floats (inf,+inf)^k

    multivariate_gaussian = MultivariateGaussian(Scope([0, 1]), np.zeros(2), np.eye(2))

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

    multivariate_gaussian = MultivariateGaussian(Scope([0, 1, 2]), mean, cov)

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


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
