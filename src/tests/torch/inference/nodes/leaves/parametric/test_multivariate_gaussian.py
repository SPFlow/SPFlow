from spflow.meta.data.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian as BaseMultivariateGaussian,
)
from spflow.base.inference.nodes.leaves.parametric.multivariate_gaussian import (
    log_likelihood,
)
from spflow.torch.structure.nodes.node import SPNProductNode
from spflow.torch.inference.nodes.node import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian,
    toBase,
    toTorch,
    marginalize,
)
from spflow.torch.inference.nodes.leaves.parametric.multivariate_gaussian import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.torch.inference.nodes.leaves.parametric.gaussian import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import math

import unittest


class TestMultivariateGaussian(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_inference(self):

        mean = np.arange(3)
        cov = np.array([[2, 2, 1], [2, 3, 2], [1, 2, 3]])

        torch_multivariate_gaussian = MultivariateGaussian(
            Scope([0, 1, 2]), mean, cov
        )
        node_multivariate_gaussian = BaseMultivariateGaussian(
            Scope([0, 1, 2]), mean.tolist(), cov.tolist()
        )

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 3)

        log_probs = log_likelihood(node_multivariate_gaussian, data)
        log_probs_torch = log_likelihood(
            torch_multivariate_gaussian, torch.tensor(data)
        )

        # make sure that probabilities match python backend probabilities
        self.assertTrue(
            np.allclose(log_probs, log_probs_torch.detach().cpu().numpy())
        )

    def test_gradient_computation(self):

        mean = np.arange(3)
        cov = np.array([[2, 2, 1], [2, 3, 2], [1, 2, 3]])

        torch_multivariate_gaussian = MultivariateGaussian(
            Scope([0, 1, 2]), mean, cov
        )

        # create dummy input data (batch size x random variables)
        data = np.random.rand(3, 3)

        log_probs_torch = log_likelihood(
            torch_multivariate_gaussian, torch.tensor(data)
        )

        # create dummy targets
        targets_torch = torch.ones(3, 1)

        loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
        loss.backward()

        self.assertTrue(torch_multivariate_gaussian.mean.grad is not None)
        self.assertTrue(
            torch_multivariate_gaussian.tril_diag_aux.grad is not None
        )
        self.assertTrue(
            torch_multivariate_gaussian.tril_nondiag.grad is not None
        )

        mean_orig = torch_multivariate_gaussian.mean.detach().clone()
        tril_diag_aux_orig = (
            torch_multivariate_gaussian.tril_diag_aux.detach().clone()
        )
        tril_nondiag_orig = (
            torch_multivariate_gaussian.tril_nondiag.detach().clone()
        )

        optimizer = torch.optim.SGD(
            torch_multivariate_gaussian.parameters(), lr=1
        )
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(
            torch.allclose(
                mean_orig - torch_multivariate_gaussian.mean.grad,
                torch_multivariate_gaussian.mean,
            )
        )
        self.assertTrue(
            torch.allclose(
                tril_diag_aux_orig
                - torch_multivariate_gaussian.tril_diag_aux.grad,
                torch_multivariate_gaussian.tril_diag_aux,
            )
        )
        self.assertTrue(
            torch.allclose(
                tril_nondiag_orig
                - torch_multivariate_gaussian.tril_nondiag.grad,
                torch_multivariate_gaussian.tril_nondiag,
            )
        )

        # verify that distribution parameters match parameters
        self.assertTrue(
            torch.allclose(
                torch_multivariate_gaussian.mean,
                torch_multivariate_gaussian.dist.loc,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch_multivariate_gaussian.cov,
                torch_multivariate_gaussian.dist.covariance_matrix,
            )
        )

    def test_gradient_optimization(self):

        # initialize distribution
        torch_multivariate_gaussian = MultivariateGaussian(
            Scope([0, 1]),
            mean=torch.tensor([1.0, -1.0]),
            cov=torch.tensor(
                [
                    [2.0, 0.5],
                    [0.5, 1.5],
                ]
            ),
        )

        torch.manual_seed(0)

        # create dummy data (unit variance Gaussian)
        data = torch.distributions.MultivariateNormal(
            loc=torch.zeros(2), covariance_matrix=torch.eye(2)
        ).sample((100000,))

        # initialize gradient optimizer
        optimizer = torch.optim.SGD(
            torch_multivariate_gaussian.parameters(), lr=0.5
        )

        # perform optimization (possibly overfitting)
        for i in range(20):

            # clear gradients
            optimizer.zero_grad()

            # compute negative log-likelihood
            nll = -log_likelihood(torch_multivariate_gaussian, data).mean()
            nll.backward()

            # update parameters
            optimizer.step()

        self.assertTrue(
            torch.allclose(
                torch_multivariate_gaussian.mean,
                torch.zeros(2),
                atol=1e-2,
                rtol=0.3,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch_multivariate_gaussian.cov,
                torch.eye(2),
                atol=1e-2,
                rtol=0.3,
            )
        )

    def test_likelihood_marginalization(self):

        # ----- full marginalization -----

        multivariate_gaussian = MultivariateGaussian(
            Scope([0, 1]),
            torch.zeros(2),
            torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
        )
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise an error and should return 1
        probs = likelihood(multivariate_gaussian, data)

        self.assertTrue(torch.allclose(probs, torch.tensor(1.0)))

        # ----- partial marginalization -----

        data = torch.tensor([[0.0, float("nan")], [float("nan"), 0.0]])
        targets = torch.tensor([[0.282095], [0.398942]])

        # inference using multivariate gaussian and partial marginalization
        mv_probs = likelihood(multivariate_gaussian, data)

        self.assertTrue(torch.allclose(mv_probs, targets))

        # inference using univariate gaussians for each random variable (combined via product node for convenience)
        univariate_gaussians = SPNProductNode(
            children=[
                Gaussian(
                    Scope([0]), 0.0, math.sqrt(2.0)
                ),  # requires standard deviation instead of variance
                Gaussian(Scope([1]), 0.0, 1.0),
            ],
        )

        uv_probs = likelihood(univariate_gaussians, data)

        # compare
        self.assertTrue(torch.allclose(mv_probs, uv_probs))

        # inference using "structurally" marginalized multivariate gaussians for each random variable (combined via product node for convenience)
        marginalized_mv_gaussians = SPNProductNode(
            children=[
                marginalize(multivariate_gaussian, [1]),
                marginalize(multivariate_gaussian, [0]),
            ],
        )

        marg_mv_probs = likelihood(marginalized_mv_gaussians, data)

        # compare
        self.assertTrue(torch.allclose(marg_mv_probs, uv_probs))

        # higher-dimensional example
        multivariate_gaussian = MultivariateGaussian(
            Scope([0, 1, 2, 3]),
            torch.zeros(4),
            torch.tensor(
                [
                    [2.0, 0.5, 0.5, 0.25],
                    [0.5, 1.0, 0.75, 0.5],
                    [0.5, 0.75, 1.5, 0.5],
                    [0.25, 0.5, 0.5, 1.25],
                ]
            ),
        )

        data = torch.tensor(
            [
                [0.0] * 4,
                [0.0, float("nan"), float("nan"), 0.0],
                [float("nan"), 0.0, 0.0, 0.0],
                [float("nan")] * 4,
            ]
        )
        targets = torch.tensor(
            [[0.02004004], [0.10194075], [0.06612934], [1.0]]
        )

        # inference using multivariate gaussian and partial marginalization
        mv_probs = likelihood(multivariate_gaussian, data)

        self.assertTrue(torch.allclose(mv_probs, targets, atol=1e-6))

    def test_support(self):

        # Support for Multivariate Gaussian distribution: floats (inf,+inf)^k

        multivariate_gaussian = MultivariateGaussian(
            Scope([0, 1]), np.zeros(2), np.eye(2)
        )

        # check infinite values
        self.assertRaises(
            ValueError,
            log_likelihood,
            multivariate_gaussian,
            torch.tensor([[-float("inf"), 0.0]]),
        )
        self.assertRaises(
            ValueError,
            log_likelihood,
            multivariate_gaussian,
            torch.tensor([[0.0, float("inf")]]),
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
