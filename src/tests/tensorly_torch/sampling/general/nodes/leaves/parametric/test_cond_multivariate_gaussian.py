import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.torch.sampling import sample
from spflow.tensorly.sampling import sample
from spflow.meta.dispatch import SamplingContext
#from spflow.torch.structure.spn import CondMultivariateGaussian
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_multivariate_gaussian import CondMultivariateGaussian
from spflow.torch.structure.general.nodes.leaves.parametric.cond_multivariate_gaussian import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy, tl_isnan


class TestMultivariateGaussian(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_joint_sampling(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # generate mean vector
        mean = torch.randn((1, 5)) * 0.1
        # generate p.s.d covariance matrix
        cov = torch.randn((5, 5)) * 0.1
        cov = cov @ cov.T

        # create distribution
        mv = CondMultivariateGaussian(
            Scope([0, 1, 2, 3, 4], [5]),
            cond_f=lambda data: {"mean": mean, "cov": cov},
        )

        # conditionally sample
        data = sample(mv, 100000)

        # estimate mean and covariance matrix for conditioned distribution from data
        mean_est = data.mean(dim=0)
        cov_est = torch.cov(data.T)

        self.assertTrue(torch.allclose(mean, mean_est, atol=0.01, rtol=0.1))
        self.assertTrue(torch.allclose(cov, cov_est, atol=0.01, rtol=0.1))

    def test_conditional_sampling(self):

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # generate mean vector
        mean = torch.randn((1, 5)) * 0.1
        # generate p.s.d covariance matrix
        cov = torch.randn((5, 5)) * 0.1
        cov = cov @ cov.T

        # define which scope variables are conditioned on
        cond_mask = torch.tensor([True, False, True, True, False])
        cond_rvs = torch.where(cond_mask)[0]
        non_cond_rvs = torch.where(~cond_mask)[0]

        d = torch.distributions.MultivariateNormal(
            loc=mean[:, cond_mask].squeeze(0),
            covariance_matrix=cov[cond_mask, :][:, cond_mask],
        )

        # generate 10 different conditional values and repeat each 100 times
        cond_data = d.sample((10,))
        cond_data = cond_data.repeat_interleave(10000, dim=0)

        # generate patially filled data array
        data = torch.full((100000, 5), float("nan"))
        data[:, cond_mask] = cond_data

        # create distribution
        mv = CondMultivariateGaussian(
            Scope([0, 1, 2, 3, 4], [5]),
            cond_f=lambda data: {"mean": mean, "cov": cov},
        )

        # conditionally sample
        data = sample(mv, data)

        # for each conditional value
        for i in range(10):

            # estimate mean and covariance matrix for conditioned distribution from data
            mean_est = data[i * 10000 : (i + 1) * 10000, ~cond_mask].mean(dim=0)
            cov_est = torch.cov(data[i * 10000 : (i + 1) * 10000, ~cond_mask].T)

            # compute analytical mean and covariance matrix for conditioned distribution
            marg_cov_inv = torch.linalg.inv(cov[torch.meshgrid(cond_rvs, cond_rvs, indexing="ij")])
            cond_cov = cov[torch.meshgrid(cond_rvs, non_cond_rvs, indexing="ij")]

            mean_exact = mean[0, ~cond_mask] + (
                (data[i * 10000, cond_mask] - mean[:, cond_mask]) @ (marg_cov_inv @ cond_cov)
            )
            cov_exact = cov[torch.meshgrid(non_cond_rvs, non_cond_rvs, indexing="ij")] - (
                cond_cov.T @ marg_cov_inv @ cond_cov
            )

            self.assertTrue(torch.allclose(mean_exact, mean_est, atol=0.01, rtol=0.1))
            self.assertTrue(torch.allclose(cov_exact, cov_est, atol=0.01, rtol=0.1))

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # generate mean vector
        mean = torch.randn((1, 5)) * 0.1
        # generate p.s.d covariance matrix
        cov = torch.randn((5, 5)) * 0.1
        cov = cov @ cov.T

        # create distribution
        mv = CondMultivariateGaussian(
            Scope([0, 1, 2, 3, 4], [5]),
            cond_f=lambda data: {"mean": mean, "cov": cov},
        )

        # conditionally sample
        data = sample(mv, 100000)
        #notNans = samples[~tl_isnan(samples)]
        mean_est = data.mean(dim=0)
        cov_est = torch.cov(data.T)

        # make sure that probabilities match python backend probabilities
        for backend in backends:
            tl.set_backend(backend)
            bernoulli_updated = updateBackend(mv)
            samples_updated = sample(bernoulli_updated, 100000)
            # check conversion from torch to python
            mean_est_updated = samples_updated.mean(dim=0)
            cov_est_updated = torch.cov(samples_updated.T)
            self.assertTrue(mean_est == mean_est_updated)
            self.assertTrue(cov_est == cov_est_updated)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
