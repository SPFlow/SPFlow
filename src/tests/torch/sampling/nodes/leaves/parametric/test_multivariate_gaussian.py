from spflow.meta.scope.scope import Scope
from spflow.torch.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian
from spflow.torch.sampling.nodes.leaves.parametric.multivariate_gaussian import sample
from spflow.torch.sampling.module import sample

import torch
import unittest


class TestMultivariateGaussian(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_joint_sampling(self):

        # generate mean vector
        mean = torch.randn((1,5)) * 0.1
        # generate p.s.d covariance matrix
        cov = torch.randn((5,5)) * 0.1
        cov = cov@cov.T

        # create distribution
        mv = MultivariateGaussian(Scope([0,1,2,3,4]), mean=mean, cov=cov)

        # conditionally sample
        data = sample(mv, 100000)

        # estimate mean and covariance matrix for conditioned distribution from data
        mean_est = data.mean(dim=0)
        cov_est = torch.cov(data.T)

        self.assertTrue(torch.allclose(mean, mean_est, atol=0.01, rtol=0.1))
        self.assertTrue(torch.allclose(cov, cov_est, atol=0.01, rtol=0.1))
    
    def test_conditional_sampling(self):
        
        # generate mean vector
        mean = torch.randn((1,5)) * 0.1
        # generate p.s.d covariance matrix
        cov = torch.randn((5,5)) * 0.1
        cov = cov@cov.T

        # define which scope variabes are conditioned on
        cond_mask = torch.tensor([True, False, True, True, False])
        cond_rvs = torch.where(cond_mask)[0]
        non_cond_rvs = torch.where(~cond_mask)[0]

        d = torch.distributions.MultivariateNormal(loc=mean[:, cond_mask].squeeze(0), covariance_matrix=cov[cond_mask, :][:, cond_mask])

        # generate 10 different conditional values and repeat each 100 times
        cond_data = d.sample((10,))
        cond_data = cond_data.repeat_interleave(10000, dim=0)
    
        # generate patially filled data array
        data = torch.full((100000, 5), float("nan"))
        data[:, cond_mask] = cond_data

        # create distribution
        mv = MultivariateGaussian(Scope([0,1,2,3,4]), mean=mean, cov=cov)

        # conditionally sample
        data = sample(mv, data)

        # for each conditional value
        for i in range(10):

            # estimate mean and covariance matrix for conditioned distribution from data
            mean_est = data[i*10000:(i+1)*10000, ~cond_mask].mean(dim=0)
            cov_est = torch.cov(data[i*10000:(i+1)*10000, ~cond_mask].T)
        
            # compute analytical mean and covariance matrix for conditioned distribution
            marg_cov_inv = torch.linalg.inv(cov[torch.meshgrid(cond_rvs, cond_rvs, indexing='ij')])
            cond_cov = cov[torch.meshgrid(cond_rvs, non_cond_rvs, indexing='ij')]

            mean_exact = mean[0, ~cond_mask] + ((data[i*10000, cond_mask]-mean[:, cond_mask])@(marg_cov_inv@cond_cov))
            cov_exact = cov[torch.meshgrid(non_cond_rvs, non_cond_rvs, indexing='ij')] - (cond_cov.T@marg_cov_inv@cond_cov)

            self.assertTrue(torch.allclose(mean_exact, mean_est, atol=0.01, rtol=0.1))
            self.assertTrue(torch.allclose(cov_exact, cov_est, atol=0.01, rtol=0.1))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()