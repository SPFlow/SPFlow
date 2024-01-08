import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.modules.module import sample
from spflow.modules.node import CondMultivariateGaussian
from spflow.torch.structure.general.node.leaf.cond_multivariate_gaussian import updateBackend
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_joint_sampling(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # generate mean vector
    mean = tl.randn((1, 5)) * 0.1
    # generate p.s.d covariance matrix
    cov = tl.randn((5, 5)) * 0.1
    cov = cov @ cov.T

    # create distribution
    mv = CondMultivariateGaussian(
        Scope([0, 1, 2, 3, 4], [5]),
        cond_f=lambda data: {"mean": mean, "cov": cov},
    )

    # conditionally sample
    data = sample(mv, 100000)

    # estimate mean and covariance matrix for conditioned distribution from data
    mean_est = tl.mean(data, axis=0)
    cov_est = np.cov(data.T)

    tc.assertTrue(np.allclose(mean, mean_est, atol=0.01, rtol=0.1))
    tc.assertTrue(np.allclose(cov, cov_est, atol=0.01, rtol=0.1))


def test_conditional_sampling(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # generate mean vector
    mean = tl.randn((1, 5), dtype=tl.float32) * 0.1
    # generate p.s.d covariance matrix
    cov = tl.randn((5, 5), dtype=tl.float32) * 0.1
    cov = cov @ cov.T

    # define which scope variables are conditioned on
    cond_mask = tl.tensor([True, False, True, True, False], dtype=bool)
    cond_rvs = tl.where(cond_mask)[0]
    non_cond_rvs = tl.where(~cond_mask)[0]

    # generate 10 different conditional values and repeat each 100 times
    cond_data = tle.multivariate_normal(
        loc=mean[:, cond_mask].squeeze(0), cov_matrix=cov[cond_mask, :][:, cond_mask], size=(10,)
    )

    cond_data = tle.repeat(cond_data, 10000, axis=0)

    # generate patially filled data array
    data = tle.full((100000, 5), float("nan"))
    data[:, cond_mask] = tl.tensor(cond_data, dtype=tl.float32)

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
        mean_est = tl.mean(data[i * 10000 : (i + 1) * 10000, ~cond_mask], axis=0)
        cov_est = np.cov(data[i * 10000 : (i + 1) * 10000, ~cond_mask].T)

        # compute analytical mean and covariance matrix for conditioned distribution
        marg_cov_inv = tle.inv(
            cov[
                tuple(
                    tl.tensor(tensor, dtype=tl.int64) for tensor in tle.ix_(cond_rvs, cond_rvs, indexing="ij")
                )
            ]
        )
        cond_cov = cov[
            tuple(
                tl.tensor(tensor, dtype=tl.int64) for tensor in tle.ix_(cond_rvs, non_cond_rvs, indexing="ij")
            )
        ]

        mean_exact = mean[0, ~cond_mask] + (
            (data[i * 10000, cond_mask] - mean[:, cond_mask])
            @ (tl.tensor(marg_cov_inv, dtype=tl.float32) @ cond_cov)
        )
        cov_exact = cov[
            tuple(
                tl.tensor(tensor, dtype=tl.int64)
                for tensor in tle.ix_(non_cond_rvs, non_cond_rvs, indexing="ij")
            )
        ] - (cond_cov.T @ tl.tensor(marg_cov_inv, dtype=tl.float32) @ cond_cov)

        tc.assertTrue(np.allclose(mean_exact, mean_est, atol=0.01, rtol=0.1))
        tc.assertTrue(np.allclose(cov_exact, cov_est, atol=0.01, rtol=0.1))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # generate mean vector
    mean = tl.randn((1, 5), dtype=tl.float64) * 0.1
    # generate p.s.d covariance matrix
    cov = tl.randn((5, 5), dtype=tl.float64) * 0.1
    cov = cov @ cov.T

    # create distribution
    mv = CondMultivariateGaussian(
        Scope([0, 1, 2, 3, 4], [5]),
        cond_f=lambda data: {"mean": mean, "cov": cov},
    )

    # conditionally sample
    data = sample(mv, 100000)
    # notNans = samples[~tle.isnan(samples)]
    mean_est = tl.mean(data, axis=0)
    cov_est = np.cov(data.T)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            mv_updated = updateBackend(mv)
            samples_updated = sample(mv_updated, 100000)
            # check conversion from torch to python
            mean_est_updated = tl.mean(samples_updated, axis=0)
            cov_est_updated = np.cov(samples_updated.T)
            tc.assertTrue(np.allclose(mean_est, mean_est_updated, atol=0.01, rtol=0.1))
            tc.assertTrue(np.allclose(cov_est, cov_est_updated, atol=0.01, rtol=0.1))


def test_change_dtype(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_dtype(torch.float32)

    # generate mean vector
    mean = tl.randn((1, 5), dtype=tl.float32) * 0.1
    # generate p.s.d covariance matrix
    cov = tl.randn((5, 5), dtype=tl.float32) * 0.1
    cov = cov @ cov.T

    # create distribution
    layer = CondMultivariateGaussian(
        Scope([0, 1, 2, 3, 4], [5]),
        cond_f=lambda data: {"mean": mean, "cov": cov},
    )

    samples = sample(layer, 100000)

    tc.assertTrue(samples.dtype == tl.float32)
    layer.to_dtype(tl.float64)

    samples = sample(layer, 100000)
    tc.assertTrue(samples.dtype == tl.float64)


def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # generate mean vector
    mean = tl.randn((1, 5), dtype=tl.float32) * 0.1
    # generate p.s.d covariance matrix
    cov = tl.randn((5, 5), dtype=tl.float32) * 0.1
    cov = cov @ cov.T

    # create distribution
    layer = CondMultivariateGaussian(
        Scope([0, 1, 2, 3, 4], [5]),
        cond_f=lambda data: {"mean": mean, "cov": cov},
    )

    samples = sample(layer, 100000)

    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer.to_device, cuda)
        return

    tc.assertTrue(samples.device.type == "cpu")
    layer.to_device(cuda)
    samples = sample(layer, 100000)
    tc.assertTrue(samples.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()