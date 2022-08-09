"""
Created on May 10, 2022

@authors: Philipp Deibert
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.meta.contexts.sampling_context import SamplingContext, init_default_sampling_context
from spflow.torch.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian

import torch
from typing import Optional


@dispatch
def sample(leaf: MultivariateGaussian, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None, sampling_ctx: Optional[SamplingContext]=None) -> torch.Tensor:
    """TODO"""
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    if any([i >= data.shape[0] for i in sampling_ctx.instance_ids]):
        raise ValueError("Some instance ids are out of bounds for data tensor.")

    nan_data = torch.isnan(data[:, leaf.scope.query])

    # group by scope rvs to sample
    for nan_mask in torch.unique(nan_data, dim=0):

        cond_mask = ~nan_mask
        cond_rvs = torch.where(cond_mask)[0]
        non_cond_rvs = torch.where(~cond_mask)[0]

        # no 'NaN' values (nothing to sample)
        if(torch.sum(nan_mask) == 0):
            continue
        # sample from full distribution
        elif(torch.sum(nan_mask) == len(leaf.scope.query)):
            sampling_ids = torch.tensor(sampling_ctx.instance_ids)[(nan_data == nan_mask).sum(dim=1) == nan_mask.shape[0]]

            data[torch.meshgrid(sampling_ids, non_cond_rvs, indexing='ij')] = leaf.dist.sample((sampling_ids.shape[0],)).squeeze(1)
        # sample from conditioned distribution
        else:
            # note: the conditional sampling implemented here is based on the algorithm described in Arnaud Doucet (2010): "A Note on Efficient Conditional Simulation of Gaussian Distributions" (https://www.stats.ox.ac.uk/~doucet/doucet_simulationconditionalgaussian.pdf)
            sampling_ids = torch.tensor(sampling_ctx.instance_ids)[(nan_data == nan_mask).sum(dim=1) == nan_mask.shape[0]]

            # sample from full distribution
            joint_samples = leaf.dist.sample((sampling_ids.shape[0],))

            # compute inverse of marginal covariance matrix of conditioning RVs
            marg_cov_inv = torch.linalg.inv(leaf.cov[torch.meshgrid(cond_rvs, cond_rvs, indexing='ij')])

            # get conditional covariance matrix
            cond_cov = leaf.cov[torch.meshgrid(cond_rvs, non_cond_rvs, indexing='ij')]

            data[torch.meshgrid(sampling_ids, non_cond_rvs, indexing='ij')] = joint_samples[:, ~cond_mask] + ((data[torch.meshgrid(sampling_ids, cond_rvs, indexing='ij')]-joint_samples[:, cond_mask])@(marg_cov_inv@cond_cov))

    return data