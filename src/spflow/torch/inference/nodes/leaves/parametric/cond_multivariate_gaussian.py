"""
Created on October 20, 2022

@authors: Philipp Deibert
"""
import torch
import torch.distributions as D
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.nodes.leaves.parametric.cond_multivariate_gaussian import CondMultivariateGaussian


@dispatch(memoize=True)
def log_likelihood(leaf: CondMultivariateGaussian, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    batch_size: int = data.shape[0]

    # retrieve values for 'mean','cov'
    mean, cov, cov_tril = leaf.retrieve_params(data, dispatch_ctx)

    if cov_tril is not None:
        cov = torch.matmul(cov_tril, cov_tril.T)

    # get information relevant for the scope
    scope_data = data[:, leaf.scope.query]

    # initialize empty tensor (number of output values matches batch_size)
    log_prob: torch.Tensor = torch.empty(batch_size, 1).to(mean.device)

    # create copy of the data where NaNs are replaced by zeros
    # TODO: alternative for initial validity checking without copying?
    _scope_data = scope_data.clone()
    _scope_data[_scope_data.isnan()] = 0.0

    # check support
    valid_ids = leaf.check_support(_scope_data).squeeze(1)

    del _scope_data  # free up memory

    # TODO: suppress checks
    if not all(valid_ids):
        raise ValueError(
            f"Encountered data instances that are not in the support of the TorchMultivariateGaussian distribution."
        )

    # ----- log probabilities -----

    marg = torch.isnan(scope_data)

    # group instances by marginalized random variables
    for marg_mask in marg.unique(dim=0):

        # get all instances with the same (marginalized) scope
        marg_ids = torch.where((marg == marg_mask).sum(dim=-1) == len(leaf.scope.query))[0]
        marg_data = scope_data[marg_ids]

        # all random variables are marginalized over
        if all(marg_mask):
            # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
            log_prob[marg_ids, 0] = 0.0
        # some random variables are marginalized over
        elif any(marg_mask):
            marg_data = marg_data[:, ~marg_mask]

            # marginalize distribution and compute (log) probabilities
            marg_mean = mean[~marg_mask]
            marg_cov = cov[~marg_mask][
                :, ~marg_mask
            ]  # TODO: better way?

            # create marginalized torch distribution
            marg_dist = D.MultivariateNormal(
                loc=marg_mean, covariance_matrix=marg_cov
            )

            # compute probabilities for values inside distribution support
            log_prob[marg_ids, 0] = marg_dist.log_prob(
                marg_data.type(torch.get_default_dtype())
            )
        # no random variables are marginalized over
        else:
            if cov_tril is not None:
                # compute probabilities for values inside distribution support
                log_prob[marg_ids, 0] = leaf.dist(mean=mean, cov_tril=cov_tril).log_prob(
                    marg_data.type(torch.get_default_dtype())
                )
            else:
                # compute probabilities for values inside distribution support
                log_prob[marg_ids, 0] = leaf.dist(mean=mean, cov=cov).log_prob(
                    marg_data.type(torch.get_default_dtype())
                )

    return log_prob