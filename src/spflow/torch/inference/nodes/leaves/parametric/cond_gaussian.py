# -*- coding: utf-8 -*-
"""Contains inference methods for ``CondGaussian`` nodes for SPFlow in the ``torch`` backend.
"""
import torch
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.torch.structure.nodes.leaves.parametric.cond_gaussian import CondGaussian


@dispatch(memoize=True)  # type: ignore
def log_likelihood(leaf: CondGaussian, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    r"""Computes log-likelihoods for ``CondGaussian`` node given input data in the ``torch`` backend.

    Log-likelihood for ``CondGaussian`` is given by the logarithm of its probability distribution function (PDF):

    .. math::

        \log(\text{PDF}(x)) = \log(\frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2}))

    where
        - :math:`x` the observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Missing values (i.e., NaN) are marginalized over.

    Args:
        node:
            Leaf node to perform inference for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    batch_size: int = data.shape[0]

    # retrieve values for 'mean','std'
    mean, std = leaf.retrieve_params(data, dispatch_ctx)

    # get information relevant for the scope
    scope_data = data[:, leaf.scope.query]

    # initialize empty tensor (number of output values matches batch_size)
    log_prob: torch.Tensor = torch.empty(batch_size, 1).to(mean.device)

    # ----- marginalization -----

    marg_ids = torch.isnan(scope_data).sum(dim=1) == len(leaf.scope.query)

    # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
    log_prob[marg_ids] = 0.0

    # ----- log probabilities -----

    # create mask based on distribution's support
    valid_ids = leaf.check_support(scope_data[~marg_ids]).squeeze(1)

    # TODO: suppress checks
    if not all(valid_ids):
        raise ValueError(
            f"Encountered data instances that are not in the support of the CondGaussian distribution."
        )

    # compute probabilities for values inside distribution support
    log_prob[~marg_ids] = leaf.dist(mean=mean, std=std).log_prob(
        scope_data[~marg_ids].type(torch.get_default_dtype())
    )

    return log_prob