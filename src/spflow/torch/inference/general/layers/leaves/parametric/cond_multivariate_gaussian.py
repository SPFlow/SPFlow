# -*- coding: utf-8 -*-
"""Contains inference methods for ``CondMultivariateGaussianLayer`` leaves for SPFlow in the ``torch`` backend.
"""
import torch
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.layers.leaves.parametric.cond_multivariate_gaussian import (
    CondMultivariateGaussianLayer,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    layer: CondMultivariateGaussianLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> torch.Tensor:
    r"""Computes log-likelihoods for ``CondMultivariateGaussianLayer`` leaves in the ``torch`` backend given input data.

    Log-likelihood for ``CondMultivariateGaussianLayer`` is given by the logarithm of its individual probability distribution functions (PDFs):

    .. math::

        \log(\text{PDF}(x)) = \log(\frac{1}{\sqrt{(2\pi)^d\det\Sigma}}\exp\left(-\frac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)\right))

    where
        - :math:`d` is the dimension of the distribution
        - :math:`x` is the :math:`d`-dim. vector of observations
        - :math:`\mu` is the :math:`d`-dim. mean vector
        - :math:`\Sigma` is the :math:`d\times d` covariance matrix

    Missing values (i.e., NaN) are marginalized over.

    Args:
        node:
            Leaf node to perform inference for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.

    Raises:
        ValueError: Data outside of support.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # retrieve values for 'mean','cov'
    mean_values, cov_values = layer.retrieve_params(data, dispatch_ctx)

    for node, mean, cov in zip(layer.nodes, mean_values, cov_values):
        dispatch_ctx.update_args(node, {"mean": mean, "cov": cov})

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return torch.concat(
        [
            log_likelihood(
                node,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for node in layer.nodes
        ],
        dim=1,
    )