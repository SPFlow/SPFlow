"""Contains inference methods for ``MultivariateGaussian`` nodes for SPFlow in the ``torch`` backend.
"""
import torch
import torch.distributions as D
from typing import Optional
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    leaf: MultivariateGaussian,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> torch.Tensor:
    r"""Computes log-likelihoods for ``MultivariateGaussian`` node in the ``torch`` backend given input data.

    Log-likelihood for ``MultivariateGaussian`` is given by the logarithm of its probability distribution function (PDF):

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
            Boolean value indicating whether or not if the data is in the support of the distribution.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.

    Raises:
        ValueError: Data outside of support.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    batch_size: int = data.shape[0]

    # get information relevant for the scope
    scope_data = data[:, leaf.scope.query]

    # initialize empty tensor (number of output values matches batch_size)
    log_prob: torch.Tensor = torch.empty(batch_size, 1).to(leaf.mean.device)

    # create copy of the data where NaNs are replaced by zeros
    # TODO: alternative for initial validity checking without copying?
    _scope_data = scope_data.clone()
    _scope_data[_scope_data.isnan()] = 0.0

    if check_support:
        # check support
        valid_ids = leaf.check_support(_scope_data, is_scope_data=True).squeeze(
            1
        )

        if not all(valid_ids):
            raise ValueError(
                f"Encountered data instances that are not in the support of the TorchMultivariateGaussian distribution."
            )

    del _scope_data  # free up memory

    # ----- log probabilities -----

    marg = torch.isnan(scope_data)

    # group instances by marginalized random variables
    for marg_mask in marg.unique(dim=0):

        # get all instances with the same (marginalized) scope
        marg_ids = torch.where(
            (marg == marg_mask).sum(dim=-1) == len(leaf.scope.query)
        )[0]
        marg_data = scope_data[marg_ids]

        # all random variables are marginalized over
        if all(marg_mask):
            # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
            log_prob[marg_ids, 0] = 0.0
        # some random variables are marginalized over
        elif any(marg_mask):
            marg_data = marg_data[:, ~marg_mask]

            # marginalize distribution and compute (log) probabilities
            marg_mean = leaf.mean[~marg_mask]
            marg_cov = leaf.cov[~marg_mask][:, ~marg_mask]  # TODO: better way?

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
            # compute probabilities for values inside distribution support
            log_prob[marg_ids, 0] = leaf.dist.log_prob(
                marg_data.type(torch.get_default_dtype())
            )

    return log_prob
