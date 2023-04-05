"""Contains sampling methods for ``MultivariateGaussian`` nodes for SPFlow in the ``base`` backend.
"""
from typing import Optional

import tensorly as tl
import numpy as np
from spflow.tensorly.utils.helper_functions import tl_isnan, tl_ix_, tl_inv, tl_unique, T

from spflow.tensorly.structure.general.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian,
)
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)


@dispatch  # type: ignore
def sample(
    leaf: MultivariateGaussian,
    data: T,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> T:
    r"""Samples from ``MultivariateGaussian`` nodes in the ``base`` backend given potential evidence.

    Samples missing values proportionally to its probability distribution function (PDF).
    If evidence is present, values are sampled from the conitioned marginal distribution.

    Args:
        leaf:
            Leaf node to sample from.
        data:
            Two-dimensional NumPy array containing potential evidence.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
        sampling_ctx:
            Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

    Returns:
        Two-dimensional NumPy array containing the sampled values together with the specified evidence.
        Each row corresponds to a sample.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, tl.shape(data)[0])

    if any([i >= tl.shape(data)[0] for i in sampling_ctx.instance_ids]):
        raise ValueError("Some instance ids are out of bounds for data tensor.")

    nan_data = tl_isnan(data[tl_ix_(sampling_ctx.instance_ids, leaf.scope.query)])

    # group by scope rvs to sample
    for nan_mask in tl_unique(nan_data, axis=0):

        cond_mask = ~nan_mask

        # no 'NaN' values (nothing to sample)
        if tl.sum(nan_mask) == 0:
            continue
        # sample from full distribution
        elif tl.sum(nan_mask) == len(leaf.scope.query):
            sampling_ids = tl.tensor(sampling_ctx.instance_ids)[tl.sum((nan_data == nan_mask),axis=1) == nan_mask.shape[0]]

            data[tl_ix_(sampling_ids, leaf.scope.query)] = leaf.dist.rvs(size=sampling_ids.shape[0])
        # sample from conditioned distribution
        else:
            # NOTE: the conditional sampling implemented here is based on the algorithm described in Arnaud Doucet (2010): "A Note on Efficient Conditional Simulation of Gaussian Distributions" (https://www.stats.ox.ac.uk/~doucet/doucet_simulationconditionalgaussian.pdf)
            sampling_ids = tl.tensor(sampling_ctx.instance_ids)[tl.sum((nan_data == nan_mask),axis=1) == nan_mask.shape[0]]

            # sample from full distribution
            joint_samples = leaf.dist.rvs(size=sampling_ids.shape[0])

            # compute inverse of marginal covariance matrix of conditioning RVs
            marg_cov_inv = tl_inv(leaf.cov[tl_ix_(cond_mask, cond_mask)])

            # get conditional covariance matrix
            cond_cov = leaf.cov[tl_ix_(cond_mask, ~cond_mask)]

            data[tl_ix_(sampling_ids, ~cond_mask)] = joint_samples[:, ~cond_mask] + (
                (data[tl_ix_(sampling_ids, cond_mask)] - joint_samples[:, cond_mask]) @ (marg_cov_inv @ cond_cov)
            )

    return data
