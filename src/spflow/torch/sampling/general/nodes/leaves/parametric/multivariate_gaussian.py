"""Contains sampling methods for ``MultivariateGaussian`` nodes for SPFlow in the ``torch`` backend.
"""
from typing import Optional

import torch

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)
from spflow.torch.structure.general.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian,
)


@dispatch  # type: ignore
def sample(
    leaf: MultivariateGaussian,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> torch.Tensor:
    r"""Samples from ``MultivariateGaussian`` nodes in the ``torch`` backend given potential evidence.

    Samples missing values proportionally to its probability distribution function (PDF).
    If evidence is present, values are sampled from the conitioned marginal distribution.

    Args:
        leaf:
            Leaf node to sample from.
        data:
            Two-dimensional PyTorch tensor containing potential evidence.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
        sampling_ctx:
            Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

    Returns:
        Two-dimensional PyTorch tensor containing the sampled values together with the specified evidence.
        Each row corresponds to a sample.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    if any([i >= data.shape[0] for i in sampling_ctx.instance_ids]):
        raise ValueError("Some instance ids are out of bounds for data tensor.")

    # compute nan_mask for specified instances
    instances_mask = torch.zeros(data.shape[0]).bool()
    instances_mask[torch.tensor(sampling_ctx.instance_ids)] = True

    nan_data = torch.isnan(
        data[
            torch.meshgrid(
                torch.where(instances_mask)[0],
                torch.tensor(leaf.scope.query),
                indexing="ij",
            )
        ]
    )

    # group by scope rvs to sample
    for nan_mask in torch.unique(nan_data, dim=0):

        cond_rvs = torch.tensor(leaf.scope.query)[torch.where(~nan_mask)[0]]  # ids for evidence RVs
        non_cond_rvs = torch.tensor(leaf.scope.query)[torch.where(nan_mask)[0]]  # RVs to be sampled

        # no 'NaN' values (nothing to sample)
        if torch.sum(nan_mask) == 0:
            continue
        # sample from full distribution
        elif torch.sum(nan_mask) == len(leaf.scope.query):
            sampling_ids = torch.tensor(sampling_ctx.instance_ids)[
                (nan_data == nan_mask).sum(dim=1) == nan_mask.shape[0]
            ]

            data[torch.meshgrid(sampling_ids, non_cond_rvs, indexing="ij")] = leaf.dist.sample(
                (sampling_ids.shape[0],)
            ).squeeze(1)
        # sample from conditioned distribution
        else:
            # note: the conditional sampling implemented here is based on the algorithm described in Arnaud Doucet (2010): "A Note on Efficient Conditional Simulation of Gaussian Distributions" (https://www.stats.ox.ac.uk/~doucet/doucet_simulationconditionalgaussian.pdf)
            sampling_ids = torch.tensor(sampling_ctx.instance_ids)[
                (nan_data == nan_mask).sum(dim=1) == nan_mask.shape[0]
            ]

            # sample from full distribution
            joint_samples = leaf.dist.sample((sampling_ids.shape[0],))

            # compute inverse of marginal covariance matrix of conditioning RVs
            marg_cov_inv = torch.linalg.inv(leaf.cov[torch.meshgrid(cond_rvs, cond_rvs, indexing="ij")])

            # get conditional covariance matrix
            cond_cov = leaf.cov[torch.meshgrid(cond_rvs, non_cond_rvs, indexing="ij")]

            data[torch.meshgrid(sampling_ids, non_cond_rvs, indexing="ij")] = joint_samples[:, nan_mask] + (
                (data[torch.meshgrid(sampling_ids, cond_rvs, indexing="ij")] - joint_samples[:, ~nan_mask])
                @ (marg_cov_inv @ cond_cov)
            )

    return data
