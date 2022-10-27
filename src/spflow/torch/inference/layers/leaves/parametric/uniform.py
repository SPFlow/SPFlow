# -*- coding: utf-8 -*-
"""Contains inference methods for ``UniformLayer`` leaves for SPFlow in the ``torch`` backend.
"""
import torch
import numpy as np
from typing import Optional
from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.torch.structure.layers.leaves.parametric.uniform import UniformLayer


@dispatch(memoize=True)  # type: ignore
def log_likelihood(layer: UniformLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    r"""Computes log-likelihoods for ``UniformLayer`` leaves in the ``torch`` backend given input data.

    Log-likelihood for ``UniformLayer`` is given by the logarithm of its individual probability distribution functions (PDFs):

    .. math::

        \log(\text{PDF}(x)) = \log(\frac{1}{\text{end} - \text{start}}\mathbf{1}_{[\text{start}, \text{end}]}(x))

    where
        - :math:`x` is the input observation
        - :math:`\mathbf{1}_{[\text{start}, \text{end}]}` is the indicator function for the given interval (evaluating to 0 if x is not in the interval)

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

    Raises:
        ValueError: Data outside of support.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    batch_size: int = data.shape[0]

    # initialize empty tensor (number of output values matches batch_size)
    log_prob: torch.Tensor = torch.empty(batch_size, layer.n_out).to(layer.start.device)

    for node_id in range(layer.n_out):

        node_ids_tensor = torch.tensor([node_id])
        node_scope = layer.scopes_out[node_id]
        scope_data = data[:, node_scope.query]

        # ----- marginalization -----

        marg_mask = torch.isnan(scope_data).sum(dim=1) == len(node_scope.query)
        marg_ids = torch.where(marg_mask)[0]

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.meshgrid(marg_ids, node_ids_tensor, indexing='ij')] = 0.0

        # ----- log probabilities -----

        # create masked based on distribution's support
        valid_ids = layer.check_support(data[~marg_mask], node_ids=[node_id])

        # TODO: suppress checks
        if not all(valid_ids):
            raise ValueError(
                f"Encountered data instances that are not in the support of the TorchUniform distribution."
            )

        if layer.support_outside[node_id]:
            torch_valid_mask = torch.zeros(len(marg_mask), dtype=torch.bool)
            torch_valid_mask[~marg_mask] |= layer.dist(node_ids=[node_id]).support.check(scope_data[~marg_mask]).squeeze(1)
            
            outside_interval_ids = torch.where(~marg_mask & ~torch_valid_mask)[0]
            inside_interval_ids = torch.where(~marg_mask & torch_valid_mask)[0]

            # TODO: torch_valid_ids does not necessarily have the same dimension as marg_ids
            log_prob[torch.meshgrid(outside_interval_ids, node_ids_tensor, indexing='ij')] = -float("inf")

            # compute probabilities for values inside distribution support
            log_prob[torch.meshgrid(inside_interval_ids, node_ids_tensor, indexing='ij')] = layer.dist(node_ids=[node_id]).log_prob(
                scope_data[inside_interval_ids].type(torch.get_default_dtype())
            )
        else:
            # compute probabilities for values inside distribution support
            log_prob[~marg_mask] = layer.dist(node_ids=[node_id]).log_prob(
                scope_data[~marg_mask].type(torch.get_default_dtype())
            )
    
    return log_prob