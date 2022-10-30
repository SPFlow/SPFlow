# -*- coding: utf-8 -*-
"""Contains learning methods for ``UniformLayer`` leaves for SPFlow in the ``torch`` backend.
"""
from typing import Optional, Union, Callable
import torch
import numpy as np
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.layers.leaves.parametric.uniform import UniformLayer


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    layer: UniformLayer,
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``UniformLayer`` leaves' parameters in the ``torch`` backend.

    All parameters of the Uniform distribution are regarded as fixed and will not be estimated.
    Therefore, this method does nothing, but check for the validity of the data.

    Args:
        leaf:
            Leaf node to estimate parameters of.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        weights:
            Optional one- or two-dimensional PyTorch tensor containing non-negative weights for all data samples and nodes.
            Must match number of samples in ``data``.
            If a one-dimensional PyTorch tensor is given, the weights are broadcast to all nodes.
            Defaults to None in which case all weights are initialized to ones.
        bias_corrections:
            Boolen indicating whether or not to correct possible biases.
            Has no effects for ``Uniform`` nodes.
            Defaults to True.
        nan_strategy:
            Optional string or callable specifying how to handle missing data.
            If 'ignore', missing values (i.e., NaN entries) are ignored.
            If a callable, it is called using ``data`` and should return another PyTorch tensor of same size.
            Defaults to None.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Raises:
        ValueError: Invalid arguments.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # select relevant data for scope
    scope_data = torch.hstack(
        [data[:, scope.query] for scope in layer.scopes_out]
    )

    if check_support:
        if torch.any(~layer.check_support(scope_data, is_scope_data=True)):
            raise ValueError(
                "Encountered values outside of the support for 'UniformLayer'."
            )

    # do nothing since there are no learnable parameters
    pass


@dispatch(memoize=True)  # type: ignore
def em(
    layer: UniformLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``UniformLayer`` in the ``torch`` backend.

    Args:
        layer:
            Leaf layer to perform EM step for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # update parameters through maximum weighted likelihood estimation (NOTE: simply for checking support)
    maximum_likelihood_estimation(
        layer,
        data,
        bias_correction=False,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
    )
