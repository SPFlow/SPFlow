# -*- coding: utf-8 -*-
"""Contains sampling methods for SPN-like product layers for SPFlow in the ``torch`` backend.
"""
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)
from spflow.torch.structure.spn.layers.product_layer import ProductLayer
from spflow.torch.inference.module import log_likelihood
from spflow.torch.sampling.module import sample

import torch
import numpy as np
from typing import Optional


@dispatch  # type: ignore
def sample(
    product_layer: ProductLayer,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> torch.Tensor:
    """Samples from SPN-like product layers in the ``torch`` backend given potential evidence.

    Can only sample from at most one output at a time, since all scopes are equal and overlap.
    Recursively samples from each input.
    Missing values (i.e., NaN) are filled with sampled values.

    Args:
        product_layer:
            Product layer to sample from.
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

    Raises:
        ValueError: Sampling from invalid number of outputs.
    """
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # all nodes in sum layer have same scope
    if any([len(out) != 1 for out in sampling_ctx.output_ids]):
        raise ValueError("'ProductLayer only allows single output sampling.")

    # all product nodes are over (all) children
    for child in product_layer.children():
        sample(
            child,
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=SamplingContext(
                sampling_ctx.instance_ids,
                [list(range(child.n_out)) for _ in sampling_ctx.instance_ids],
            ),
        )

    return data
