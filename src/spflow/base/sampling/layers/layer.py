# -*- coding: utf-8 -*-
"""Contains sampling methods for SPN-like layers for SPFlow in the ``base`` backend.
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
from spflow.base.structure.layers.layer import (
    SPNSumLayer,
    SPNProductLayer,
    SPNPartitionLayer,
    SPNHadamardLayer,
)
from spflow.base.inference.module import log_likelihood
from spflow.base.sampling.module import sample

import numpy as np
from typing import Optional


@dispatch  # type: ignore
def sample(
    sum_layer: SPNSumLayer,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> np.ndarray:
    """Samples from SPN-like sum layers in the ``base`` backend given potential evidence.

    Can only sample from at most one output at a time, since all scopes are equal and overlap.
    Samples from each input proportionally to its weighted likelihoods given the evidence.
    Missing values (i.e., NaN) are filled with sampled values.

    Args:
        sum_layer:
            Sum layer to sample from.
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

    Raises:
        ValueError: Sampling from invalid number of outputs.
    """
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # compute log-likelihoods of this module (needed to initialize log-likelihood cache for placeholder)
    log_likelihood(
        sum_layer, data, check_support=check_support, dispatch_ctx=dispatch_ctx
    )

    # sample accoding to sampling_context
    for node_ids in np.unique(sampling_ctx.output_ids, axis=0):
        if len(node_ids) != 1 or (len(node_ids) == 0 and sum_layer.n_out != 1):
            raise ValueError(
                "Too many output ids specified for outputs over same scope."
            )

        # single node id
        node_id = node_ids[0]
        node_instance_ids = np.array(sampling_ctx.instance_ids)[
            np.where(sampling_ctx.output_ids == node_ids)[0]
        ].tolist()

        sample(
            sum_layer.nodes[node_id],
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=SamplingContext(
                node_instance_ids, [[] for i in node_instance_ids]
            ),
        )

    return data


@dispatch  # type: ignore
def sample(
    product_layer: SPNProductLayer,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> np.ndarray:
    """Samples from SPN-like product layers in the ``base`` backend given potential evidence.

    Can only sample from at most one output at a time, since all scopes are equal and overlap.
    Recursively samples from each input.
    Missing values (i.e., NaN) are filled with sampled values.

    Args:
        product_layer:
            Product layer to sample from.
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

    Raises:
        ValueError: Sampling from invalid number of outputs.
    """
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    for node_ids in np.unique(sampling_ctx.output_ids, axis=0):
        if len(node_ids) != 1 or (
            len(node_ids) == 0 and product_layer.n_out != 1
        ):
            raise ValueError(
                "Too many output ids specified for outputs over same scope."
            )

    # all product nodes are over (all) children
    for child in product_layer.children:
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


@dispatch  # type: ignore
def sample(
    partition_layer: SPNPartitionLayer,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> np.ndarray:
    """Samples from SPN-like partition layers in the ``base`` backend given potential evidence.

    Can only sample from at most one output at a time, since all scopes are equal and overlap.
    Recursively samples from each input.
    Missing values (i.e., NaN) are filled with sampled values.

    Args:
        partition_layer:
            Partition layer to sample from.
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

    Raises:
        ValueError: Sampling from invalid number of outputs.
    """
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # sample accoding to sampling_context
    for node_ids in np.unique(sampling_ctx.output_ids, axis=0):
        if len(node_ids) != 1 or (
            len(node_ids) == 0 and partition_layer.n_out != 1
        ):
            raise ValueError(
                "Too many output ids specified for outputs over same scope."
            )

        node_id = node_ids[0]
        node_instance_ids = np.array(sampling_ctx.instance_ids)[
            np.where(sampling_ctx.output_ids == node_ids)[0]
        ].tolist()

        sample(
            partition_layer.nodes[node_id],
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=SamplingContext(
                node_instance_ids, [[] for _ in node_instance_ids]
            ),
        )

    return data


@dispatch  # type: ignore
def sample(
    hadamard_layer: SPNHadamardLayer,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> np.ndarray:
    """Samples from SPN-like element-wise product layers in the ``base`` backend given potential evidence.

    Can only sample from at most one output at a time, since all scopes are equal and overlap.
    Recursively samples from each input.
    Missing values (i.e., NaN) are filled with sampled values.

    Args:
        hadamard_layer:
            Hadamard layer to sample from.
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

    Raises:
        ValueError: Sampling from invalid number of outputs.
    """
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # sample accoding to sampling_context
    for node_ids in np.unique(sampling_ctx.output_ids, axis=0):
        if len(node_ids) != 1 or (
            len(node_ids) == 0 and hadamard_layer.n_out != 1
        ):
            raise ValueError(
                "Too many output ids specified for outputs over same scope."
            )

        node_id = node_ids[0]
        node_instance_ids = np.array(sampling_ctx.instance_ids)[
            np.where(sampling_ctx.output_ids == node_ids)[0]
        ].tolist()

        sample(
            hadamard_layer.nodes[node_id],
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=SamplingContext(
                node_instance_ids, [[] for _ in node_instance_ids]
            ),
        )

    return data
