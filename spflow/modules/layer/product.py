"""Contains SPN-like product layer for SPFlow in the ``torch`` backend.
"""
from copy import deepcopy
from typing import List, Optional, Union
from collections.abc import Iterable

import numpy as np
import torch

from spflow.meta.data.scope import Scope
from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)

from torch import Tensor
from spflow import log_likelihood
from spflow.modules.module import Module
from spflow.utils.projections import (
    proj_convex_to_real,
    proj_real_to_convex,
)


class ProductLayer(Module):
    r"""Layer representing multiple SPN-like product nodes over all inputs in the ``torch`` backend.

    Represents multiple products of its inputs over pair-wise disjoint scopes.

    Methods:
        inputs():
            Iterator over all modules that are inputs to the module in a directed graph.

    Attributes:
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(self, inputs: list[Module], **kwargs) -> None:
        r"""Initializes ``ProductLayer`` object.

        Args:
            n_nodes:
                Integer specifying the number of nodes the layer should represent.
            inputs:
                Non-empty list of modules that are inputs to the layer.
                The output scopes for all child modules need to be pair-wise disjoint.
        Raises:
            ValueError: Invalid arguments.
        """
        super().__init__(inputs=inputs, **kwargs)

        self.n_in = inputs[0].event_shape[-1]
        self.n_scopes = inputs[0].event_shape[-2]
        self._n_out = self.n_in

        self.event_shape = (self.n_in, self.n_scopes, self.n_out)

        if not inputs:
            raise ValueError("'ProductLayer' requires at least one child to be specified.")


        # compute scope
        self.scope = inputs[0].scope


    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> list[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]

    def parameters(self):
        return self.inputs[0].parameters()


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: ProductLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[ProductLayer, Module, None]:
    """Structural marginalization for SPN-like product layer objects in the ``torch`` backend.

    Structurally marginalizes the specified layer module.
    If the layer's scope contains non of the random variables to marginalize, then the layer is returned unaltered.
    If the layer's scope is fully marginalized over, then None is returned.
    If the layer's scope is partially marginalized over, then a new product layer over the marginalized child modules is returned.
    If the marginalized product layer has only one input and 'prune' is set, then the product node is pruned and the input is returned directly.

    Args:
        layer:
            Layer module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            If set to True and the marginalized node has a single input, the input is returned directly.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        (Marginalized) product layer or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute layer scope (same for all outputs)
    layer_scope = layer.scope
    marg_child = None
    mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

    # layer scope is being fully marginalized over
    if len(mutual_rvs) == len(layer_scope.query):
        # passing this loop means marginalizing over the whole scope of this branch
        pass
    # node scope is being partially marginalized
    elif mutual_rvs:

        # marginalize child modules
        marg_child_layer = marginalize(layer.inputs[0], marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

        # if marginalized child is not None
        if marg_child_layer:
            marg_child = marg_child_layer

    else:
        marg_child = layer.inputs[0]
    if marg_child == None:
        return None

    # ToDo: check if this is correct / prune if only one scope is left?
    elif prune and marg_child.event_shape[-2] == 1:
        return marg_child
    else:
        return ProductLayer(inputs=marg_child)

@dispatch(memoize=True)  # type: ignore
def updateBackend(
    product_layer: ProductLayer, dispatch_ctx: Optional[DispatchContext] = None
) -> ProductLayer:
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        product_node:
            Product node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return ProductLayer(
        n_nodes=product_layer.n_out,
        inputs=[updateBackend(child, dispatch_ctx=dispatch_ctx) for child in product_layer.inputs],
    )


@dispatch(memoize=True)  # type: ignore
def toNodeBased(product_layer: ProductLayer, dispatch_ctx: Optional[DispatchContext] = None):
    from spflow.structure.spn.layer import ProductLayer as ProductLayerNode

    """Conversion for ``ProductLayer`` from ``layerbased`` to ``nodebased``.

    Args:
        prduct_layer:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return ProductLayerNode(
        n_nodes=product_layer.n_out,
        inputs=[toNodeBased(child, dispatch_ctx=dispatch_ctx) for child in product_layer.inputs],
    )


@dispatch(memoize=True)  # type: ignore
def toLayerBased(product_layer: ProductLayer, dispatch_ctx: Optional[DispatchContext] = None) -> ProductLayer:
    """Conversion for ``ProductLayer`` from ``layerbased`` to ``nodebased``.

    Args:
        prduct_layer:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return ProductLayer(
        n_nodes=product_layer.n_out,
        inputs=[toLayerBased(child, dispatch_ctx=dispatch_ctx) for child in product_layer.inputs],
    )


@dispatch  # type: ignore
def sample(
    product_layer: ProductLayer,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
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
    #if any([len(out) != 1 for out in sampling_ctx.output_ids]):
    #    raise ValueError("'ProductLayer only allows single output sampling.")


    sample(
        product_layer.inputs[0],
        data,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
        sampling_ctx=sampling_ctx,
    )
    return data


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    product_layer: ProductLayer,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    """Computes log-likelihoods for SPN-like product layers in the ``torch`` backend given input data.

    Log-likelihoods for product nodes are the sum of its input likelihoods (product in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        product_layer:
            Product layer to perform inference for.
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
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    ll = log_likelihood(
            product_layer.inputs[0],
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
        ) # shape: (batch_size, inputs.num_scopes, inputs.num_outputs)

    # multiply childen (sum in log-space)
    return torch.sum(ll, dim=1) # shape: (batch_size, product_layer.num_outputs)
