"""Contains SPN-like sum layer for SPFlow in the ``torch`` backend.
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


class SumLayer(Module):
    r"""Layer representing multiple SPN-like sum nodes over all inputs in the 'base' backend.

    Represents multiple convex combinations of its inputs over the same scope.
    Internally, the weights are represented as unbounded parameters that are projected onto convex combination for each node, representing the actual weights.

    Methods:
        inputs():
            Iterator over all modules that are inputs to the module in a directed graph.

    Attributes:
        weights_aux:
            Two-dimensional PyTorch tensor containing weights for each input and node.
            Each row corresponds to a node.
        weights:
            Two-dimensional PyTorch tensor containing non-negative weights for each input and node, summing up to one (projected from 'weights_aux').
            Each row corresponds to a node.
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(
        self,
        n_nodes: int,
        inputs: Module,
        weights: Tensor = None,
        **kwargs,
    ) -> None:
        r"""Initializes ``SumLayer`` object.

        Args:
            n_nodes:
                Integer specifying the number of nodes the layer should represent.
            inputs:
                Non-empty list of modules that are inputs to the layer.
                The output scopes for all input modules need to be equal.
            weights:
                Optional list of floats, list of lists of floats, one- to two-dimensional NumPy array or two-dimensional
                PyTorch tensor containing non-negative weights. There should be weights for each of the node and inputs.
                Each row corresponds to a sum node and values should sum up to one. If it is a list of floats, a one-dimensional
                NumPy array or a one-dimensonal PyTorch tensor, the same weights are reused for all sum nodes.
                Defaults to 'None' in which case weights are initialized to random weights in (0,1) and normalized per row.

        Raises:
            ValueError: Invalid arguments.
        """



        if n_nodes < 1:
            raise ValueError("Number of nodes for 'SumLayer' must be greater of equal to 1.")

        if not input:
            raise ValueError("'SumLayer' requires at least one input to be specified.")


        self._n_out = n_nodes
        self.n_in = inputs.event_shape[-1] # number of outputs from input
        self.n_scopes = inputs.event_shape[-2] # number of scopes from input
        self.event_shape = (self.n_in, self.n_scopes, self._n_out)

        # compute scope
        self.scope = inputs.scope

        super().__init__(inputs=[inputs], **kwargs)

        self.normalization_dim = 2

        # parse weights
        if weights is None:
            weights = (
                    # weights has shape (n_nodes, n_scopes, n_inputs) to prevent permutation at ll and sample
                    torch.rand((self.event_shape[2], self.event_shape[1], self.event_shape[0]), device=self.device)
                    + 1e-08
            )  # avoid zeros
            weights /= torch.sum(weights, axis=self.normalization_dim, keepdims=True)

        # register auxiliary parameters for weights as torch parameters
        self.logits = torch.nn.Parameter()
        # initialize weights
        self.weights = weights

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> list[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]

    @property
    def log_weights(self) -> Tensor:
        """Returns the weights of all nodes as a two-dimensional PyTorch tensor."""
        # project auxiliary weights onto weights that sum up to one
        return torch.nn.functional.log_softmax(self.logits, dim=self.normalization_dim)

    @property
    def weights(self) -> Tensor:
        """Returns the weights of all nodes as a two-dimensional PyTorch tensor."""
        # project auxiliary weights onto weights that sum up to one
        return torch.nn.functional.softmax(self.logits, dim=self.normalization_dim)

    @weights.setter
    def weights(
        self,
        values: Tensor,
    ) -> None:
        """Sets the weights of all nodes to specified values.

        Args:
            values:
                List of floats, list of lists of floats, one- to two-dimensional NumPy array or two-dimensional
                PyTorch tensor containing non-negative weights. There should be weights for each of the node and inputs.
                Each row corresponds to a sum node and values should sum up to one. If it is a list of floats, a one-dimensional
                NumPy array or a one-dimensonal PyTorch tensor, the same weights are reused for all sum nodes.
                Defaults to 'None' in which case weights are initialized to random weights in (0,1) and normalized per row.

        Raises:
            ValueError: Invalid values.
        """
        if values.shape != (self.event_shape[2], self.event_shape[1], self.event_shape[0]):
            raise ValueError(f"Invalid shape for weights: {values.shape}.")
        if not torch.all(values > 0):
            raise ValueError("Weights for 'SumLayer' must be all positive.")
        """
        if not torch.allclose(
            torch.tensor(torch.sum(values, axis=self.normalization_dim), device=self.device),
            torch.tensor(1.0, device=self.device),
        ):
            raise ValueError("Weights for 'SumLayer' must sum up to one in last dimension.")
        """
        self.logits.data = proj_convex_to_real(values)


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: SumLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[None, SumLayer]:
    """Structural marginalization for SPN-like sum layer objects in the ``torch`` backend.

    Structurally marginalizes the specified layer module.
    If the layer's scope contains non of the random variables to marginalize, then the layer is returned unaltered.
    If the layer's scope is fully marginalized over, then None is returned.
    If the layer's scope is partially marginalized over, then a new sum layer over the marginalized input modules is returned.

    Args:
        layer:
            Layer module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect here. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        (Marginalized) sum layer or None if it is completely marginalized.
    """

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute layer scope (same for all outputs)
    layer_scope = layer.scope
    marg_child = None
    # for idx,s in enumerate(layer_scope):
    mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

    # layer scope is being fully marginalized over
    if len(mutual_rvs) == len(layer_scope.query):
        # passing this loop means marginalizing over the whole scope of this branch
        pass
    # node scope is being partially marginalized
    elif mutual_rvs:
        # marginalize child modules
        # for child in layer.structured_inputs[idx]:
        marg_child_layer = marginalize(layer.inputs[0], marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

        # if marginalized child is not None
        if marg_child_layer:
            marg_child = marg_child_layer

    else:
        marg_child = layer.inputs[0]

    if marg_child == None:
        return None

    else:
        return SumLayer(n_nodes=layer.n_out, inputs=marg_child)


@dispatch(memoize=True)  # type: ignore
def toNodeBased(sum_layer: SumLayer, dispatch_ctx: Optional[DispatchContext] = None):
    from spflow.structure.spn.layer import SumLayer as SumLayerNode

    """Conversion for ``SumLayer`` from ``layerbased`` to ``nodebased``.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return SumLayerNode(
        n_nodes=sum_layer.n_out,
        inputs=[toNodeBased(input, dispatch_ctx=dispatch_ctx) for input in sum_layer.inputs],
        weights=sum_layer.weights,
    )


@dispatch(memoize=True)  # type: ignore
def toLayerBased(sum_layer: SumLayer, dispatch_ctx: Optional[DispatchContext] = None) -> SumLayer:
    """Conversion for ``SumLayer`` from ``layerbased`` to ``nodebased``.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return SumLayer(
        n_nodes=sum_layer.n_out,
        inputs=[toLayerBased(input, dispatch_ctx=dispatch_ctx) for input in sum_layer.inputs],
        weights=sum_layer.weights,
    )


@dispatch  # type: ignore
def sample(
    sum_layer: SumLayer,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    """Samples from SPN-like sum layers in the ``torch`` backend given potential evidence.

    Can only sample from at most one output at a time, since all scopes are equal and overlap.
    Samples from each input proportionally to its weighted likelihoods given the evidence.
    Missing values (i.e., NaN) are filled with sampled values.

    Args:
        sum_layer:
            Sum layer to sample from.
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
    #    raise ValueError("'SumLayer only allows single output sampling.")

    inputs = sum_layer.inputs[0]

    # returns for each output node the input branch to be sampled from
    output_ids = torch.distributions.Categorical(sum_layer.weights[sampling_ctx.output_ids][:,0,...]).sample()

    sample(
        inputs,
        data,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
        sampling_ctx=SamplingContext(sampling_ctx.instance_ids, output_ids),
    )

    return data


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    sum_layer: SumLayer,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    """Computes log-likelihoods for SPN-like sum layers in the ``torch`` backend given input data.

    Log-likelihoods for sum nodes are the logarithm of the sum of weighted exponentials (LogSumExp) of its input likelihoods (weighted sum in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        sum_layer:
            Sum layer to perform inference for.
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

    # compute input log-likelihoods
    ll = log_likelihood( # shape: (batch_size, child_num_scopes, child_num_nodes)
        sum_layer.inputs[0],
        data,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
    )
    weighted_lls = ll.unsqueeze(1) + sum_layer.log_weights.unsqueeze(0)

    # sum over inputs
    return torch.logsumexp(weighted_lls, dim=-1).permute(0,2,1) # shape: (batch_size, num_scopes, num_nodes)

@dispatch(memoize=True)  # type: ignore
def em(
    layer: SumLayer,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``SumLayer`` in the ``torch`` backend.

    Args:
        layer:
            Layer to perform EM step for.
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

    with torch.no_grad():
        # ----- expectation step -----
        child_lls = dispatch_ctx.cache["log_likelihood"][layer.inputs[0]] # shape: (batch_size, 1, num_scopes, num_nodes_child)
        node_lls = dispatch_ctx.cache["log_likelihood"][layer]
        log_expectations = layer.log_weights.unsqueeze(0) + torch.log(node_lls.grad).permute(0,2,1).unsqueeze(-1) + child_lls.unsqueeze(1) - node_lls.permute(0,2,1).unsqueeze(-1)
        log_expectations = log_expectations.logsumexp(0)
        log_expectations = log_expectations - log_expectations.logsumexp(0)

        # ----- maximization step -----
        layer.weights = torch.exp(log_expectations)

        # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients

    em(layer.inputs[0], data, check_support=check_support, dispatch_ctx=dispatch_ctx)