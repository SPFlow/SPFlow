"""Contains SPN-like sum layer for SPFlow in the ``base`` backend.
"""
from copy import deepcopy
from typing import List, Optional, Union
from collections.abc import Iterable

import numpy as np

from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.tensor import Tensor
from spflow import tensor as T
from spflow.modules.module import Module
from spflow import log_likelihood
from spflow.modules.nested_module import NestedModule
from spflow.modules.node import SumNode

from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.modules.node import (
    proj_convex_to_real,
    proj_real_to_convex,
)


class SumLayer(NestedModule):
    r"""Layer representing multiple SPN-like sum nodes over all children in the ``base`` backend.

    Represents multiple convex combinations of its children over the same scope.

    Attributes:
        children:
            Non-empty list of modules that are children to the node in a directed graph.
        weights:
            Two-dimensional NumPy array containing non-negative weights for each input.
            Each row corresponds to a sum node with values summing up to one.
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``SumNode`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        n_nodes: int,
        children: list[Module],
        weights: Optional[Union[Tensor, list[list[float]], list[float]]] = None,
        **kwargs,
    ) -> None:
        r"""Initializes ``SumLayer`` object.

        Args:
            n_nodes:
                Integer specifying the number of nodes the layer should represent.
            children:
                Non-empty list of modules that are children to the layer.
                The output scopes for all child modules need to be equal.
            weights:
                Optional list of floats, list of lists of floats or one- to two-dimensional NumPy array,
                containing non-negative weights. There should be weights for each of the node and inputs.
                Each row corresponds to a sum node and values should sum up to one. If it is a list of floats
                or one-dimensional NumPy array, the same weights are reused for all sum nodes.
                Defaults to 'None' in which case weights are initialized to random weights in (0,1) and normalized per row.

        Raises:
            ValueError: Invalid arguments.
        """
        if n_nodes < 1:
            raise ValueError("Number of nodes for 'SumLayer' must be greater of equal to 1.")

        if len(children) == 0:
            raise ValueError("'SumLayer' requires at least one child to be specified.")

        super().__init__(children=children, **kwargs)

        self._n_out = n_nodes
        self.n_in = sum(child.n_out for child in self.children)

        # create input placeholder
        ph = self.create_placeholder(list(range(self.n_in)))

        # create sum nodes
        self.nodes = [SumNode(children=[ph]) for _ in range(n_nodes)]

        # parse weights
        if weights is None:
            weights = (
                T.random.random_tensor((self.n_out, self.n_in), dtype=self.dtype, device=self.device) + 1e-08
            )  # avoid zeros
            weights /= T.sum(weights, axis=-1, keepdims=True)

        if self.backend == "pytorch":
            self._weights = torch.nn.Parameter(requires_grad=True)
        else:
            self._weights = None
        self.weights = weights

        # compute scopee
        self.scope = Scope([int(x) for x in self.nodes[0].scope.query], self.nodes[0].scope.evidence)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> list[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]

    @property
    def weights(self) -> Tensor:
        """Returns the weights of all nodes as a two-dimensional NumPy array."""
        return T.vstack([proj_real_to_convex(node._weights) for node in self.nodes])

    @weights.setter
    def weights(self, values: Union[Tensor, list[list[float]], list[float]]) -> None:
        """Sets the weights of all nodes to specified values.

        Args:
            values:
                List of floats, list of lists of floats or one- to two-dimensional NumPy array,
                containing non-negative weights. There should be weights for each of the node and inputs.
                Each row corresponds to a sum node and values should sum up to one. If it is a list of floats
                or one-dimensional NumPy array, the same weights are reused for all sum nodes.
                Two-dimensional NumPy array containing non-negative weights for each input.

        Raises:
            ValueError: Invalid values.
        """
        if isinstance(values, list):
            values = T.tensor(values, dtype=self.dtype, device=self.device)
        if T.ndim(values) != 1 and T.ndim(values) != 2:
            raise ValueError(
                f"Numpy array of weight values for 'SumLayer' is expected to be one- or two-dimensional, but is {values.ndim}-dimensional."
            )
        if not T.all(values > 0):
            raise ValueError("Weights for 'SumLayer' must be all positive.")
        if not T.allclose(values.sum(axis=-1), 1.0):
            raise ValueError("Weights for 'SumLayer' must sum up to one in last dimension.")
        if not (T.shape(values)[-1] == self.n_in):
            raise ValueError(
                "Number of weights for 'SumLayer' in last dimension does not match total number of child outputs."
            )

        # same weights for all sum nodes
        if T.ndim(values) == 1:
            for node in self.nodes:
                node.weights = T.copy(values)
        if T.ndim(values) == 2:
            # same weights for all sum nodes
            if T.shape(values)[0] == 1:
                for node in self.nodes:
                    if self.backend == "pytorch":
                        node._weights.data = (
                            T.copy(T.squeeze(proj_convex_to_real(values), axis=0))
                            .type(self.dtype)
                            .to(self.device)
                        )
                    else:
                        node._weights = T.copy(T.squeeze(proj_convex_to_real(values), axis=0)).astype(
                            self.dtype
                        )
            # different weights for all sum nodes
            elif values.shape[0] == self.n_out:
                for node, node_values in zip(self.nodes, values):
                    if self.backend == "pytorch":
                        node._weights.data = (
                            T.copy(proj_convex_to_real(node_values)).type(self.dtype).to(self.device)
                        )
                    else:
                        node._weights = T.copy(proj_convex_to_real(node_values)).astype(self.dtype)
            # incorrect number of specified weights
            else:
                raise ValueError(
                    f"Incorrect number of weights for 'SumLayer'. Size of first dimension must be either 1 or {self.n_out}, but is {values.shape[0]}."
                )

    def parameters(self):
        params = []
        for child in self.children:
            params.extend(list(child.parameters()))
        for node in self.nodes:
            params.insert(0, node._weights)
        return params

    def to_dtype(self, dtype):
        self.dtype = dtype
        self.weights = self.weights
        for node in self.nodes:
            node.dtype = dtype
        for child in self.children:
            child.to_dtype(dtype)

    def to_device(self, device):
        if self.backend == "numpy":
            raise ValueError("it is not possible to change the device of models that have a numpy backend")
        self.device = device
        self.weights = self.weights
        for node in self.nodes:
            node.device = device
        for child in self.children:
            child.to_device(device)


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: SumLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[SumLayer, Module, None]:
    """Structural marginalization for SPN-like sum layer objects in the ``base`` backend.

    Structurally marginalizes the specified layer module.
    If the layer's scope contains non of the random variables to marginalize, then the layer is returned unaltered.
    If the layer's scope is fully marginalized over, then None is returned.
    If the layer's scope is partially marginalized over, then a new sum layer over the marginalized child modules is returned.

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

    # compute node scope (node only has single output)
    layer_scope = layer.scope

    mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

    # node scope is being fully marginalized
    if len(mutual_rvs) == len(layer_scope.query):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:
        # TODO: pruning
        marg_children = []

        # marginalize child modules
        for child in layer.children:
            marg_child = marginalize(child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)

        return SumLayer(n_nodes=layer.n_out, children=marg_children, weights=layer.weights)
    else:
        return deepcopy(layer)


@dispatch(memoize=True)  # type: ignore # ToDo: überprüfen ob sum_layer.weights ein parameter ist
def updateBackend(sum_layer: SumLayer, dispatch_ctx: Optional[DispatchContext] = None) -> SumLayer:
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    if isinstance(sum_layer.weights, torch.Tensor):  # torch.nn.parameter.Parameter):
        return SumLayer(
            n_nodes=sum_layer.n_out,
            children=[updateBackend(child, dispatch_ctx=dispatch_ctx) for child in sum_layer.children],
            weights=T.tensor(sum_layer.weights.data, dtype=sum_layer.dtype, device=sum_layer.device),
        )
    elif isinstance(sum_layer.weights, np.ndarray):
        return SumLayer(
            n_nodes=sum_layer.n_out,
            children=[updateBackend(child, dispatch_ctx=dispatch_ctx) for child in sum_layer.children],
            weights=T.tensor(sum_layer.weights, dtype=sum_layer.dtype, device=sum_layer.device),
        )

    else:
        raise NotImplementedError("updateBackend has no implementation for this backend")


@dispatch(memoize=True)  # type: ignore
def toNodeBased(sum_layer: SumLayer, dispatch_ctx: Optional[DispatchContext] = None) -> SumLayer:
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
        children=[toNodeBased(child, dispatch_ctx=dispatch_ctx) for child in sum_layer.children],
        weights=sum_layer.weights,
    )


@dispatch(memoize=True)  # type: ignore
def toLayerBased(sum_layer: SumLayer, dispatch_ctx: Optional[DispatchContext] = None):
    from spflow.structure.spn.layer_layerbased import SumLayer as SumLayerLayer

    """Conversion for ``SumLayer`` from ``layerbased`` to ``nodebased``.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return SumLayerLayer(
        n_nodes=sum_layer.n_out,
        children=[toLayerBased(child, dispatch_ctx=dispatch_ctx) for child in sum_layer.children],
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
    sampling_ctx = init_default_sampling_context(sampling_ctx, T.shape(data)[0])

    # compute log-likelihoods of this module (needed to initialize log-likelihood cache for placeholder)
    log_likelihood(sum_layer, data, check_support=check_support, dispatch_ctx=dispatch_ctx)

    # sample accoding to sampling_context
    for node_ids, indices in zip(*sampling_ctx.unique_outputs_ids(return_indices=True)):
        if len(node_ids) != 1 or (len(node_ids) == 0 and sum_layer.n_out != 1):
            raise ValueError("Too many output ids specified for outputs over same scope.")

        # single node id
        node_id = node_ids[0]
        node_instance_ids = T.tensor(sampling_ctx.instance_ids, dtype=int)[indices]

        sample(
            sum_layer.nodes[int(node_id)],
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=SamplingContext(node_instance_ids, [[] for i in node_instance_ids]),
        )

    return data


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
        child_lls = torch.hstack([dispatch_ctx.cache["log_likelihood"][child] for child in layer.children])

        # TODO: output shape ?
        # get cached log-likelihood gradients w.r.t. module log-likelihoods
        expectations = layer.weights.data * (
            dispatch_ctx.cache["log_likelihood"][layer].grad
            * torch.exp(child_lls)
            / torch.exp(dispatch_ctx.cache["log_likelihood"][layer])
        ).sum(dim=0)

        # ----- maximization step -----
        layer.weights = expectations / expectations.sum(dim=0)

        # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients

    # recursively call EM on children
    for child in layer.children:
        em(child, data, check_support=check_support, dispatch_ctx=dispatch_ctx)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    sum_layer: SumLayer,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    """Computes log-likelihoods for SPN-like sum layers in the ``base`` backend given input data.

    Log-likelihoods for sum nodes are the logarithm of the sum of weighted exponentials (LogSumExp) of its input likelihoods (weighted sum in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        sum_layer:
            Sum layer to perform inference for.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional NumPy array containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute child log-likelihoods
    child_lls = T.concatenate(
        [
            log_likelihood(
                child,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for child in sum_layer.children
        ],
        axis=1,
    )

    # set placeholder values
    sum_layer.set_placeholders("log_likelihood", child_lls, dispatch_ctx, overwrite=False)

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return T.concatenate(
        [
            log_likelihood(
                node,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for node in sum_layer.nodes
        ],
        axis=1,
    )
