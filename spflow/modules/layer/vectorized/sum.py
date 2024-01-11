"""Contains SPN-like sum layer for SPFlow in the ``torch`` backend.
"""
from copy import deepcopy
from typing import List, Optional, Union
from collections.abc import Iterable

import numpy as np
import tensorly as tl
import torch

from spflow.meta.data.scope import Scope
from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)

from spflow.utils import Tensor
from spflow import tensor as T
from spflow import log_likelihood
from spflow.modules.module import Module
from spflow.modules.node import (
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
        inputs: list[Module],
        weights: Optional[Union[Tensor, list[list[float]], list[float]]] = None,
        **kwargs,
    ) -> None:
        r"""Initializes ``SumLayer`` object.

        Args:
            n_nodes:
                Integer specifying the number of nodes the layer should represent.
            inputs:
                Non-empty list of modules that are inputs to the layer.
                The output scopes for all child modules need to be equal.
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

        if not inputs:
            raise ValueError("'SumLayer' requires at least one child to be specified.")

        super().__init__(inputs=inputs, **kwargs)

        self._n_out = n_nodes
        self.n_in = sum(child.n_out for child in self.inputs)

        # parse weights
        if weights is None:
            weights = (
                T.random.random_tensor((self.n_out, self.n_in), dtype=self.dtype, device=self.device) + 1e-08
            )  # avoid zeros
            weights /= T.sum(weights, axis=-1, keepdims=True)

        # register auxiliary parameters for weights as torch parameters
        if self.backend == "pytorch":
            self._weights = torch.nn.Parameter(requires_grad=True)
        else:
            self._weights = None
        # initialize weights
        self.weights = weights

        # compute scope
        scope = None

        for child in inputs:
            for s in child.scopes_out:
                if scope is None:
                    scope = s
                else:
                    if not scope.equal_query(s):
                        raise ValueError(
                            f"'SumLayer' requires child scopes to have the same query variables."
                        )

                scope = scope.join(s)

        self.scope = scope

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> list[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]

    @property
    def weights(self) -> np.ndarray:
        """Returns the weights of all nodes as a two-dimensional PyTorch tensor."""
        # project auxiliary weights onto weights that sum up to one
        return proj_real_to_convex(self._weights)

    @weights.setter
    def weights(
        self,
        values: Union[Tensor, list[list[float]], list[float]],
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
        if isinstance(values, list) or isinstance(values, np.ndarray):
            values = T.tensor(values, dtype=self.dtype, device=self.device)
        if values.ndim != 1 and values.ndim != 2:
            raise ValueError(
                f"Torch tensor of weight values for 'SumLayer' is expected to be one- or two-dimensional, but is {values.ndim}-dimensional."
            )
        if not T.all(values > 0):
            raise ValueError("Weights for 'SumLayer' must be all positive.")
        if not T.allclose(
            T.tensor(T.sum(values, axis=-1), dtype=self.dtype, device=self.device),
            T.tensor(1.0, dtype=self.dtype, device=self.device),
        ):
            raise ValueError("Weights for 'SumLayer' must sum up to one in last dimension.")
        if not (values.shape[-1] == self.n_in):
            raise ValueError(
                "Number of weights for 'SumLayer' in last dimension does not match total number of child outputs."
            )

        # same weights for all sum nodes
        if self.backend == "pytorch":
            if values.ndim == 1:
                self._weights.data = (
                    proj_convex_to_real(values.repeat((self.n_out, 1)).clone())
                    .type(self.dtype)
                    .to(self.device)
                )
            if values.ndim == 2:
                # same weights for all sum nodes
                if values.shape[0] == 1:
                    self._weights.data = (
                        proj_convex_to_real(values.repeat((self.n_out, 1)).clone())
                        .type(self.dtype)
                        .to(self.device)
                    )
                # different weights for all sum nodes
                elif values.shape[0] == self.n_out:
                    self._weights.data = proj_convex_to_real(values.clone()).type(self.dtype).to(self.device)
                # incorrect number of specified weights
                else:
                    raise ValueError(
                        f"Incorrect number of weights for 'SumLayer'. Size of first dimension must be either 1 or {self.n_out}, but is {values.shape[0]}."
                    )
        elif self.backend == "numpy":
            if values.ndim == 1:
                self._weights = proj_convex_to_real(
                    values.reshape(1, -1).repeat((self.n_out), 0).copy()
                ).astype(self.dtype)
            if values.ndim == 2:
                # same weights for all sum nodes
                if values.shape[0] == 1:
                    self._weights = proj_convex_to_real(values.repeat((self.n_out), 0).copy()).astype(
                        self.dtype
                    )
                # different weights for all sum nodes
                elif values.shape[0] == self.n_out:
                    self._weights = proj_convex_to_real(values.copy()).astype(self.dtype)
                # incorrect number of specified weights
                else:
                    raise ValueError(
                        f"Incorrect number of weights for 'SumLayer'. Size of first dimension must be either 1 or {self.n_out}, but is {values.shape[0]}."
                    )

    def to_dtype(self, dtype):
        self.dtype = dtype
        self.weights = self.weights
        for child in self.inputs:
            child.to_dtype(dtype)

    def to_device(self, device):
        if self.backend == "numpy":
            raise ValueError("it is not possible to change the device of models that have a numpy backend")
        self.device = device
        self.weights = self.weights
        for child in self.inputs:
            child.to_device(device)

    def parameters(self):
        params = []
        for child in self.inputs:
            params.extend(list(child.parameters()))
        params.insert(0, self._weights)
        return params


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
        marg_inputs = []

        # marginalize child modules
        for child in layer.inputs:
            marg_child = marginalize(child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

            # if marginalized child is not None
            if marg_child:
                marg_inputs.append(marg_child)

        return SumLayer(n_nodes=layer.n_out, inputs=marg_inputs, weights=layer.weights)
    else:
        return deepcopy(layer)


@dispatch(memoize=True)  # type: ignore
def updateBackend(sum_layer: SumLayer, dispatch_ctx: Optional[DispatchContext] = None) -> SumLayer:
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    if isinstance(sum_layer.weights, np.ndarray):
        return SumLayer(
            n_nodes=sum_layer.n_out,
            inputs=[updateBackend(child, dispatch_ctx=dispatch_ctx) for child in sum_layer.inputs],
            weights=T.tensor(sum_layer.weights),
        )
    elif torch.is_tensor(sum_layer.weights):
        return SumLayer(
            n_nodes=sum_layer.n_out,
            inputs=[updateBackend(child, dispatch_ctx=dispatch_ctx) for child in sum_layer.inputs],
            weights=T.tensor(sum_layer.weights.data),
        )
    else:
        raise NotImplementedError("updateBackend has no implementation for this backend")


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
        inputs=[toNodeBased(child, dispatch_ctx=dispatch_ctx) for child in sum_layer.inputs],
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
        inputs=[toLayerBased(child, dispatch_ctx=dispatch_ctx) for child in sum_layer.inputs],
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
    if any([len(out) != 1 for out in sampling_ctx.output_ids]):
        raise ValueError("'SumLayer only allows single output sampling.")

    # create mask for instane ids
    instance_ids_mask = T.zeros(data.shape[0], dtype=bool)
    instance_ids_mask[sampling_ctx.instance_ids] = True

    # compute log likelihoods for sum "nodes"
    partition_ll = T.concatenate(
        [
            log_likelihood(
                child,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for child in sum_layer.inputs
        ],
        axis=1,
    )

    inputs = sum_layer.inputs

    for node_id, instances in sampling_ctx.group_output_ids(sum_layer.n_out):
        # sample branches
        input_ids = T.multinomial(
            sum_layer.weights[node_id] * T.exp(partition_ll[instances]),
            num_samples=1,
        ).flatten()

        # get correct child id and corresponding output id
        child_ids, output_ids = sum_layer.input_to_output_ids(input_ids)

        # group by child ids
        for child_id in T.unique(T.tensor(child_ids)):
            child_instance_ids = T.tolist(T.tensor(instances)[T.tensor(child_ids) == child_id])
            child_output_ids = T.tolist(T.unsqueeze(T.tensor(output_ids)[T.tensor(child_ids) == child_id], 1))

            # sample from partition node
            sample(
                inputs[int(child_id)],
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
                sampling_ctx=SamplingContext(child_instance_ids, child_output_ids),
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

    # compute child log-likelihoods
    child_lls = T.concatenate(
        [
            log_likelihood(
                child,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for child in sum_layer.inputs
        ],
        axis=1,
    )

    weighted_lls = T.unsqueeze(child_lls, 1) + T.log(sum_layer.weights)

    return T.logsumexp(weighted_lls, axis=-1)
