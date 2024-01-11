"""Contains ``SumNode`` for SPFlow in the ``base`` backend.
"""
from copy import deepcopy
from typing import List, Optional, Union
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from torch import Tensor
from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow import log_likelihood

from spflow.modules.node.node import Node
from spflow.modules.module import Module
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.utils.projections import proj_convex_to_real, proj_real_to_convex


class SumNode(Node):
    """SPN-like sum node in the ``base`` backend.

    Represents a convex combination of its inputs over the same scope.

    Attributes:
        inputs:
            Non-empty list of modules that are inputs to the node in a directed graph.
        weights:
            One-dimensional NumPy array containing non-negative weights for each input, summing up to one.
        n_out:
            Integer indicating the number of outputs. One for nodes.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(
        self,
        inputs: list[Module],
        weights: Optional[Union[Tensor, list[float]]] = None,
    ) -> None:
        r"""Initializes ``SumNode`` object.

        Args:
            inputs:
                Non-empty list of modules that are inputs to the node.
                The output scopes for all child modules need to be equal.
            weights:
                Optional list of floats, or one-dimensional NumPy array containing non-negative weights for each input, summing up to one.
                Defaults to 'None' in which case weights are initialized to random weights in (0,1) and normalized.

        Raises:
            ValueError: Invalid arguments.
        """
        super().__init__(inputs=inputs)

        if not inputs:
            raise ValueError("'SumNode' requires at least one child to be specified.")

        scope = None

        for child in inputs:
            for sc in child.scopes_out:
                if scope is None:
                    scope = sc
                else:
                    if not scope.equal_query(sc):
                        raise ValueError(f"'SumNode' requires child scopes to have the same query variables.")

                scope = scope.join(sc)

        self.scope = scope
        self.n_in = sum(child.n_out for child in inputs)

        if weights is None:
            weights = torch.rand(self.n_in) + 1e-08
            weights = weights / weights.sum()
        else:
            weights = torch.tensor(weights)

        if weights.ndim != 1:
            raise ValueError(
                f"Numpy array of weight weights for 'SumNode' is expected to be one-dimensional, but is {weights.ndim}-dimensional."
            )
        if not torch.all(weights > 0):
            raise ValueError("Weights for 'SumNode' must be all positive.")
        if not torch.isclose(torch.sum(weights), torch.tensor(1.0)):
            raise ValueError("Weights for 'SumNode' must sum up to one.")
        if not (len(weights) == self.n_in):
            raise ValueError("Number of weights for 'SumNode' does not match total number of child outputs.")

        self.log_weights = nn.Parameter(proj_convex_to_real(weights))

    @property
    def weights(self) -> Tensor:
        """Returns the weights of the node as a NumPy array."""

        # return self._weights
        return proj_real_to_convex(self.log_weights)

    @weights.setter
    def weights(self, values: Union[Tensor, list[float]]) -> None:
        """Sets the weights of the node to specified values.

        Args:
            values:
                One-dimensional NumPy array or list of floats of non-negative values summing up to one.
                Number of values must match number of total inputs to the node.

        Raises:
            ValueError: Invalid values.
        """
        if isinstance(values, list):
            values = torch.tensor(values, device=self.device)

        values = torch.to(proj_convex_to_real(values), device=self.device)
        self.log_weights.data = values

    def describe_node(self) -> str:
        formatted_weights = [f"{num:.3f}" for num in self.weights.tolist()]
        return f"weights=[{', '.join(formatted_weights)}]"


@dispatch(memoize=True)  # type: ignore
def marginalize(
    sum_node: SumNode,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[SumNode, None]:
    r"""Structural marginalization for ``SumNode`` objects in the ``base`` backend.

    Structurally marginalizes the specified sum node.
    If the sum node's scope contains non of the random variables to marginalize, then the node is returned unaltered.
    If the sum node's scope is fully marginalized over, then None is returned.
    If the sum node's scope is partially marginalized over, then a new sum node over the marginalized child modules is returned.

    Args:
        sum_node:
            Sum node module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect when marginalizing sum nodes. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        (Marginalized) sum node or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute node scope (node only has single output)
    node_scope = sum_node.scope

    mutual_rvs = set(node_scope.query).intersection(set(marg_rvs))

    # node scope is being fully marginalized
    if len(mutual_rvs) == len(node_scope.query):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:
        marg_inputs = []

        # marginalize child modules
        for child in sum_node.inputs:
            marg_child = marginalize(child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

            # if marginalized child is not None
            if marg_child:
                marg_inputs.append(marg_child)

        return SumNode(inputs=marg_inputs, weights=sum_node.weights)
    else:
        return deepcopy(sum_node)


@dispatch(memoize=True)  # type: ignore # ToDo: überprüfen ob sum_layer.weights ein parameter ist
def toLayerBased(sum_node: SumNode, dispatch_ctx: Optional[DispatchContext] = None) -> SumNode:
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    if isinstance(sum_node.weights, np.ndarray):
        return SumNode(
            inputs=[toLayerBased(child, dispatch_ctx=dispatch_ctx) for child in sum_node.inputs],
            weights=torch.tensor(sum_node.weights),
        )
    elif torch.is_tensor(sum_node.weights):
        return SumNode(
            inputs=[toLayerBased(child, dispatch_ctx=dispatch_ctx) for child in sum_node.inputs],
            weights=torch.tensor(sum_node.weights.data),
        )
    else:
        raise NotImplementedError("toLayerBased has no implementation for this backend")


@dispatch(memoize=True)  # type: ignore # ToDo: überprüfen ob sum_layer.weights ein parameter ist
def toNodeBased(sum_node: SumNode, dispatch_ctx: Optional[DispatchContext] = None) -> SumNode:
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    if isinstance(sum_node.weights, np.ndarray):
        return SumNode(
            inputs=[toNodeBased(child, dispatch_ctx=dispatch_ctx) for child in sum_node.inputs],
            weights=torch.tensor(sum_node.weights),
        )
    elif torch.is_tensor(sum_node.weights):
        return SumNode(
            inputs=[toNodeBased(child, dispatch_ctx=dispatch_ctx) for child in sum_node.inputs],
            weights=torch.tensor(sum_node.weights.data),
        )
    else:
        raise NotImplementedError("toNodeBased has no implementation for this backend")


@dispatch  # type: ignore
def sample(
    node: SumNode,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    """Samples from SPN-like sum nodes in the ``base`` backend given potential evidence.

    Samples from each input proportionally to its weighted likelihoods given the evidence.
    Missing values (i.e., NaN) are filled with sampled values.

    Args:
        node:
            Sum node to sample from.
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
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # compute log likelihoods of data instances (TODO: only compute for relevant instances? might clash with cashed values or cashing in general)
    child_lls = torch.cat(
        [
            log_likelihood(
                child,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for child in node.inputs
        ],
        dim=1,
    )

    # take child likelihoods into account when sampling
    # FIXME: this is missing the prior -> posterior renormalization?
    # FIXME: this is not taking the log of the weights
    sampling_weights = node.weights + child_lls[sampling_ctx.instance_ids]

    # sample branch for each instance id
    # this solution is based on a trick described here: https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix/34190035#34190035
    cum_sampling_weights = sampling_weights.cumsum(dim=1)
    random_choices = torch.rand(sampling_weights.shape[0], device=node.device).unsqueeze(1)
    branches = (cum_sampling_weights < random_choices).sum(dim=1)

    # group sampled branches
    for branch in torch.tensor(torch.unique(branches), dtype=int, device=node.device):
        # group instances by sampled branch
        branch_instance_ids = torch.tensor(sampling_ctx.instance_ids, dtype=int, device=node.device)[
            branches == branch
        ]
        # get corresponding child and output id for sampled branch
        child_ids, output_ids = node.input_to_output_ids([branch])

        # sample from child module
        data = sample(
            node.inputs[child_ids[0]],
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=SamplingContext(
                branch_instance_ids,
                [[output_ids[0]] for _ in range(len(branch_instance_ids))],
            ),
        )

    return data


@dispatch(memoize=True)  # type: ignore
def em(
    node: SumNode,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``SumNode`` in the ``torch`` backend.

    Args:
        node:
            Node to perform EM step for.
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
        # node.weights = node.log_weights.grad / node.log_weights.grad.sum()
        # ----- expectation step -----
        child_lls = torch.hstack([dispatch_ctx.cache["log_likelihood"][child] for child in node.inputs])
        node_lls = dispatch_ctx.cache["log_likelihood"][node]
        log_expectations = node.log_weights + node_lls.grad.log() + child_lls - node_lls
        log_expectations = log_expectations.logsumexp(0)
        log_expectations = log_expectations - log_expectations.logsumexp(0)
        # log_expectations = node.log_weights.grad / node.log_weights.grad.sum()
        # print((le2.log() - log_expectations).abs().sum())

        # ----- maximization step -----
        node.log_weights.data = log_expectations

        # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients

    # recursively call EM on inputs
    for input in node.inputs:
        em(input, data, check_support=check_support, dispatch_ctx=dispatch_ctx)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    sum_node: SumNode,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    """Computes log-likelihoods for SPN-like sum nodes in the ``base`` backend given input data.

    Log-likelihood for sum node is the logarithm of the sum of weighted exponentials (LogSumExp) of its input likelihoods (weighted sum in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        sum_node:
            Sum node to perform inference for.
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
    child_lls = torch.cat(
        [
            log_likelihood(
                child,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for child in sum_node.inputs
        ],
        axis=1,
    )
    weighted_inputs = child_lls + sum_node.log_weights
    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return torch.logsumexp(weighted_inputs, dim=-1, keepdims=True)
