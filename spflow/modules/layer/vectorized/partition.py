"""Contains SPN-like partition layer for SPFlow in the ``torch`` backend.
"""
from copy import deepcopy
from typing import List, Optional, Union
from collections.abc import Iterable

import numpy as np

from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow import tensor as T
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.modules.module import Module
from spflow.tensor.ops import Tensor


class PartitionLayer(Module):
    """Layer representing multiple SPN-like product nodes in the ``torch`` backend as combinations of inputs from different partitions.

    A partition is a group of inputs over the same scope. Different partitions have pair-wise disjoint scopes.
    The layer represents all possible combinations of products selecting a single input from each partition.
    The resulting outputs all have the same scopes.

    Example:

        layer = PartitionLayer([[node1, node2], [node3], [node4, node5, node6]])

        In this example the layer will have 2*1*3=6 product nodes over the following inputs (in this order):

            node1, node3, node4
            node1, node3, node5
            node1, node3, node6
            node2, node3, node4
            node2, node3, node5
            node2, node3, node6

    Methods:
        inputs():
            Iterator over all modules that are inputs to the module in a directed graph.

    Attributes:
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
        modules_per_partition:
            List of integers keeping track of the number of total inputs each input partition represents.
        partition_scopes:
            List of scopes keeping track of the scopes each partition represents.
    """

    def __init__(self, child_partitions: list[list[Module]], **kwargs) -> None:
        r"""Initializes ``PartitionLayer`` object.

        Args:
            child_partitions:
                Non-empty list of lists of modules that are inputs to the layer.
                The output scopes for all child modules in a partition need to be qual.
                The output scopes for different partitions need to be pair-wise disjoint.
        Raises:
            ValueError: Invalid arguments.
        """
        if len(child_partitions) == 0:
            raise ValueError("No partitions for 'PartitionLayer' specified.")

        scope = Scope()
        self.partition_sizes = []
        self.modules_per_partition = []
        self.partition_scopes = []

        # parse partitions
        for partition in child_partitions:
            # check if partition is empty
            if len(partition) == 0:
                raise ValueError("All partitions for 'PartitionLayer' must be non-empty")

            self.modules_per_partition.append(len(partition))
            partition_scope = Scope()
            size = 0

            # iterate over modules in this partition
            for child in partition:
                # increase total number of outputs of this partition
                size += child.n_out

                # for each output scope
                for s in child.scopes_out:
                    # check if query scope is the same
                    if partition_scope.equal_query(s) or partition_scope.isempty():
                        partition_scope = partition_scope.join(s)
                    else:
                        raise ValueError("Scopes of modules inside a partition must have same query scope.")

            # add partition size to list
            self.partition_sizes.append(size)
            self.partition_scopes.append(partition_scope)

            # check if partition is pairwise disjoint to the overall scope so far (makes sure all partitions have pair-wise disjoint scopes)
            if partition_scope.isdisjoint(scope):
                scope = scope.join(partition_scope)
            else:
                raise ValueError("Scopes of partitions must be pair-wise disjoint.")

        super().__init__(inputs=sum(child_partitions, []), **kwargs)

        self.n_in = sum(self.partition_sizes)
        if self.backend == "pytorch":
            self._n_out = int(T.prod(T.tensor(self.partition_sizes)).item())  # instead of item()
        else:
            self._n_out = int(T.prod(T.tensor(self.partition_sizes)))
        self.scope = scope

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> list[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]

    def parameters(self):
        params = []
        for child in self.inputs:
            params.extend(list(child.parameters()))
        return params


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: PartitionLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[PartitionLayer, Module, None]:
    """Structural marginalization for SPN-like partition layer objects in the ``torch`` backend.

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
        (Marginalized) partition layer or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute layer scope (same for all outputs)
    layer_scope = layer.scope

    mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

    # layer scope is being fully marginalized over
    if len(mutual_rvs) == len(layer_scope.query):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:
        marg_partitions = []

        inputs = list(layer.inputs)
        partitions = np.split(inputs, np.cumsum(layer.modules_per_partition[:-1]))

        for partition_scope, partition_inputs in zip(layer.partition_scopes, partitions):
            partition_inputs = T.tolist(partition_inputs)
            partition_mutual_rvs = set(partition_scope.query).intersection(set(marg_rvs))

            # partition scope is being fully marginalized over
            if len(partition_mutual_rvs) == len(partition_scope.query):
                # drop partition entirely
                continue
            # node scope is being partially marginalized
            elif partition_mutual_rvs:
                # marginalize child modules
                marg_partitions.append(
                    [
                        marginalize(
                            child,
                            marg_rvs,
                            prune=prune,
                            dispatch_ctx=dispatch_ctx,
                        )
                        for child in partition_inputs
                    ]
                )
            else:
                marg_partitions.append(deepcopy(partition_inputs))

        # if product node has only one child after marginalization and pruning is true, return child directly
        if len(marg_partitions) == 1 and len(marg_partitions[0]) == 1 and prune:
            return marg_partitions[0][0]
        else:
            return PartitionLayer(child_partitions=marg_partitions)
    else:
        return deepcopy(layer)


@dispatch(memoize=True)  # type: ignore
def updateBackend(
    partition_layer: PartitionLayer, dispatch_ctx: Optional[DispatchContext] = None
) -> PartitionLayer:
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        product_node:
            Product node to be converted.
        dispatch_ctx:
            Dispatch context.
    """

    inputs = partition_layer.inputs
    partitions = np.split(inputs, np.cumsum(partition_layer.modules_per_partition[:-1]))

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return PartitionLayer(
        child_partitions=[
            [updateBackend(child, dispatch_ctx=dispatch_ctx) for child in partition]
            for partition in partitions
        ]
    )


@dispatch(memoize=True)  # type: ignore
def toNodeBased(partition_layer: PartitionLayer, dispatch_ctx: Optional[DispatchContext] = None):
    from spflow.structure.spn.layer import PartitionLayer as PartitionLayerNode

    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        product_node:
            Product node to be converted.
        dispatch_ctx:
            Dispatch context.
    """

    inputs = partition_layer.inputs
    partitions = np.split(inputs, np.cumsum(partition_layer.modules_per_partition[:-1]))

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return PartitionLayerNode(
        child_partitions=[
            [toNodeBased(child, dispatch_ctx=dispatch_ctx) for child in partition] for partition in partitions
        ]
    )


@dispatch(memoize=True)  # type: ignore
def toLayerBased(
    partition_layer: PartitionLayer, dispatch_ctx: Optional[DispatchContext] = None
) -> PartitionLayer:
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        product_node:
            Product node to be converted.
        dispatch_ctx:
            Dispatch context.
    """

    inputs = partition_layer.inputs
    partitions = np.split(inputs, np.cumsum(partition_layer.modules_per_partition[:-1]))

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return PartitionLayer(
        child_partitions=[
            [toLayerBased(child, dispatch_ctx=dispatch_ctx) for child in partition]
            for partition in partitions
        ]
    )


@dispatch  # type: ignore
def sample(
    partition_layer: PartitionLayer,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    """Samples from SPN-like partition layers in the ``torch`` backend given potential evidence.

    Can only sample from at most one output at a time, since all scopes are equal and overlap.
    Recursively samples from each input.
    Missing values (i.e., NaN) are filled with sampled values.

    Args:
        partition_layer:
            Partition layer to sample from.
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
        raise ValueError("'PartitionLayer only allows single output sampling.")

    # TODO: precompute indices
    partition_indices = T.array_split(
        T.arange(0, partition_layer.n_in),
        T.cumsum(T.tensor(partition_layer.partition_sizes), axis=0, dtype=int)[:-1],
    )
    input_ids_per_node = T.cartesian_product(*partition_indices)

    inputs = partition_layer.inputs

    # sample accoding to sampling_context
    for node_id, instances in sampling_ctx.group_output_ids(partition_layer.n_out):
        # get input ids for this node
        input_ids = input_ids_per_node[node_id]
        child_ids, output_ids = partition_layer.input_to_output_ids(T.tolist(input_ids))

        # group by child ids
        for child_id in np.unique(child_ids):
            child_output_ids = np.array(output_ids)[np.array(child_ids) == child_id].tolist()

            # sample from partition node
            sample(
                inputs[int(child_id)],
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
                sampling_ctx=SamplingContext(instances, [child_output_ids for _ in instances]),
            )

    return data


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    partition_layer: PartitionLayer,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    """Computes log-likelihoods for SPN-like partition layers in the ``torch`` backend given input data.

    Log-likelihoods for product nodes are the sum of its input likelihoods (product in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        partition_layer:
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
    child_lls = T.concatenate(
        [
            log_likelihood(
                child,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for child in partition_layer.inputs
        ],
        axis=1,
    )

    # compute all combinations of input indices
    partition_indices = T.array_split(
        T.arange(0, partition_layer.n_in),
        np.cumsum(T.tensor(partition_layer.partition_sizes, dtype=int), axis=0)[:-1],
    )
    indices = T.tensor(T.cartesian_product(*partition_indices), dtype=int)

    # multiply inputs (sum in log-space)
    return T.sum(child_lls[:, indices], axis=-1)
