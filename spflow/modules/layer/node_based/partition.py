"""Contains SPN-like partition layer for SPFlow in the ``base`` backend.
"""
import itertools
from copy import deepcopy
from typing import List, Optional, Union
from collections.abc import Iterable
import numpy as np
import tensorly as tl
import torch

from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.modules.module import Module
from spflow.tensor.ops import Tensor
from spflow.modules.nested_module import NestedModule
from spflow.modules.node import ProductNode

from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class PartitionLayer(NestedModule):
    """Layer representing multiple SPN-like product nodes in the ``base`` backend as combinations of inputs from different partitions.

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

    Attributes:
        children:
            Non-empty list of modules that are children to the node in a directed graph.
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``ProductNode`` objects for the nodes in this layer.
        modules_per_partition:
            List of integers keeping track of the number of total inputs each input partition represents.
        partition_scopes:
            List of scopes keeping track of the scopes each partition represents.
    """

    def __init__(self, child_partitions: list[list[Module]], **kwargs) -> None:
        r"""Initializes ``PartitionLayer`` object.

        Args:
            child_partitions:
                Non-empty list of lists of modules that are children to the layer.
                The output scopes for all child modules in a partition need to be qual.
                The output scopes for different partitions need to be pair-wise disjoint.
        Raises:
            ValueError: Invalid arguments.
        """
        if len(child_partitions) == 0:
            raise ValueError("No partitions for 'PartitionLayer' specified.")

        scope = Scope()
        partition_sizes = []
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
            partition_sizes.append(size)
            self.partition_scopes.append(partition_scope)

            # check if partition is pairwise disjoint to the overall scope so far (makes sure all partitions have pair-wise disjoint scopes)
            if partition_scope.isdisjoint(scope):
                scope = scope.join(partition_scope)
            else:
                raise ValueError("Scopes of partitions must be pair-wise disjoint.")

        super().__init__(children=sum(child_partitions, []), **kwargs)

        self.n_in = sum(partition_sizes)
        self.nodes = []

        # create placeholders and nodes
        for input_ids in itertools.product(
            *np.split(list(range(self.n_in)), np.cumsum(partition_sizes[:-1]))
        ):
            ph = self.create_placeholder(input_ids)
            self.nodes.append(ProductNode(children=[ph]))

        self._n_out = len(self.nodes)
        self.scope = Scope([int(x) for x in scope.query], scope.evidence)

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
        for child in self.children:
            params.extend(list(child.parameters()))
        return params

    def to_dtype(self, dtype):
        self.dtype = dtype
        for node in self.nodes:
            node.dtype = dtype
        for child in self.children:
            child.to_dtype(dtype)

    def to_device(self, device):
        if self.backend == "numpy":
            raise ValueError("it is not possible to change the device of models that have a numpy backend")
        self.device = device
        for node in self.nodes:
            node.device = device
        for child in self.children:
            child.to_device(device)


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: PartitionLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[PartitionLayer, Module, None]:
    """Structural marginalization for SPN-like partition layer objects in the ``base`` backend.

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

        children = layer.children
        partitions = np.split(children, np.cumsum(layer.modules_per_partition[:-1]))

        for partition_scope, partition_children in zip(layer.partition_scopes, partitions):
            partition_children = partition_children.tolist()
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
                        for child in partition_children
                    ]
                )
            else:
                marg_partitions.append(deepcopy(partition_children))

        # if product node has only one input after marginalization and pruning is true, return input directly
        if (
            len(marg_partitions) == 1
            and len(marg_partitions[0]) == 1
            and marg_partitions[0][0].n_out == 1
            and prune
        ):
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

    children = partition_layer.children
    partitions = np.split(children, np.cumsum(partition_layer.modules_per_partition[:-1]))

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return PartitionLayer(
        child_partitions=[
            [updateBackend(child, dispatch_ctx=dispatch_ctx) for child in partition]
            for partition in partitions
        ]
    )


@dispatch(memoize=True)  # type: ignore
def toNodeBased(
    partition_layer: PartitionLayer, dispatch_ctx: Optional[DispatchContext] = None
) -> PartitionLayer:
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        product_node:
            Product node to be converted.
        dispatch_ctx:
            Dispatch context.
    """

    children = partition_layer.children
    partitions = np.split(children, np.cumsum(partition_layer.modules_per_partition[:-1]))

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return PartitionLayer(
        child_partitions=[
            [toNodeBased(child, dispatch_ctx=dispatch_ctx) for child in partition] for partition in partitions
        ]
    )


@dispatch(memoize=True)  # type: ignore
def toLayerBased(partition_layer: PartitionLayer, dispatch_ctx: Optional[DispatchContext] = None):
    from spflow.structure.spn.layer_layerbased import PartitionLayer as PartitionLayerLayer

    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        product_node:
            Product node to be converted.
        dispatch_ctx:
            Dispatch context.
    """

    children = partition_layer.children
    partitions = np.split(children, np.cumsum(partition_layer.modules_per_partition[:-1]))

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return PartitionLayerLayer(
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
    sampling_ctx = init_default_sampling_context(sampling_ctx, T.shape(data)[0])

    # sample accoding to sampling_context
    for node_ids, indices in zip(*sampling_ctx.unique_outputs_ids(return_indices=True)):
        if len(node_ids) != 1 or (len(node_ids) == 0 and partition_layer.n_out != 1):
            raise ValueError("Too many output ids specified for outputs over same scope.")

        node_id = node_ids[0]
        node_instance_ids = T.tensor(sampling_ctx.instance_ids, dtype=int)[indices]

        sample(
            partition_layer.nodes[node_id],
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=SamplingContext(node_instance_ids, [[] for _ in node_instance_ids]),
        )

    return data


@dispatch(memoize=True)  # type: ignore
def em(
    layer: PartitionLayer,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``PartitionLayer`` in the ``torch`` backend.

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

    # recursively call EM on children
    for child in layer.children:
        em(child, data, check_support=check_support, dispatch_ctx=dispatch_ctx)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    partition_layer: PartitionLayer,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    """Computes log-likelihoods for SPN-like partition layers in the 'base' backend given input data.

    Log-likelihoods for product nodes are the sum of its input likelihoods (product in linear space).
    Missing values (i.e., NaN) are marginalized over.

    Args:
        partition_layer:
            Product layer to perform inference for.
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
            for child in partition_layer.children
        ],
        axis=1,
    )

    # set placeholder values
    partition_layer.set_placeholders("log_likelihood", child_lls, dispatch_ctx, overwrite=False)

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return T.concatenate(
        [
            log_likelihood(
                node,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for node in partition_layer.nodes
        ],
        axis=1,
    )