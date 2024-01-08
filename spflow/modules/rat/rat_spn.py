"""Contains the SPFlow architecture for Random and Tensorized Sum-Product Networks (RAT-SPNs) in the ``base`` backend.
"""
from typing import List, Optional, Union
from collections.abc import Iterable

import tensorly as tl

from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.tensor.ops import Tensor
from spflow import tensor as T

from spflow.autoleaf import AutoLeaf
from spflow.modules.module import Module
from spflow.modules.layer import CondSumLayer
from spflow.structure.spn.layer.hadamard_layer import HadamardLayer
from spflow.structure.spn.layer.partition_layer import PartitionLayer
from spflow.structure.spn.layer.sum_layer import SumLayer
from spflow.modules.node import CondSumNode
from spflow.modules.node import SumNode, marginalize
from spflow.modules.rat.region_graph import Partition, Region, RegionGraph
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class RatSPN(Module):
    r"""Module architecture for Random and Tensorized Sum-Product Networks (RAT-SPNs) in the ``base`` backend.

    Constructs a RAT-SPN from a specified ``RegionGraph`` instance.
    For details see (Peharz et al., 2020): "Random Sum-Product Networks: A Simple and Effective Approach to Probabilistic Deep Learning".

    Attributes:
        n_root_nodes:
            Integer specifying the number of sum nodes in the root region (C in the original paper).
        n_region_nodes:
            Integer specifying the number of sum nodes in each (non-root) region (S in the original paper).
        n_leaf_ndoes:
            Integer specifying the number of leaf nodes in each leaf region (I in the original paper).
        root_node:
            SPN-like sum node that represents the root of the model.
        root_region:
            SPN-like sum layer that represents the root region of the model.
    """

    def __init__(
        self,
        region_graph: RegionGraph,
        feature_ctx: FeatureContext,
        n_root_nodes: int,
        n_region_nodes: int,
        n_leaf_nodes: int,
    ) -> None:
        r"""Initializer for ``RatSPN`` object.

        Args:
            region_graph:
                ``RegionGraph`` instance to create RAT-SPN architecture from.
            feature_ctx:
                ``FeatureContext`` instance specifying the domains of the scopes.
                Scope must match the region graphs scope.
            n_root_nodes:
                Integer specifying the number of sum nodes in the root region (C in the original paper).
            n_region_nodes:
                Integer specifying the number of sum nodes in each (non-root) region (S in the original paper).
            n_leaf_ndoes:
                Integer specifying the number of leaf nodes in each leaf region (I in the original paper).

        Raises:
            ValueError: Invalid arguments.
        """
        super().__init__(children=[])
        self.n_root_nodes = n_root_nodes
        self.n_region_nodes = n_region_nodes
        self.n_leaf_nodes = n_leaf_nodes
        self.region_graph = region_graph
        self.feature_ctx = feature_ctx

        if n_root_nodes < 1:
            raise ValueError(f"Specified value of 'n_root_nodes' must be at least 1, but is {n_root_nodes}.")
        if n_region_nodes < 1:
            raise ValueError(
                f"Specified value for 'n_region_nodes' must be at least 1, but is {n_region_nodes}."
            )
        if n_leaf_nodes < 1:
            raise ValueError(f"Specified value for 'n_leaf_nodes' must be at least 1, but is {n_leaf_nodes}.")

        # create RAT-SPN from region graph
        self.from_region_graph(region_graph, feature_ctx)

    def from_region_graph(
        self,
        region_graph: RegionGraph,
        feature_ctx: FeatureContext,
    ) -> None:
        r"""Function to create explicit RAT-SPN from an abstract region graph.

        Args:
            region_graph:
                ``RegionGraph`` instance to create RAT-SPN architecture from.
            feature_ctx:
                ``FeatureContext`` instance specifying the domains of the scopes.
                Scope must match the region graphs scope.

        Raises:
            ValueError: Invalid arguments.
        """

        def convert_partition(partition: Partition) -> PartitionLayer:
            return PartitionLayer(
                child_partitions=[
                    [convert_region(region, n_nodes=self.n_region_nodes)] for region in partition.regions
                ]
            )

        def convert_region(region: Region, n_nodes: int) -> Union[SumLayer, HadamardLayer, Module]:
            # non-leaf region
            if region.partitions:
                children = [convert_partition(partition) for partition in region.partitions]
                sum_layer = (
                    CondSumLayer(children=children, n_nodes=n_nodes)
                    if region.scope.is_conditional()
                    else SumLayer(children=children, n_nodes=n_nodes)
                )
                return sum_layer
            # leaf region
            else:
                # split leaf scope into univariate ones and combine them element-wise
                if len(region.scope.query) > 1:
                    partition_signatures = [
                        [feature_ctx.select([rv])] * self.n_leaf_nodes for rv in region.scope.query
                    ]
                    child_partitions = [[AutoLeaf(signatures)] for signatures in partition_signatures]
                    return HadamardLayer(child_partitions=child_partitions)
                # create univariate leaf region
                elif len(region.scope.query) == 1:
                    signatures = [feature_ctx.select(region.scope.query)] * self.n_leaf_nodes
                    return AutoLeaf(signatures)
                else:
                    raise ValueError(
                        "Query scope for region is empty and cannot be converted into appropriate RAT-SPN layer representation."
                    )

        if feature_ctx.scope != region_graph.scope:
            raise ValueError(
                f"Scope of specified feature context {feature_ctx.scope} does not match scope of specified region graph {region_graph.scope}."
            )

        if region_graph.root_region is not None:
            self.root_region = convert_region(region_graph.root_region, n_nodes=self.n_root_nodes)
            self.root_node = (
                CondSumNode(children=[self.root_region])
                if region_graph.scope.is_conditional()
                else SumNode(children=[self.root_region])
            )
        else:
            self.root_region = None
            self.root_node = None

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Returns one since RAT-SPNs always have a single output."""
        return 1

    @property
    def scopes_out(self) -> list[Scope]:
        """Returns the output scopes of the RAT-SPN."""
        return self.root_node.scopes_out

    def to_dtype(self, dtype):
        self.dtype = dtype
        self.root_node.to_dtype(dtype)

    def to_device(self, device):
        if self.backend == "numpy":
            raise ValueError("it is not possible to change the device of models that have a numpy backend")
        self.device = device
        self.root_node.to_device(device)


@dispatch(memoize=True)  # type: ignore
def marginalize(
    rat_spn: RatSPN,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[RatSPN, None]:
    r"""Structural marginalization for ``RatSPN`` objects in the ``base`` backend.

    Args:
        rat_spn:
           ``RatSPN`` instance to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect here. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Raises:
        (Marginalized) RAT-SPN or None (if completely maginalized over).
    """
    # since root node and root region all have the same scope, both are them are either fully marginalized or neither
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    marg_root_node = marginalize(rat_spn.root_node, marg_rvs, prune=False, dispatch_ctx=dispatch_ctx)

    if marg_root_node is None:
        return None
    else:
        # initialize new empty RAT-SPN
        marg_rat = RatSPN(
            RegionGraph(),
            n_root_nodes=rat_spn.n_root_nodes,
            n_region_nodes=rat_spn.n_region_nodes,
            n_leaf_nodes=rat_spn.n_leaf_nodes,
        )
        marg_rat.root_node = marg_root_node
        marg_rat.root_region = marg_root_node.chs[0]

        return marg_rat


@dispatch(memoize=True)  # type: ignore
def updateBackend(rat_spn: RatSPN) -> RatSPN:
    return RatSPN(
        rat_spn.region_graph,
        rat_spn.feature_ctx,
        rat_spn.n_root_nodes,
        rat_spn.n_region_nodes,
        rat_spn.n_leaf_nodes,
    )


@dispatch  # type: ignore
def sample(
    rat_spn: RatSPN,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    r"""Samples from RAT-SPNs in the ``base`` backend given potential evidence.

    Missing values (i.e., NaN) are filled with sampled values.

    Args:
        rat_spn:
            ``RatSpn`` instance to sample from.
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
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, T.shape(data)[0])

    return sample(
        rat_spn.root_node,
        data,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
        sampling_ctx=sampling_ctx,
    )


@dispatch(memoize=True)  # type: ignore
def em(
    rat_spn: RatSPN,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``RatSPN`` in the ``torch`` backend.

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

    # call EM on root node
    em(
        rat_spn.root_node,
        data,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
    )


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    rat_spn: RatSPN,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    """Computes log-likelihoods for RAT-SPNs nodes in the ``base`` backend given input data.

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
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return log_likelihood(
        rat_spn.root_node,
        data,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
    )
