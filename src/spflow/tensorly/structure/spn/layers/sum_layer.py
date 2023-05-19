"""Contains SPN-like sum layer for SPFlow in the ``base`` backend.
"""
from copy import deepcopy
from typing import Iterable, List, Optional, Union

import tensorly as tl
import torch
from ....utils.helper_functions import tl_vstack, tl_allclose, tl_squeeze, T
from spflow.meta.structure import MetaModule
from spflow.tensorly.structure.module import Module
from spflow.tensorly.structure.nested_module import NestedModule
from spflow.tensorly.structure.spn.nodes.sum_node import SumNode
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.tensorly.structure.spn.nodes.sum_node import (
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
        children: List[MetaModule],
        weights: Optional[Union[T, List[List[float]], List[float]]] = None,
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
            weights = tl.random.random_tensor((self.n_out, self.n_in)) + 1e-08  # avoid zeros
            weights /= tl.sum(weights, axis=-1, keepdims=True)

        if self.backend == "pytorch":
            self._weights = torch.nn.Parameter(requires_grad=True)
        else:
            self._weights = None
        self.weights = weights

        #if weights is not None:
        #    self.weights = weights


        # compute scope
        self.scope = self.nodes[0].scope

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> List[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]

    @property
    def weights(self) -> T:
        """Returns the weights of all nodes as a two-dimensional NumPy array."""
        return tl_vstack([proj_real_to_convex(node._weights) for node in self.nodes])

    @weights.setter
    def weights(self, values: Union[T, List[List[float]], List[float]]) -> None:
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
            values = tl.tensor(values)
        if tl.ndim(values) != 1 and tl.ndim(values) != 2:
            raise ValueError(
                f"Numpy array of weight values for 'SumLayer' is expected to be one- or two-dimensional, but is {values.ndim}-dimensional."
            )
        if not tl.all(values > 0):
            raise ValueError("Weights for 'SumLayer' must be all positive.")
        if not tl_allclose(values.sum(axis=-1), 1.0):
            raise ValueError("Weights for 'SumLayer' must sum up to one in last dimension.")
        if not (tl.shape(values)[-1] == self.n_in):
            raise ValueError(
                "Number of weights for 'SumLayer' in last dimension does not match total number of child outputs."
            )

        # same weights for all sum nodes
        if tl.ndim(values) == 1:
            for node in self.nodes:
                node.weights = tl.copy(values)
        if tl.ndim(values) == 2:
            # same weights for all sum nodes
            if tl.shape(values)[0] == 1:
                for node in self.nodes:
                    if self.backend == "pytorch":
                        node._weights.data = tl.copy(tl_squeeze(proj_convex_to_real(values), axis=0))
                    else:
                        node._weights = tl.copy(tl_squeeze(proj_convex_to_real(values), axis=0))
            # different weights for all sum nodes
            elif values.shape[0] == self.n_out:
                for node, node_values in zip(self.nodes, values):
                    if self.backend == "pytorch":
                        node._weights.data = tl.copy(proj_convex_to_real(node_values))
                    else:
                        node._weights = tl.copy(proj_convex_to_real(node_values))
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
            params.insert(0,node._weights)
        return params

@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: SumLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[SumLayer, MetaModule, None]:
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
