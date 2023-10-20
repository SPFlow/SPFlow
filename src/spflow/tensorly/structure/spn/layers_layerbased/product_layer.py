"""Contains SPN-like product layer for SPFlow in the ``torch`` backend.
"""
from copy import deepcopy
from typing import Iterable, List, Optional, Union

from spflow.base.structure.spn.layers.product_layer import (
    ProductLayer as BaseProductLayer,
)
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.tensorly.structure.module import Module
from spflow.meta.structure import MetaModule


class ProductLayer(Module):
    r"""Layer representing multiple SPN-like product nodes over all children in the ``torch`` backend.

    Represents multiple products of its children over pair-wise disjoint scopes.

    Methods:
        children():
            Iterator over all modules that are children to the module in a directed graph.

    Attributes:
        n_out:
            Integer indicating the number of outputs. Equal to the number of nodes represented by the layer.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(self, n_nodes: int, children: List[MetaModule], **kwargs) -> None:
        r"""Initializes ``ProductLayer`` object.

        Args:
            n_nodes:
                Integer specifying the number of nodes the layer should represent.
            children:
                Non-empty list of modules that are children to the layer.
                The output scopes for all child modules need to be pair-wise disjoint.
        Raises:
            ValueError: Invalid arguments.
        """
        if n_nodes < 1:
            raise ValueError("Number of nodes for 'ProductLayer' must be greater of equal to 1.")

        self._n_out = n_nodes

        if not children:
            raise ValueError("'ProductLayer' requires at least one child to be specified.")

        super().__init__(children=children, **kwargs)

        # compute scope
        scope = Scope()

        for child in children:
            for s in child.scopes_out:
                if not scope.isdisjoint(s):
                    raise ValueError(f"'ProductNode' requires child scopes to be pair-wise disjoint.")

                scope = scope.join(s)

        self.scope = scope

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def scopes_out(self) -> List[Scope]:
        """Returns the output scopes this layer represents."""
        return [self.scope for _ in range(self.n_out)]

    def parameters(self):
        params = []
        for child in self.children:
            params.extend(list(child.parameters()))
        return params


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: ProductLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[ProductLayer, MetaModule, None]:
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

    mutual_rvs = set(layer_scope.query).intersection(set(marg_rvs))

    # layer scope is being fully marginalized over
    if len(mutual_rvs) == len(layer_scope.query):
        return None
    # node scope is being partially marginalized
    elif mutual_rvs:

        marg_children = []

        # marginalize child modules
        for child in layer.children:
            marg_child = marginalize(child, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

            # if marginalized child is not None
            if marg_child:
                marg_children.append(marg_child)

        # if product node has only one child after marginalization and pruning is true, return child directly
        if len(marg_children) == 1 and prune:
            return marg_children[0]
        else:
            return ProductLayer(n_nodes=layer.n_out, children=marg_children)
    else:
        return deepcopy(layer)


@dispatch(memoize=True)  # type: ignore
def updateBackend(product_layer: ProductLayer, dispatch_ctx: Optional[DispatchContext] = None) -> ProductLayer:
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
        children=[updateBackend(child, dispatch_ctx=dispatch_ctx) for child in product_layer.children]
    )

@dispatch(memoize=True)  # type: ignore
def toNodeBased(product_layer: ProductLayer, dispatch_ctx: Optional[DispatchContext] = None):
    from spflow.tensorly.structure.spn.layers import ProductLayer as ProductLayerNode
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
        children=[toNodeBased(child, dispatch_ctx=dispatch_ctx) for child in product_layer.children]
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
        children=[toLayerBased(child, dispatch_ctx=dispatch_ctx) for child in product_layer.children]
    )