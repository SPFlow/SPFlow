# -*- coding: utf-8 -*-
"""Contains Poisson leaf layer for SPFlow in the 'base' backend.
"""
from typing import List, Union, Optional, Iterable, Tuple
import numpy as np

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.scope.scope import Scope
from spflow.base.structure.module import Module
from spflow.base.structure.nodes.leaves.parametric.poisson import Poisson


class PoissonLayer(Module):
    r"""Layer of multiple (univariate) Poisson distribution leaf node in the 'base' backend.

    Represents multiple univariate Poisson distributions with independent scopes, each with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \lambda^k\frac{e^{-\lambda}}{k!}

    where
        - :math:`k` is the number of occurrences
        - :math:`\lambda` is the rate parameter

    Attributes:
        l:
            One-dimensional NumPy array containing the rate parameters (:math:`\lambda`) for each of the independent Poisson distributions (greater than or equal to 0.0).
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``Poisson`` objects for the nodes in this layer.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], l: Union[int, float, List[float], np.ndarray]=1.0, n_nodes: int=1, **kwargs) -> None:
        r"""Initializes ``PoissonLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            l:
                Floating point, list of floats or one-dimensional NumPy array containing the rate parameters (:math:`\lambda`) for each of the independent Poisson distributions (greater than or equal to 0.0).
                If a single floating point value is given, it is broadcast to all nodes.
                Defaults to 1.0.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'PoissonLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'PoissonLayer' was empty.")

            self._n_out = len(scope)

        super(PoissonLayer, self).__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [Poisson(s) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(l)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out
    
    @property
    def l(self) -> np.ndarray:
        """Returns the rate parameters of the represented distributions."""
        return np.array([node.l for node in self.nodes])

    def set_params(self, l: Union[int, float, List[float], np.ndarray]) -> None:
        r"""Sets the parameters for the represented distributions.

        Args:
            l:
                Floating point, list of floats or one-dimensional NumPy array containing the rate parameters (:math:`\lambda`) for each of the independent Poisson distributions (greater than or equal to 0.0).
                If a single floating point value is given, it is broadcast to all nodes.
                Defaults to 1.0.
        """
        if isinstance(l, int) or isinstance(l, float):
            l = np.array([float(l) for _ in range(self.n_out)])
        if isinstance(l, list):
            l = np.array(l)
        if(l.ndim != 1):
            raise ValueError(f"Numpy array of 'l' values for 'PoissonLayer' is expected to be one-dimensional, but is {l.ndim}-dimensional.")
        if(l.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'l' values for 'PoissonLayer' must match number of output nodes {self.n_out}, but is {l.shape[0]}")

        for node_l, node in zip(l, self.nodes):
            node.set_params(node_l)
    
    def get_params(self) -> Tuple[np.ndarray]:
        """Returns the parameters of the represented distribution.

        Returns:
            One-dimensional NumPy arrays representing the rate parameters.
        """
        return (self.l,)
    
    # TODO: dist

    # TODO: check support


@dispatch(memoize=True)  # type: ignore
def marginalize(layer: PoissonLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[PoissonLayer, Poisson, None]:
    r"""Structural marginalization for ``PoissonLayer`` objects.

    Structurally marginalizes the specified layer module.
    If the layer's scope contains non of the random variables to marginalize, then the layer is returned unaltered.
    If the layer's scope is fully marginalized over, then None is returned.

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
        Unaltered leaf layer or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # marginalize nodes
    marg_scopes = []
    marg_params = []

    for node in layer.nodes:
        marg_node = marginalize(node, marg_rvs, prune=prune)

        if marg_node is not None:
            marg_scopes.append(marg_node.scope)
            marg_params.append(marg_node.get_params())

    if len(marg_scopes) == 0:
        return None
    elif len(marg_scopes) == 1 and prune:
        new_node = Poisson(marg_scopes[0], *marg_params[0])
        return new_node
    else:
        new_layer = PoissonLayer(marg_scopes, *[np.array(p) for p in zip(*marg_params)])
        return new_layer