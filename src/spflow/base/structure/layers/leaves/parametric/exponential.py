"""
Created on August 12, 2022

@authors: Philipp Deibert
"""
from typing import List, Union, Optional, Iterable, Tuple
import numpy as np

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.scope.scope import Scope
from spflow.base.structure.module import Module
from spflow.base.structure.nodes.leaves.parametric.exponential import Exponential


class ExponentialLayer(Module):
    """Layer representing multiple (univariate) exponential leaf nodes.

    Args:
        scope: TODO
        l: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], l: Union[int, float, List[float], np.ndarray]=1.0, n_nodes: int=1, **kwargs) -> None:
        """TODO"""
        
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'ExponentialLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'ExponentialLayer' was empty.")

            self._n_out = len(scope)

        super(ExponentialLayer, self).__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [Exponential(s) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(l)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module."""
        return self._n_out

    @property
    def l(self) -> np.ndarray:
        return np.array([node.l for node in self.nodes])

    def set_params(self, l: Union[int, float, List[float], np.ndarray]) -> None:

        if isinstance(l, int) or isinstance(l, float):
            l = np.array([float(l) for _ in range(self.n_out)])
        if isinstance(l, list):
            l = np.array(l)
        if(l.ndim != 1):
            raise ValueError(f"Numpy array of 'l' values for 'ExponentialLayer' is expected to be one-dimensional, but is {l.ndim}-dimensional.")
        if(l.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'l' values for 'ExponentialLayer' must match number of output nodes {self.n_out}, but is {l.shape[0]}")

        for node_l, node in zip(l, self.nodes):
            node.set_params(node_l)
    
    def get_params(self) -> Tuple[np.ndarray]:
        return (self.l,)


@dispatch(memoize=True)
def marginalize(layer: ExponentialLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[ExponentialLayer, Exponential, None]:
    """TODO"""
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
        new_node = Exponential(marg_scopes[0], *marg_params[0])
        return new_node
    else:
        new_layer = ExponentialLayer(marg_scopes, *[np.array(p) for p in zip(*marg_params)])
        return new_layer