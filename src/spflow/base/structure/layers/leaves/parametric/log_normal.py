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
from spflow.base.structure.nodes.leaves.parametric.log_normal import LogNormal


class LogNormalLayer(Module):
    """Layer representing multiple (univariate) log-normal leaf nodes.

    Args:
        scope: TODO
        mean: TODO
        std: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], mean: Union[float, List[float], np.ndarray]=0.0, std: Union[float, List[float], np.ndarray]=1.0, n_nodes: int=1, **kwargs) -> None:
        """TODO"""
        
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'LogNormalLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'LogNormalLayer' was empty.")

            self._n_out = len(scope)
        
        super(LogNormalLayer, self).__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [LogNormal(s, 0.0, 1.0) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(mean, std)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module."""
        return self._n_out

    @property
    def mean(self) -> np.ndarray:
        return np.array([node.mean for node in self.nodes])
    
    @property
    def std(self) -> np.ndarray:
        return np.array([node.std for node in self.nodes])

    def set_params(self, mean: Union[int, float, List[float], np.ndarray]=0.0, std: Union[int, float, List[float], np.ndarray]=1.0) -> None:

        if isinstance(mean, int) or isinstance(mean, float):
            mean = np.array([mean for _ in range(self.n_out)])
        if isinstance(mean, list):
            mean = np.array(mean)
        if(mean.ndim != 1):
            raise ValueError(f"Numpy array of 'mean' values for 'LogNormalLayer' is expected to be one-dimensional, but is {mean.ndim}-dimensional.")
        if(mean.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'mean' values for 'LogNormalLayer' must match number of output nodes {self.n_out}, but is {mean.shape[0]}")

        if isinstance(std, int) or isinstance(std, float):
            std = np.array([float(std) for _ in range(self.n_out)])
        if isinstance(std, list):
            std = np.array(std)
        if(std.ndim != 1):
            raise ValueError(f"Numpy array of 'std' values for 'LogNormalLayer' is expected to be one-dimensional, but is {std.ndim}-dimensional.")
        if(std.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'std' values for 'LogNormalLayer' must match number of output nodes {self.n_out}, but is {std.shape[0]}")

        for node_mean, node_std, node in zip(mean, std, self.nodes):
            node.set_params(node_mean, node_std)
    
    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mean, self.std


@dispatch(memoize=True)
def marginalize(layer: LogNormalLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[LogNormalLayer, LogNormal, None]:
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
        new_node = LogNormal(marg_scopes[0], *marg_params[0])
        return new_node
    else:
        new_layer = LogNormalLayer(marg_scopes, *[np.array(p) for p in zip(*marg_params)])
        return new_layer