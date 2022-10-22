"""
Created on October 18, 2022

@authors: Philipp Deibert
"""
from typing import List, Union, Optional, Iterable, Tuple, Literal, Callable
import numpy as np

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.scope.scope import Scope
from spflow.base.structure.module import Module
from spflow.base.structure.nodes.leaves.parametric.cond_gaussian import CondGaussian


class CondGaussianLayer(Module):
    """Layer representing multiple conditional (univariate) gaussian leaf nodes.

    Args:
        scope: TODO
        cond_f: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], cond_f: Optional[Union[Callable, List[Callable]]]=None, n_nodes: int=1, **kwargs) -> None:
        """TODO"""
        
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'CondGaussianLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'CondGaussianLayer' was empty.")

            self._n_out = len(scope)
        
        super(CondGaussianLayer, self).__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [CondGaussian(s) for s in scope]

        # compute scope
        self.scopes_out = scope

        self.set_cond_f(cond_f)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module."""
        return self._n_out
    
    def set_cond_f(self, cond_f: Optional[Union[List[Callable], Callable]]=None) -> None:

        if isinstance(cond_f, List) and len(cond_f) != self.n_out:
            raise ValueError("'CondGaussianLayer' received list of 'cond_f' functions, but length does not not match number of conditional nodes.")

        self.cond_f = cond_f
    
    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> Tuple[np.ndarray, np.ndarray]:

        mean, std, cond_f = None, None, None

        # check dispatch cache for required conditional parameters 'mean','std'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if value for 'mean','std' specified (highest priority)
            if "mean" in args:
                mean = args["mean"]
            if "std" in args:
                std = args["std"]
            # check if alternative function to provide 'mean','std' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'mean','std' specified (lowest priority)
            cond_f = self.cond_f
        
        # if neither 'mean' and 'std' nor 'cond_f' is specified (via node or arguments)
        if (mean is None or std is None) and cond_f is None:
            raise ValueError("'CondGaussianLayer' requires either 'mean' and 'std' or 'cond_f' to retrieve 'mean','std' to be specified.")

        # if 'mean' or 'std' was not already specified, retrieve it
        if mean is None or std is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, List):
                mean = []
                std = []

                for f in cond_f:
                    args = f(data)
                    mean.append(args['mean'])
                    std.append(args['std'])

                mean = np.array(mean)
                std = np.array(std)
            else:
                args = cond_f(data)
                mean = args['mean']
                std = args['std']

        if isinstance(mean, int) or isinstance(mean, float):
            mean = np.array([mean for _ in range(self.n_out)])
        if isinstance(mean, list):
            mean = np.array(mean)
        if(mean.ndim != 1):
            raise ValueError(f"Numpy array of 'mean' values for 'CondGaussianLayer' is expected to be one-dimensional, but is {mean.ndim}-dimensional.")
        if(mean.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'mean' values for 'CondGaussianLayer' must match number of output nodes {self.n_out}, but is {mean.shape[0]}")

        if isinstance(std, int) or isinstance(std, float):
            std = np.array([float(std) for _ in range(self.n_out)])
        if isinstance(std, list):
            std = np.array(std)
        if(std.ndim != 1):
            raise ValueError(f"Numpy array of 'std' values for 'CondGaussianLayer' is expected to be one-dimensional, but is {std.ndim}-dimensional.")
        if(std.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'std' values for 'CondGaussianLayer' must match number of output nodes {self.n_out}, but is {std.shape[0]}")
        return mean, std

    def get_params(self) -> Tuple:
        return tuple([])
    
    # TODO: check support


@dispatch(memoize=True)
def marginalize(layer: CondGaussianLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[CondGaussianLayer, CondGaussian, None]:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # marginalize nodes
    marg_scopes = []

    for node in layer.nodes:
        marg_node = marginalize(node, marg_rvs, prune=prune)

        if marg_node is not None:
            marg_scopes.append(marg_node.scope)

    if len(marg_scopes) == 0:
        return None
    elif len(marg_scopes) == 1 and prune:
        new_node = CondGaussian(marg_scopes[0])
        return new_node
    else:
        new_layer = CondGaussianLayer(marg_scopes)
        return new_layer