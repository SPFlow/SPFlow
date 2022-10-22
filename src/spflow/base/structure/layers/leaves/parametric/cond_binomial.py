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
from spflow.base.structure.nodes.leaves.parametric.cond_binomial import CondBinomial


class CondBinomialLayer(Module):
    """Layer representing multiple conditional (univariate) binomial leaf nodes.

    Args:
        scope: TODO
        n: TODO
        cond_f: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], n: Union[int, List[int], np.ndarray], cond_f: Optional[Union[Callable, List[Callable]]]=None, n_nodes: int=1, **kwargs) -> None:
        """TODO"""
        
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'CondBinomialLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'CondBinomialLayer' was empty.")

            self._n_out = len(scope)
        
        super(CondBinomialLayer, self).__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [CondBinomial(s, 1) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(n)

        self.set_cond_f(cond_f)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module."""
        return self._n_out

    @property
    def n(self) -> np.ndarray:
        return np.array([node.n for node in self.nodes])

    def set_cond_f(self, cond_f: Optional[Union[List[Callable], Callable]]=None) -> None:

        if isinstance(cond_f, List) and len(cond_f) != self.n_out:
            raise ValueError("'CondBinomialLayer' received list of 'cond_f' functions, but length does not not match number of conditional nodes.")

        self.cond_f = cond_f

    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> np.ndarray:

        p, cond_f = None, None

        # check dispatch cache for required conditional parameter 'p'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if a value for 'p' is specified (highest priority)
            if "p" in args:
                p = args["p"]
            # check if alternative function to provide 'p' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'p' specified (lowest priority)
            cond_f = self.cond_f
        
        # if neither 'p' nor 'cond_f' is specified (via node or arguments)
        if p is None and cond_f is None:
            raise ValueError("'CondBinomialLayer' requires either 'p' or 'cond_f' to retrieve 'p' to be specified.")

        # if 'p' was not already specified, retrieve it
        if p is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, List):
                p = np.array([f(data)['p'] for f in cond_f])
            else:
                p = cond_f(data)['p']

        if isinstance(p, int) or isinstance(p, float):
            p = np.array([p for _ in range(self.n_out)])
        if isinstance(p, list):
            p = np.array(p)
        if(p.ndim != 1):
            raise ValueError(f"Numpy array of 'p' values for 'CondBinomialLayer' is expected to be one-dimensional, but is {p.ndim}-dimensional.")
        if(p.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'p' values for 'CondBinomialLayer' must match number of output nodes {self.n_out}, but is {p.shape[0]}")

        return p

    def set_params(self, n: Union[int, List[int], np.ndarray]) -> None:

        if isinstance(n, int):
            n = np.array([n for _ in range(self.n_out)])
        if isinstance(n, list):
            n = np.array(n)
        if(n.ndim != 1):
            raise ValueError(f"Numpy array of 'n' values for 'CondBinomialLayer' is expected to be one-dimensional, but is {n.ndim}-dimensional.")
        if(n.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'n' values for 'CondBinomialLayer' must match number of output nodes {self.n_out}, but is {n.shape[0]}")

        node_scopes = np.array([s.query[0] for s in self.scopes_out])

        for node_scope in np.unique(node_scopes):
            # at least one such element exists
            n_values = n[node_scopes == node_scope]
            if not np.all(n_values == n_values[0]):
                raise ValueError("All values of 'n' for 'CondBinomialLayer' over the same scope must be identical.")

        for node_n, node in zip(n, self.nodes):
            node.set_params(node_n)

    def get_params(self) -> Tuple[np.ndarray]:
        return (self.n,)
    
    # TODO: check support


@dispatch(memoize=True)
def marginalize(layer: CondBinomialLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[CondBinomialLayer, CondBinomial, None]:
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
        new_node = CondBinomial(marg_scopes[0], np.array(marg_params[0]))
        return new_node
    else:
        new_layer = CondBinomialLayer(marg_scopes, np.array(sum(marg_params, tuple())))
        return new_layer