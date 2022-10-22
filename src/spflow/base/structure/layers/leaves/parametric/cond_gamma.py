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
from spflow.base.structure.nodes.leaves.parametric.cond_gamma import CondGamma


class CondGammaLayer(Module):
    """Layer representing multiple conditional (univariate) gamma leaf nodes.

    Args:
        scope: TODO
        cond_f: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], cond_f: Optional[Union[Callable,List[Callable]]]=None, n_nodes: int=1, **kwargs) -> None:
        """TODO"""
        
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'CondGammaLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'CondGammaLayer' was empty.")

            self._n_out = len(scope)
        
        super(CondGammaLayer, self).__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [CondGamma(s) for s in scope]

        # compute scope
        self.scopes_out = scope

        self.set_cond_f(cond_f)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module."""
        return self._n_out
    
    def set_cond_f(self, cond_f: Optional[Union[List[Callable], Callable]]=None) -> None:

        if isinstance(cond_f, List) and len(cond_f) != self.n_out:
            raise ValueError("'CondGammaLayer' received list of 'cond_f' functions, but length does not not match number of conditional nodes.")

        self.cond_f = cond_f
    
    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> Tuple[np.ndarray, np.ndarray]:

        alpha, beta, cond_f = None, None, None

        # check dispatch cache for required conditional parameters 'alpha','beta'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if values 'alpha','beta' are specified (highest priority)
            if "alpha" in args:
                alpha = args["alpha"]
            if "beta" in args:
                beta = args["beta"]
            # check if alternative function to provide 'alpha','beta' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'alpha','beta' specified (lowest priority)
            cond_f = self.cond_f
        
        # if neither 'alpha' or 'beta' nor 'cond_f' is specified (via node or arguments)
        if (alpha is None or beta is None) and cond_f is None:
            raise ValueError("'CondBinomialLayer' requires either 'alpha' and 'beta' or 'cond_f' to retrieve 'alpha','beta to be specified.")

        # if 'alpha' or 'beta' was not already specified, retrieve it
        if alpha is None or beta is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, List):
                alpha = []
                beta = []

                for f in cond_f:
                    args = f(data)
                    alpha.append(args['alpha'])
                    beta.append(args['beta'])

                alpha = np.array(alpha)
                beta = np.array(beta)
            else:
                args = cond_f(data)
                alpha = args['alpha']
                beta = args['beta']

        if isinstance(alpha, int) or isinstance(alpha, float):
            alpha = np.array([alpha for _ in range(self.n_out)])
        if isinstance(alpha, list):
            alpha = np.array(alpha)
        if(alpha.ndim != 1):
            raise ValueError(f"Numpy array of 'alpha' values for 'CondGammaLayer' is expected to be one-dimensional, but is {alpha.ndim}-dimensional.")
        if(alpha.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'alpha' values for 'CondGammaLayer' must match number of output nodes {self.n_out}, but is {alpha.shape[0]}")

        if isinstance(beta, int) or isinstance(beta, float):
            beta = np.array([float(beta) for _ in range(self.n_out)])
        if isinstance(beta, list):
            beta = np.array(beta)
        if(beta.ndim != 1):
            raise ValueError(f"Numpy array of 'beta' values for 'CondGammaLayer' is expected to be one-dimensional, but is {beta.ndim}-dimensional.")
        if(beta.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'beta' values for 'CondGammaLayer' must match number of output nodes {self.n_out}, but is {beta.shape[0]}")

        return alpha, beta
    
    def get_params(self) -> Tuple:
        return tuple([])

    # TODO: check support


@dispatch(memoize=True)
def marginalize(layer: CondGammaLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[CondGammaLayer, CondGamma, None]:
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
        new_node = CondGamma(marg_scopes[0])
        return new_node
    else:
        new_layer = CondGammaLayer(marg_scopes)
        return new_layer