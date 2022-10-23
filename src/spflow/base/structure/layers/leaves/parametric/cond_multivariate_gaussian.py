"""
Created on October 18, 2022

@authors: Philipp Deibert
"""
from typing import List, Union, Optional, Iterable, Tuple, Callable
import numpy as np

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.scope.scope import Scope
from spflow.base.structure.module import Module
from spflow.base.structure.nodes.leaves.parametric.cond_multivariate_gaussian import CondMultivariateGaussian
from spflow.base.structure.nodes.leaves.parametric.cond_gaussian import CondGaussian


class CondMultivariateGaussianLayer(Module):
    """Layer representing multiple conditional multivariate gaussian leaf nodes.

    Args:
        scope: TODO
        cond_f: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], cond_f: Optional[Union[Callable,List[Callable]]]=None, n_nodes: int=1, **kwargs) -> None:
        """TODO"""

        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'CondMultivariateGaussianLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'CondMultivariateGaussianLayer' was empty.")

            self._n_out = len(scope)

        super(CondMultivariateGaussianLayer, self).__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [CondMultivariateGaussian(s) for s in scope]

        # compute scope
        self.scopes_out = scope

        self.set_cond_f(cond_f)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module."""
        return self._n_out
    
    def set_cond_f(self, cond_f: Optional[Union[List[Callable], Callable]]=None) -> None:

        if isinstance(cond_f, List) and len(cond_f) != self.n_out:
            raise ValueError("'CondMultivariateGaussianLayer' received list of 'cond_f' functions, but length does not not match number of conditional nodes.")

        self.cond_f = cond_f
    
    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> Tuple[List[np.ndarray], List[np.ndarray]]:

        mean, cov, cond_f = None, None, None

        # check dispatch cache for required conditional parameters 'mean','cov'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if value for 'mean','cov' specified (highest priority)
            if "mean" in args:
                mean = args["mean"]
            if "cov" in args:
                cov = args["cov"]
            # check if alternative function to provide 'mean','cov' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'mean','cov' specified (lowest priority)
            cond_f = self.cond_f
        
        # if neither 'mean' and 'cov' nor 'cond_f' is specified (via node or arguments)
        if (mean is None or cov is None) and cond_f is None:
            raise ValueError("'CondMultivariateGaussianLayer' requires either 'mean' and 'cov' or 'cond_f' to retrieve 'mean','std' to be specified.")

        # if 'mean' or 'cov' was not already specified, retrieve it
        if mean is None or cov is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, List):
                mean = []
                cov = []

                for f in cond_f:
                    args = f(data)
                    mean.append(args['mean'])
                    cov.append(args['cov'])

            else:
                args = cond_f(data)
                mean = args['mean']
                cov = args['cov']

        if isinstance(mean, list):
            # can be a list of values specifying a single mean (broadcast to all nodes)
            if all([isinstance(m, float) or isinstance(m, int) for m in mean]):
                mean = [np.array(mean) for _ in range(self.n_out)]
            # can also be a list of different means
            else:
                mean = [m if isinstance(m, np.ndarray) else np.array(m) for m in mean]
        elif isinstance(mean, np.ndarray):
            # can be a one-dimensional numpy array specifying single mean (broadcast to all nodes)
            if(mean.ndim == 1):
                mean = [mean for _ in range(self.n_out)]
            # can also be an array of different means
            else:
                mean = [m for m in mean]
        else:
            raise ValueError(f"Specified 'mean' for 'CondMultivariateGaussianLayer' is of unknown type {type(mean)}.")

        if isinstance(cov, list):
            # can be a list of lists of values specifying a single cov (broadcast to all nodes)
            if all([
                all([isinstance(c, float) or isinstance(c, int) for c in l]) for l in cov
            ]):
                cov = [np.array(cov) for _ in range(self.n_out)]
            # can also be a list of different covs
            else:
                cov = [c if isinstance(c, np.ndarray) else np.array(c) for c in cov]
        elif isinstance(cov, np.ndarray):
            # can be a two-dimensional numpy array specifying single cov (broadcast to all nodes)
            if(cov.ndim == 2):
                cov = [cov for _ in range(self.n_out)]
            # can also be an array of different covs
            else:
                cov = [c for c in cov]
        else:
            raise ValueError(f"Specified 'cov' for 'CondMultivariateGaussianLayer' is of unknown type {type(cov)}.")

        if len(mean) != self.n_out:
            raise ValueError(f"Length of list of 'mean' values for 'CondMultivariateGaussianLayer' must match number of output nodes {self.n_out}, but is {len(mean)}")
        if len(cov) != self.n_out:
            raise ValueError(f"Length of list of 'cov' values for 'CondMultivariateGaussianLayer' must match number of output nodes {self.n_out}, but is {len(cov)}")

        for m, c, s in zip(mean, cov, self.scopes_out):
            if(m.ndim != 1):
                raise ValueError(f"All numpy arrays of 'mean' values for 'CondMultivariateGaussianLayer' are expected to be one-dimensional, but at least one is {m.ndim}-dimensional.")
            if(m.shape[0] != len(s.query)):
                raise ValueError(f"Dimensions of a mean vector for 'CondMultivariateGaussianLayer' do not match corresponding scope size.")

            if(c.ndim != 2):
                raise ValueError(f"All numpy arrays of 'cov' values for 'CondMultivariateGaussianLayer' are expected to be two-dimensional, but at least one is {c.ndim}-dimensional.")
            if(c.shape[0] != len(s.query) or c.shape[1] != len(s.query)):
                raise ValueError(f"Dimensions of a covariance matrix for 'CondMultivariateGaussianLayer' do not match corresponding scope size.")

        return mean, cov

    def get_params(self) -> Tuple:
        return tuple([])

    # TODO: check support


@dispatch(memoize=True)
def marginalize(layer: CondMultivariateGaussianLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[CondMultivariateGaussianLayer, CondMultivariateGaussian, CondGaussian, None]:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # marginalize nodes
    marg_nodes = []
    marg_scopes = []

    for node in layer.nodes:
        marg_node = marginalize(node, marg_rvs, prune=prune)

        if marg_node is not None:
            marg_scopes.append(marg_node.scope)
            marg_nodes.append(marg_node)

    if len(marg_scopes) == 0:
        return None
    elif len(marg_scopes) == 1 and prune:
        return marg_nodes.pop()
    else:
        new_layer = CondMultivariateGaussianLayer(marg_scopes)
        return new_layer