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
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian


class MultivariateGaussianLayer(Module):
    """Layer representing multiple multivariate gaussian leaf nodes.

    Args:
        scope: TODO
        mean: TODO
        cov: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], mean: Optional[Union[List[float], List[List[float]], np.ndarray]]=None, cov: Optional[Union[List[List[float]], List[List[List[float]]], np.ndarray]]=None, n_nodes: int=1, **kwargs) -> None:
        """TODO"""

        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'MultivariateGaussianLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'MultivariateGaussianLayer' was empty.")

            self._n_out = len(scope)

        super(MultivariateGaussianLayer, self).__init__(children=[], **kwargs)

        if(mean is None):
            mean = [np.zeros(len(s.query)) for s in scope]
        if(cov is None):
            cov = [np.eye(len(s.query)) for s in scope]

        # create leaf nodes
        self.nodes = [MultivariateGaussian(s) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(mean, cov)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module."""
        return self._n_out
 
    @property
    def mean(self) -> List[np.ndarray]:
        return [node.mean for node in self.nodes]

    @property
    def cov(self) -> List[np.ndarray]:
        return [node.cov for node in self.nodes]

    def set_params(self, mean: Optional[Union[List[float], List[List[float]], np.ndarray]]=None, cov: Optional[Union[List[List[float]], List[List[List[float]]], np.ndarray]]=None) -> None:

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
            raise ValueError(f"Specified 'mean' for 'MultivariateGaussianLayer' is of unknown type {type(mean)}.")

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
            raise ValueError(f"Specified 'cov' for 'MultivariateGaussianLayer' is of unknown type {type(cov)}.")

        if len(mean) != self.n_out:
            raise ValueError(f"Length of list of 'mean' values for 'MultivariateGaussianLayer' must match number of output nodes {self.n_out}, but is {len(mean)}")
        if len(cov) != self.n_out:
            raise ValueError(f"Length of list of 'cov' values for 'MultivariateGaussianLayer' must match number of output nodes {self.n_out}, but is {len(cov)}")

        for m, c, s in zip(mean, cov, self.scopes_out):
            if(m.ndim != 1):
                raise ValueError(f"All numpy arrays of 'mean' values for 'MultivariateGaussianLayer' are expected to be one-dimensional, but at least one is {m.ndim}-dimensional.")
            if(m.shape[0] != len(s.query)):
                raise ValueError(f"Dimensions of a mean vector for 'MultivariateGaussianLayer' do not match corresponding scope size.")

            if(c.ndim != 2):
                raise ValueError(f"All numpy arrays of 'cov' values for 'MultivariateGaussianLayer' are expected to be two-dimensional, but at least one is {c.ndim}-dimensional.")
            if(c.shape[0] != len(s.query) or c.shape[1] != len(s.query)):
                raise ValueError(f"Dimensions of a covariance matrix for 'MultivariateGaussianLayer' do not match corresponding scope size.")

        for node_mean, node_cov, node in zip(mean, cov, self.nodes):
            node.set_params(node_mean, node_cov)

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mean, self.cov
    
    # TODO: check support


@dispatch(memoize=True)
def marginalize(layer: MultivariateGaussianLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[MultivariateGaussianLayer, MultivariateGaussian, Gaussian, None]:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # marginalize nodes
    marg_nodes = []
    marg_scopes = []
    marg_params = []

    for node in layer.nodes:
        marg_node = marginalize(node, marg_rvs, prune=prune)

        if marg_node is not None:
            marg_scopes.append(marg_node.scope)
            marg_params.append(marg_node.get_params())
            marg_nodes.append(marg_node)

    if len(marg_scopes) == 0:
        return None
    elif len(marg_scopes) == 1 and prune:
        return marg_nodes.pop()
    else:
        new_layer = MultivariateGaussianLayer(marg_scopes, *[np.array(p) for p in zip(*marg_params)])
        return new_layer