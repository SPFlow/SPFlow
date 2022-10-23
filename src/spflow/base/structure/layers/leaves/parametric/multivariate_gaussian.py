"""
Created on August 12, 2022

@authors: Philipp Deibert
"""
from typing import List, Union, Optional, Iterable, Tuple, Literal
import numpy as np

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.scope.scope import Scope
from spflow.meta.types.feature_types import FeatureType
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
            mean = np.zeros((1,len(scope[0].query)))
        if(cov is None):
            cov = np.eye(len(scope[0].query))

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
            mean = np.array(mean)
        if mean.ndim == 1:
            mean = np.vstack([mean for _ in range(self.n_out)])
        if mean.ndim == 2 and mean.shape[0] == 1:
            mean = np.vstack([mean for _ in range(self.n_out)])
        if(mean.ndim != 2):
            raise ValueError(f"Numpy array of 'mean' values for 'MultivariateGaussianLayer' is expected to be two-dimensional, but is {mean.ndim}-dimensional.")
        if(mean.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'mean' values for 'MultivariateGaussianLayer' must match number of output nodes {self.n_out}, but is {mean.shape[0]}")

        if isinstance(cov, list):
            cov = np.array(cov)
        if cov.ndim == 2:
            cov = np.stack([cov for _ in range(self.n_out)])
        if cov.ndim == 3 and cov.shape[0] == 1:
            cov = np.vstack([cov for _ in range(self.n_out)])
        if(cov.ndim != 3):
            raise ValueError(f"Numpy array of 'cov' values for 'MultivariateGaussianLayer' is expected to be three-dimensional, but is {cov.ndim}-dimensional.")
        if(cov.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'cov' values for 'MultivariateGaussianLayer' must match number of output nodes {self.n_out}, but is {cov.shape[0]}")

        for node_mean, node_cov, node in zip(mean, cov, self.nodes):
            node.set_params(node_mean, node_cov)

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mean, self.cov
    
    @classmethod
    def accepts(self, signatures: List[Tuple[List[Literal[FeatureType]], Scope]]) -> bool:  # type: ignore
        # layer has at least one output
        if len(signatures) < 1:
            return False

        # all output signatures should be accepted by Bernoulli leaf nodes
        if not all([MultivariateGaussian.accepts([node_signature]) for node_signature in signatures]):
            return False

        return True
    
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