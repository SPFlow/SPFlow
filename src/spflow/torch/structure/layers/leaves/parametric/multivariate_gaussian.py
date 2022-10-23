"""
Created on October 22, 2022

@authors: Philipp Deibert
"""
from typing import List, Union, Optional, Iterable, Tuple
from functools import reduce
import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from ....nodes.leaves.parametric.projections import proj_bounded_to_real, proj_real_to_bounded

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.scope.scope import Scope
from spflow.torch.structure.module import Module
from spflow.torch.structure.nodes.leaves.parametric.multivariate_gaussian import MultivariateGaussian
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.layers.leaves.parametric.multivariate_gaussian import MultivariateGaussianLayer as BaseMultivariateGaussianLayer


class MultivariateGaussianLayer(Module):
    """Layer representing multiple multivariate gaussian leaf nodes in the Torch backend.

    Args:
        scope: TODO
        mean: TODO
        cov: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], mean: Optional[Union[List[float], List[List[float]], np.ndarray, torch.Tensor]]=None, cov: Optional[Union[List[List[float]], List[List[List[float]]], np.ndarray, torch.Tensor]]=None, n_nodes: int=1, **kwargs) -> None:
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
            mean = [torch.zeros(len(s.query)) for s in scope]
        if(cov is None):
            cov = [torch.eye(len(s.query)) for s in scope]

        # create leaf nodes
        self.nodes = torch.nn.ModuleList([MultivariateGaussian(s) for s in scope])

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.union(s2), self.scopes_out)

        # parse weights
        self.set_params(mean, cov)

    @property
    def n_out(self) -> int:
        return self._n_out
    
    @property
    def mean(self) -> List[np.ndarray]:
        return [node.mean for node in self.nodes]
    
    @property
    def cov(self) -> List[np.ndarray]:
        return [node.cov for node in self.nodes]

    def dist(self, node_ids: Optional[List[int]]=None) -> List[D.Distribution]:
    
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return [self.nodes[i].dist for i in node_ids]

    def set_params(self, mean: Optional[Union[List[float], np.ndarray, torch.Tensor, List[List[float]], List[np.ndarray], List[torch.Tensor]]]=None, cov: Optional[Union[List[List[float]], np.ndarray, torch.Tensor, List[List[List[float]]], List[np.ndarray], List[torch.Tensor]]]=None) -> None:

        if isinstance(mean, list):
            # can be a list of values specifying a single mean (broadcast to all nodes)
            if all([isinstance(m, float) or isinstance(m, int) for m in mean]):
                mean = [np.array(mean) for _ in range(self.n_out)]
            # can also be a list of different means
            else:
                mean = [m if isinstance(m, np.ndarray) else np.array(m) for m in mean]
        elif isinstance(mean, np.ndarray) or isinstance(mean, torch.Tensor):
            # can be a one-dimensional numpy array/torch tensor specifying single mean (broadcast to all nodes)
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
        elif isinstance(cov, np.ndarray) or isinstance(cov, torch.Tensor):
            # can be a two-dimensional numpy array/torch tensor specifying single cov (broadcast to all nodes)
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
                raise ValueError(f"All tensors of 'mean' values for 'MultivariateGaussianLayer' are expected to be one-dimensional, but at least one is {m.ndim}-dimensional.")
            if(m.shape[0] != len(s.query)):
                raise ValueError(f"Dimensions of a mean vector for 'MultivariateGaussianLayer' do not match corresponding scope size.")

            if(c.ndim != 2):
                raise ValueError(f"All tensors of 'cov' values for 'MultivariateGaussianLayer' are expected to be two-dimensional, but at least one is {c.ndim}-dimensional.")
            if(c.shape[0] != len(s.query) or c.shape[1] != len(s.query)):
                raise ValueError(f"Dimensions of a covariance matrix for 'MultivariateGaussianLayer' do not match corresponding scope size.")

        for node_mean, node_cov, node in zip(mean, cov, self.nodes):
            node.set_params(node_mean, node_cov)

    def get_params(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return (self.mean, self.cov)
    
    def check_support(self, data: torch.Tensor, node_ids: Optional[List[int]]=None) -> torch.Tensor:
        r"""Checks if instances are part of the support of the MultivariateGaussian distribution.

        .. math::

            TODO

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Torch tensor containing possible distribution instances.
            node_ids: TODO
        Returns:
            Torch tensor indicating for each possible distribution instance, whether they are part of the support (True) or not (False).
        """

        return [node.check_support(data) for node in self.nodes]


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
        new_layer = MultivariateGaussianLayer(marg_scopes, *[list(p) for p in zip(*marg_params)])
        return new_layer


@dispatch(memoize=True)
def toTorch(layer: BaseMultivariateGaussianLayer, dispatch_ctx: Optional[DispatchContext]=None) -> MultivariateGaussianLayer:
    return MultivariateGaussianLayer(scope=layer.scopes_out, mean=layer.mean, cov=layer.cov)


@dispatch(memoize=True)
def toBase(torch_layer: MultivariateGaussianLayer, dispatch_ctx: Optional[DispatchContext]=None) -> BaseMultivariateGaussianLayer:
    return BaseMultivariateGaussianLayer(scope=torch_layer.scopes_out, mean=[m.detach().numpy() for m in torch_layer.mean], cov=[c.detach().numpy() for c in torch_layer.cov])