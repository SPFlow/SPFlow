"""
Created on August 15, 2022

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
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.layers.leaves.parametric.gaussian import GaussianLayer as BaseGaussianLayer


class GaussianLayer(Module):
    """Layer representing multiple (univariate) gaussian leaf nodes in the Torch backend.

    Args:
        scope: TODO
        mean: TODO
        std: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], mean: Union[int, float, List[float], np.ndarray, torch.Tensor]=0.0, std: Union[int, float, List[float], np.ndarray, torch.Tensor]=1.0, n_nodes: int=1, **kwargs) -> None:
        """TODO"""
        
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'GaussianLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'GaussianLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")

        super(GaussianLayer, self).__init__(children=[], **kwargs)

        # register auxiliary torch parameter for rate l of each implicit node
        self.mean = Parameter()
        self.std_aux = Parameter()

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.union(s2), self.scopes_out)

        # parse weights
        self.set_params(mean, std)
    
    @property
    def n_out(self) -> int:
        return self._n_out
    
    @property
    def std(self) -> torch.Tensor:
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.std_aux, lb=0.0)  # type: ignore

    def dist(self, node_ids: Optional[List[int]]=None) -> D.Distribution:

        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Normal(loc=self.mean[node_ids], scale=self.std[node_ids])

    def set_params(self, mean: Union[int, float, List[float], np.ndarray, torch.Tensor], std: Union[int, float, List[float], np.ndarray, torch.Tensor]) -> None:
    
        if isinstance(mean, int) or isinstance(mean, float):
            mean = torch.tensor([mean for _ in range(self.n_out)])
        elif isinstance(mean, list) or isinstance(mean, np.ndarray):
            mean = torch.tensor(mean)
        if(mean.ndim != 1):
            raise ValueError(f"Numpy array of 'mean' values for 'GaussianLayer' is expected to be one-dimensional, but is {mean.ndim}-dimensional.")
        if(mean.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'mean' values for 'GaussianLayer' must match number of output nodes {self.n_out}, but is {mean.shape[0]}")
        
        if not torch.any(torch.isfinite(mean)):
            raise ValueError(
                f"Values of 'mean' for 'GaussianLayer' must be finite, but was: {mean}"
            )

        if isinstance(std, int) or isinstance(std, float):
            std = torch.tensor([std for _ in range(self.n_out)])
        elif isinstance(std, list) or isinstance(std, np.ndarray):
            std = torch.tensor(std)
        if(std.ndim != 1):
            raise ValueError(f"Numpy array of 'std' values for 'GaussianLayer' is expected to be one-dimensional, but is {std.ndim}-dimensional.")
        if(std.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'std' values for 'GaussianLayer' must match number of output nodes {self.n_out}, but is {std.shape[0]}")        
        
        if torch.any(std <= 0.0) or not torch.any(torch.isfinite(std)):
            raise ValueError(
                f"Value of 'std' for 'GaussianLayer' must be greater than 0, but was: {std}"
            )

        self.mean.data = mean
        self.std_aux.data = proj_bounded_to_real(std, lb=0.0)

    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.mean, self.std)
    
    def check_support(self, data: torch.Tensor, node_ids: Optional[List[int]]=None) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Gaussian distribution.

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
        if node_ids is None:
            node_ids = list(range(self.n_out))
        
        # all query scopes are univariate
        scope_data = data[:, [self.scopes_out[node_id].query[0] for node_id in node_ids]]

        # NaN values do not throw an error but are simply flagged as False
        valid = self.dist(node_ids).support.check(scope_data)  # type: ignore

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        # set nan_entries back to True
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf()

        return valid


@dispatch(memoize=True)
def marginalize(layer: GaussianLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[GaussianLayer, Gaussian, None]:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    marginalized_node_ids = []
    marginalized_scopes = []

    for i, scope in enumerate(layer.scopes_out):

        # compute marginalized query scope
        marg_scope = [rv for rv in scope.query if rv not in marg_rvs]

        # node not marginalized over
        if len(marg_scope) == 1:
            marginalized_node_ids.append(i)
            marginalized_scopes.append(scope)
    
    if len(marginalized_node_ids) == 0:
        return None
    elif len(marginalized_node_ids) == 1 and prune:
        node_id = marginalized_node_ids.pop()
        return Gaussian(scope=marginalized_scopes[0], mean=layer.mean[node_id].item(), std=layer.std[node_id].item())
    else:
        return GaussianLayer(scope=marginalized_scopes, mean=layer.mean[marginalized_node_ids].detach(), std=layer.std[marginalized_node_ids].detach())


@dispatch(memoize=True)
def toTorch(layer: BaseGaussianLayer, dispatch_ctx: Optional[DispatchContext]=None) -> GaussianLayer:
    return GaussianLayer(scope=layer.scopes_out, mean=layer.mean, std=layer.std)


@dispatch(memoize=True)
def toBase(torch_layer: GaussianLayer, dispatch_ctx: Optional[DispatchContext]=None) -> BaseGaussianLayer:
    return BaseGaussianLayer(scope=torch_layer.scopes_out, mean=torch_layer.mean.detach().numpy(), std=torch_layer.std.detach().numpy())