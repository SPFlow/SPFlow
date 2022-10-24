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
from spflow.torch.structure.nodes.leaves.parametric.uniform import Uniform
from spflow.base.structure.layers.leaves.parametric.uniform import UniformLayer as BaseUniformLayer


class UniformLayer(Module):
    """Layer representing multiple (univariate) uniform leaf nodes in the Torch backend.

    Args:
        scope: TODO
        start: TODO
        end: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], start: Union[int, float, List[float], np.ndarray, torch.Tensor], end: Union[int, float, List[float], np.ndarray, torch.Tensor], support_outside: Union[bool, List[bool], np.ndarray, torch.Tensor]=True, n_nodes: int=1, **kwargs) -> None:
        """TODO"""
        
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'UniformLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'UniformLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")

        super(UniformLayer, self).__init__(children=[], **kwargs)

        # register interval bounds as torch buffers (should not be changed)
        self.register_buffer("start", torch.empty(size=[]))
        self.register_buffer("end", torch.empty(size=[]))
        self.register_buffer("end_next", torch.empty(size=[]))
        self.register_buffer("support_outside", torch.empty(size=[]))

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.union(s2), self.scopes_out)

        # parse weights
        self.set_params(start, end, support_outside)

    @property
    def n_out(self) -> int:
        return self._n_out

    def dist(self, node_ids: Optional[List[int]]=None) -> D.Distribution:

        if node_ids is None:
            node_ids = list(range(self.n_out))
        
        # create Torch distribution with specified parameters
        return D.Uniform(low=self.start[node_ids], high=self.end_next[node_ids])

    def set_params(self, start: Union[int, float, List[float], np.ndarray, torch.Tensor], end: Union[int, float, List[float], np.ndarray, torch.Tensor], support_outside: Union[bool, List[bool], np.ndarray, torch.Tensor]) -> None:

        if isinstance(start, int) or isinstance(start, float):
            start = torch.tensor([start for _ in range(self.n_out)])
        elif isinstance(start, list) or isinstance(start, np.ndarray):
            start = torch.tensor(start)
        if(start.ndim != 1):
            raise ValueError(f"Numpy array of 'start' values for 'UniformLayer' is expected to be one-dimensional, but is {start.ndim}-dimensional.")
        if(start.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'start' values for 'UniformLayer' must match number of output nodes {self.n_out}, but is {start.shape[0]}")
        
        if not torch.any(torch.isfinite(start)):
            raise ValueError(
                f"Values of 'start' for 'UniformLayer' must be finite, but was: {start}"
            )

        if isinstance(end, int) or isinstance(end, float):
            end = torch.tensor([end for _ in range(self.n_out)])
        elif isinstance(end, list) or isinstance(end, np.ndarray):
            end = torch.tensor(end)
        if(end.ndim != 1):
            raise ValueError(f"Numpy array of 'end' values for 'UniformLayer' is expected to be one-dimensional, but is {end.ndim}-dimensional.")
        if(end.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'end' values for 'UniformLayer' must match number of output nodes {self.n_out}, but is {end.shape[0]}")        
        
        if not torch.any(torch.isfinite(end)):
            raise ValueError(
                f"Value of 'end' for 'UniformLayer' must be finite, but was: {end}"
            )
        
        if not torch.all(start < end):
            raise ValueError(
                f"Lower bounds for Uniform distribution must be less than upper bounds, but were: {start}, {end}"
            )
        
        if isinstance(support_outside, bool):
            support_outside = torch.tensor([support_outside for _ in range(self.n_out)])
        elif isinstance(support_outside, list) or isinstance(support_outside, np.ndarray):
            support_outside = torch.tensor(support_outside)
        if(support_outside.ndim != 1):
            raise ValueError(f"Numpy array of 'support_outside' values for 'UniformLayer' is expected to be one-dimensional, but is {support_outside.ndim}-dimensional.")
        if(support_outside.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'support_outside' values for 'UniformLayer' must match number of output nodes {self.n_out}, but is {support_outside.shape[0]}")        
        
        if not torch.any(torch.isfinite(support_outside)):
            raise ValueError(
                f"Value of 'support_outside' for 'UniformLayer' must be greater than 0, but was: {support_outside}"
            )
    
        # since torch Uniform distribution excludes the upper bound, compute next largest number
        end_next = torch.nextafter(end, torch.tensor(float('inf')))

        self.start.data = start
        self.end.data = end
        self.end_next.data = end_next
        self.support_outside.data = support_outside

    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.start, self.end, self.support_outside)
    
    def check_support(self, data: torch.Tensor, node_ids: Optional[List[int]]=None) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Uniform distribution.

        .. math::

            TODO

        Args:
            data:
                Torch tensor containing possible distribution instances.
            node_ids: TODO
        Returns:
            Torch tensor indicating for each possible distribution instance, whether they are part of the support (True) or not (False) of each specified distribution.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))
        
        # all query scopes are univariate
        scope_data = data[:, [self.scopes_out[node_id].query[0] for node_id in node_ids]]

        # torch distribution support is an interval, despite representing a distribution over a half-open interval
        # end is adjusted to the next largest number to make sure that desired end is part of the distribution interval
        # may cause issues with the support check; easier to do a manual check instead
        valid = torch.ones(scope_data.shape, dtype=torch.bool)

        # check if values are within valid range
        valid &= ((scope_data >= self.start[torch.tensor(node_ids)]) & (scope_data < self.end[torch.tensor(node_ids)]))
        valid |= self.support_outside[torch.tensor(node_ids)]

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~(scope_data[~nan_mask & valid].isinf())

        return valid


@dispatch(memoize=True)
def marginalize(layer: UniformLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[UniformLayer, Uniform, None]:
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
        return Uniform(scope=marginalized_scopes[0], start=layer.start[node_id].item(), end=layer.end[node_id].item(), support_outside=layer.support_outside[node_id].item())
    else:
        return UniformLayer(scope=marginalized_scopes, start=layer.start[marginalized_node_ids].detach(), end=layer.end[marginalized_node_ids].detach(), support_outside=layer.support_outside[marginalized_node_ids].detach())


@dispatch(memoize=True)
def toTorch(layer: BaseUniformLayer, dispatch_ctx: Optional[DispatchContext]=None) -> UniformLayer:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return UniformLayer(scope=layer.scopes_out, start=layer.start, end=layer.end, support_outside=layer.support_outside)


@dispatch(memoize=True)
def toBase(torch_layer: UniformLayer, dispatch_ctx: Optional[DispatchContext]=None) -> BaseUniformLayer:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseUniformLayer(scope=torch_layer.scopes_out, start=torch_layer.start.numpy(), end=torch_layer.end.numpy(), support_outside=torch_layer.support_outside.numpy())