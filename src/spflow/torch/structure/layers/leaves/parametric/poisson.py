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
from spflow.torch.structure.nodes.leaves.parametric.poisson import Poisson
from spflow.base.structure.layers.leaves.parametric.poisson import PoissonLayer as BasePoissonLayer


class PoissonLayer(Module):
    """Layer representing multiple (univariate) poisson leaf nodes in the Torch backend.

    Args:
        scope: TODO
        l: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], l: Union[int, float, List[float], np.ndarray, torch.Tensor]=1.0, n_nodes: int=1, **kwargs) -> None:
        """TODO"""
        
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'PoissonLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'PoissonLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")

        super(PoissonLayer, self).__init__(children=[], **kwargs)

        # register auxiliary torch parameter for rate l of each implicit node
        self.l_aux = Parameter()

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.union(s2), self.scopes_out)
    
        # parse weights
        self.set_params(l)

    @property
    def n_out(self) -> int:
        return self._n_out

    @property
    def l(self) -> torch.Tensor:
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.l_aux, lb=0.0)  # type: ignore

    def dist(self, node_ids: Optional[List[int]]=None) -> D.Distribution:

        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Poisson(rate=self.l[node_ids])

    def set_params(self, l: Union[int, float, List[float], np.ndarray, torch.Tensor]) -> None:
    
        if isinstance(l, int) or isinstance(l, float):
            l = torch.tensor([l for _ in range(self.n_out)])
        elif isinstance(l, list) or isinstance(l, np.ndarray):
            l = torch.tensor(l)
        if(l.ndim != 1):
            raise ValueError(f"Numpy array of 'l' values for 'PoissonLayer' is expected to be one-dimensional, but is {l.ndim}-dimensional.")
        if(l.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'l' values for 'PoissonLayer' must match number of output nodes {self.n_out}, but is {l.shape[0]}")
        
        if torch.any(l < 0) or not torch.any(torch.isfinite(l)):
            raise ValueError(
                f"Values for 'l' of 'PoissonLayer' must to greater of equal to 0, but was: {l}"
            )

        self.l_aux.data = proj_bounded_to_real(l, lb=0.0)

    def get_params(self) -> Tuple[torch.Tensor]:
        return (self.l,)
    
    def check_support(self, data: torch.Tensor, node_ids: Optional[List[int]]=None) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Poisson distribution.

        .. math::

            TODO

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

        valid = self.dist(node_ids).support.check(scope_data)  # type: ignore

        # check for infinite values
        mask = valid.clone()
        valid[mask] &= ~scope_data[mask].isinf().sum(dim=-1).bool()

        return valid


@dispatch(memoize=True)
def marginalize(layer: PoissonLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[PoissonLayer, Poisson, None]:
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
        return Poisson(scope=marginalized_scopes[0], l=layer.l[node_id].item())
    else:
        return PoissonLayer(scope=marginalized_scopes, l=layer.l[marginalized_node_ids].detach())


@dispatch(memoize=True)
def toTorch(layer: BasePoissonLayer, dispatch_ctx: Optional[DispatchContext]=None) -> PoissonLayer:
    return PoissonLayer(scope=layer.scopes_out, l=layer.l)


@dispatch(memoize=True)
def toBase(torch_layer: PoissonLayer, dispatch_ctx: Optional[DispatchContext]=None) -> BasePoissonLayer:
    return BasePoissonLayer(scope=torch_layer.scopes_out, l=torch_layer.l.detach().numpy())