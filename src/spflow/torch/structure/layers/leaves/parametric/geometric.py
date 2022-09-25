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
from spflow.torch.structure.nodes.leaves.parametric.geometric import Geometric
from spflow.base.structure.layers.leaves.parametric.geometric import GeometricLayer as BaseGeometricLayer


class GeometricLayer(Module):
    """Layer representing multiple (univariate) geometric leaf nodes in the Torch backend.

    Args:
        scope: TODO
        p: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], p: Union[int, float, List[float], np.ndarray, torch.Tensor]=0.5, n_nodes: int=1, **kwargs) -> None:
        """TODO"""
        
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'GeometricLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'GeometricLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")

        super(GeometricLayer, self).__init__(children=[], **kwargs)

        # register auxiliary torch parameter for success probability p of each implicit node
        self.p_aux = Parameter()

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.union(s2), self.scopes_out)

        # parse weights
        self.set_params(p)

    @property
    def n_out(self) -> int:
        return self._n_out

    @property
    def p(self) -> torch.Tensor:
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)  # type: ignore

    def dist(self, node_ids: Optional[List[int]]=None) -> D.Distribution:

        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Geometric(probs=self.p[node_ids])

    def set_params(self, p: Union[int, float, List[float], np.ndarray, torch.Tensor]) -> None:
    
        if isinstance(p, int) or isinstance(p, float):
            p = torch.tensor([p for _ in range(self.n_out)])
        elif isinstance(p, list) or isinstance(p, np.ndarray):
            p = torch.tensor(p)
        if(p.ndim != 1):
            raise ValueError(f"Numpy array of 'p' values for 'GeometricLayer' is expected to be one-dimensional, but is {p.ndim}-dimensional.")
        if(p.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'p' values for 'GeometricLayer' must match number of output nodes {self.n_out}, but is {p.shape[0]}")
        
        if torch.any(p <= 0) or not torch.any(torch.isfinite(p)):
            raise ValueError(
                f"Values for 'p' of 'GeometricLayer' must to greater of equal to 0, but was: {p}"
            )

        self.p_aux.data = proj_bounded_to_real(p, lb=0.0, ub=1.0)

    def get_params(self) -> Tuple[torch.Tensor]:
        return (self.p,)
    
    def check_support(self, data: torch.Tensor, node_ids: Optional[List[int]]=None) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Geometric distribution.

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
        # data needs to be offset by -1 due to the different definitions between SciPy and PyTorch
        valid = self.dist(node_ids).support.check(scope_data - 1)  # type: ignore

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        # set nan_entries back to True
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf()

        return valid


@dispatch(memoize=True)
def marginalize(layer: GeometricLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[GeometricLayer, Geometric, None]:
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
        return Geometric(scope=marginalized_scopes[0], p=layer.p[node_id].item())
    else:
        return GeometricLayer(scope=marginalized_scopes, p=layer.p[marginalized_node_ids].detach())


@dispatch(memoize=True)
def toTorch(layer: BaseGeometricLayer, dispatch_ctx: Optional[DispatchContext]=None) -> GeometricLayer:
    return GeometricLayer(scope=layer.scopes_out, p=layer.p)


@dispatch(memoize=True)
def toBase(torch_layer: GeometricLayer, dispatch_ctx: Optional[DispatchContext]=None) -> BaseGeometricLayer:
    return BaseGeometricLayer(scope=torch_layer.scopes_out, p=torch_layer.p.detach().numpy())