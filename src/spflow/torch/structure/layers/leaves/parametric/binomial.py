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
from spflow.torch.structure.nodes.leaves.parametric.binomial import Binomial
from spflow.base.structure.layers.leaves.parametric.binomial import BinomialLayer as BaseBinomialLayer


class BinomialLayer(Module):
    """Layer representing multiple (univariate) binomial leaf nodes in the Torch backend.

    Args:
        scope: TODO
        n: TODO
        p: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], n: Union[int, List[int], np.ndarray, torch.Tensor], p: Union[int, float, List[float], np.ndarray, torch.Tensor]=0.5, n_nodes: int=1, **kwargs) -> None:
        """TODO"""
        
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'BinomialLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'BinomialLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")

        super(BinomialLayer, self).__init__(children=[], **kwargs)
    
        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("n", torch.empty(size=[]))

        # register auxiliary torch parameter for the success probabilities p for each implicit node
        self.p_aux = Parameter()

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.union(s2), self.scopes_out)
    
        # parse weights
        self.set_params(n, p)

    @property
    def n_out(self) -> int:
        return self._n_out

    @property
    def p(self) -> torch.Tensor:
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)  # type: ignore

    @p.setter
    def p(self, p: Union[int, float, List[float], np.ndarray, torch.Tensor]) -> None:

        if isinstance(p, float) or isinstance(p, int):
            p = torch.tensor([p for _ in range(self.n_out)])
        elif isinstance(p, list) or isinstance(p, np.ndarray):
            p = torch.tensor(p)
        if p.ndim != 1:
            raise ValueError(f"Numpy array of 'p' values for 'BinomialLayer' is expected to be one-dimensional, but is {p.ndim}-dimensional.")
        if p.shape[0] == 1:
            p = torch.hstack([p for _ in range(self.n_out)])
        if(p.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'p' values for 'BinomialLayer' must match number of output nodes {self.n_out}, but is {p.shape[0]}")
        if torch.any(p < 0.0) or torch.any(p > 1.0) or not all(torch.isfinite(p)):
            raise ValueError(
                f"Values of 'p' for 'BinomialLayer' distribution must to be between 0.0 and 1.0, but are: {p}"
            )

        self.p_aux.data = proj_bounded_to_real(p, lb=0.0, ub=1.0)
    
    def dist(self, node_ids: Optional[List[int]]=None) -> D.Distribution:

        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Binomial(total_count=self.n[node_ids], probs=self.p[node_ids])

    def set_params(self, n: Union[int, List[int], np.ndarray, torch.Tensor], p: Union[int, float, List[float], np.ndarray, torch.Tensor]=0.5) -> None:
    
        if isinstance(n, int) or isinstance(n, float):
            n = torch.tensor([n for _ in range(self.n_out)])
        elif isinstance(n, list) or isinstance(n, np.ndarray):
            n = torch.tensor(n)
        if(n.ndim != 1):
            raise ValueError(f"Numpy array of 'n' values for 'BinomialLayer' is expected to be one-dimensional, but is {n.ndim}-dimensional.")
        if(n.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'n' values for 'BinomialLayer' must match number of output nodes {self.n_out}, but is {n.shape[0]}")
        
        if torch.any(n < 0) or not torch.any(torch.isfinite(n)):
            raise ValueError(
                f"Values for 'n' of 'BinomialLayer' must to greater of equal to 0, but was: {n}"
            )

        if not torch.all((torch.remainder(n, 1.0) == torch.tensor(0.0))):
            raise ValueError(
                f"Values for 'n' of 'BinomialLayer' must be (equal to) an integer value, but was: {n}"
            )
        
        node_scopes = torch.tensor([s.query[0] for s in self.scopes_out])

        for node_scope in torch.unique(node_scopes):
            # at least one such element exists
            n_values = n[node_scopes == node_scope]
            if not torch.all(n_values == n_values[0]):
                raise ValueError("All values of 'n' for 'BinomialLayer' over the same scope must be identical.")

        self.p = p
        self.n.data = n

    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.p, self.n)
    
    def check_support(self, data: torch.Tensor, node_ids: Optional[List[int]]=None) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Binomial distribution.

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
def marginalize(layer: BinomialLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[BinomialLayer, Binomial, None]:
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
        return Binomial(scope=marginalized_scopes[0], n=layer.n[node_id].item(), p=layer.p[node_id].item())
    else:
        return BinomialLayer(scope=marginalized_scopes, n=layer.n[marginalized_node_ids].detach(), p=layer.p[marginalized_node_ids].detach())


@dispatch(memoize=True)
def toTorch(layer: BaseBinomialLayer, dispatch_ctx: Optional[DispatchContext]=None) -> BinomialLayer:
    return BinomialLayer(scope=layer.scopes_out, n=layer.n, p=layer.p)


@dispatch(memoize=True)
def toBase(torch_layer: BinomialLayer, dispatch_ctx: Optional[DispatchContext]=None) -> BaseBinomialLayer:
    return BaseBinomialLayer(scope=torch_layer.scopes_out, n=torch_layer.n.numpy(), p=torch_layer.p.detach().numpy())