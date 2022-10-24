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
from spflow.torch.structure.nodes.leaves.parametric.gamma import Gamma
from spflow.base.structure.layers.leaves.parametric.gamma import GammaLayer as BaseGammaLayer


class GammaLayer(Module):
    """Layer representing multiple (univariate) gamma leaf nodes in the Torch backend.

    Args:
        scope: TODO
        alpha: TODO
        beta: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], alpha: Union[int, float, List[float], np.ndarray, torch.Tensor]=1.0, beta: Union[int, float, List[float], np.ndarray, torch.Tensor]=1.0, n_nodes: int=1, **kwargs) -> None:
        """TODO"""
        
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'GammaLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'GammaLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")

        super(GammaLayer, self).__init__(children=[], **kwargs)

        # register auxiliary torch parameter for rate l of each implicit node
        self.alpha_aux = Parameter()
        self.beta_aux = Parameter()

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.union(s2), self.scopes_out)
    
        # parse weights
        self.set_params(alpha, beta)
    
    @property
    def n_out(self) -> int:
        return self._n_out
    
    @property
    def alpha(self) -> torch.Tensor:
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.alpha_aux, lb=0.0)  # type: ignore

    @property
    def beta(self) -> torch.Tensor:
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.beta_aux, lb=0.0)  # type: ignore

    def dist(self, node_ids: Optional[List[int]]=None) -> D.Distribution:

        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Gamma(concentration=self.alpha[node_ids], rate=self.beta[node_ids])

    def set_params(self, alpha: Union[int, float, List[float], np.ndarray, torch.Tensor], beta: Union[int, float, List[float], np.ndarray, torch.Tensor]) -> None:
    
        if isinstance(alpha, int) or isinstance(alpha, float):
            alpha = torch.tensor([alpha for _ in range(self.n_out)])
        elif isinstance(alpha, list) or isinstance(alpha, np.ndarray):
            alpha = torch.tensor(alpha)
        if(alpha.ndim != 1):
            raise ValueError(f"Numpy array of 'alpha' values for 'GammaLayer' is expected to be one-dimensional, but is {alpha.ndim}-dimensional.")
        if(alpha.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'alpha' values for 'GammaLayer' must match number of output nodes {self.n_out}, but is {alpha.shape[0]}")
        
        if torch.any(alpha <= 0.0) or not torch.any(torch.isfinite(alpha)):
            raise ValueError(
                f"Values of 'alpha' for 'GammaLayer' must be greater than 0, but was: {alpha}"
            )

        if isinstance(beta, int) or isinstance(beta, float):
            beta = torch.tensor([beta for _ in range(self.n_out)])
        elif isinstance(beta, list) or isinstance(beta, np.ndarray):
            beta = torch.tensor(beta)
        if(beta.ndim != 1):
            raise ValueError(f"Numpy array of 'beta' values for 'GammaLayer' is expected to be one-dimensional, but is {beta.ndim}-dimensional.")
        if(beta.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'beta' values for 'GammaLayer' must match number of output nodes {self.n_out}, but is {beta.shape[0]}")        
        
        if torch.any(beta <= 0.0) or not torch.any(torch.isfinite(beta)):
            raise ValueError(
                f"Value of 'beta' for 'GammaLayer' must be greater than 0, but was: {beta}"
            )

        self.alpha_aux.data = proj_bounded_to_real(alpha, lb=0.0)
        self.beta_aux.data = proj_bounded_to_real(beta, lb=0.0)

    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.alpha, self.beta)
    
    def check_support(self, data: torch.Tensor, node_ids: Optional[List[int]]=None) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Gamma distribution.

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
def marginalize(layer: GammaLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[GammaLayer, Gamma, None]:
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
        return Gamma(scope=marginalized_scopes[0], alpha=layer.alpha[node_id].item(), beta=layer.beta[node_id].item())
    else:
        return GammaLayer(scope=marginalized_scopes, alpha=layer.alpha[marginalized_node_ids].detach(), beta=layer.beta[marginalized_node_ids].detach())


@dispatch(memoize=True)
def toTorch(layer: BaseGammaLayer, dispatch_ctx: Optional[DispatchContext]=None) -> GammaLayer:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return GammaLayer(scope=layer.scopes_out, alpha=layer.alpha, beta=layer.beta)


@dispatch(memoize=True)
def toBase(torch_layer: GammaLayer, dispatch_ctx: Optional[DispatchContext]=None) -> BaseGammaLayer:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseGammaLayer(scope=torch_layer.scopes_out, alpha=torch_layer.alpha.detach().numpy(), beta=torch_layer.beta.detach().numpy())