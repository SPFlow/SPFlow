"""
Created on August 15, 2022

@authors: Philipp Deibert
"""
from typing import List, Union, Optional, Iterable, Tuple
from functools import reduce
import numpy as np
import torch
import torch.distributions as D
from ....nodes.leaves.parametric.projections import proj_bounded_to_real, proj_real_to_bounded

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.scope.scope import Scope
from spflow.torch.structure.module import Module
from spflow.torch.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric
from spflow.base.structure.layers.leaves.parametric.hypergeometric import HypergeometricLayer as BaseHypergeometricLayer


class HypergeometricLayer(Module):
    """Layer representing multiple (univariate) hypergeometric leaf nodes in the Torch backend.

    Args:
        scope: TODO
        N: TODO
        M: TODO
        n: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], N: Union[int, List[int], np.ndarray], M: Union[int, List[int], np.ndarray], n: Union[int, List[int], np.ndarray], n_nodes: int=1, **kwargs) -> None:
        """TODO"""
        
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'HypergeometricLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'HypergeometricLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")

        super(HypergeometricLayer, self).__init__(children=[], **kwargs)
    
        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("N", torch.empty(size=[]))
        self.register_buffer("M", torch.empty(size=[]))
        self.register_buffer("n", torch.empty(size=[]))

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.union(s2), self.scopes_out)

        # parse weights
        self.set_params(N, M, n)
    
    @property
    def n_out(self) -> int:
        return self._n_out
    
    def set_params(self, N: Union[int, List[int], np.ndarray, torch.Tensor], M: Union[int, List[int], np.ndarray, torch.Tensor], n: Union[int, List[int], np.ndarray, torch.Tensor]) -> None:

        if isinstance(N, int) or isinstance(N, float):
            N = torch.tensor([N for _ in range(self.n_out)])
        elif isinstance(N, list) or isinstance(N, np.ndarray):
            N = torch.tensor(N)
        if(N.ndim != 1):
            raise ValueError(f"Torch tensor of 'N' values for 'HypergeometricLayer' is expected to be one-dimensional, but is {N.ndim}-dimensional.")
        if(N.shape[0] != self.n_out):
            raise ValueError(f"Length of torch tensor of 'N' values for 'HypergeometricLayer' must match number of output nodes {self.n_out}, but is {N.shape[0]}")

        if isinstance(M, int) or isinstance(M, float):
            M = torch.tensor([M for _ in range(self.n_out)])
        elif isinstance(n, list) or isinstance(M, np.ndarray):
            M = torch.tensor(M)
        if(M.ndim != 1):
            raise ValueError(f"Torch tensor of 'M' values for 'HypergeometricLayer' is expected to be one-dimensional, but is {M.ndim}-dimensional.")
        if(M.shape[0] != self.n_out):
            raise ValueError(f"Length of torch tensor of 'M' values for 'HypergeometricLayer' must match number of output nodes {self.n_out}, but is {M.shape[0]}")

        if isinstance(n, int) or isinstance(n, float):
            n = torch.tensor([n for _ in range(self.n_out)])
        elif isinstance(n, list) or isinstance(n, np.ndarray):
            n = torch.tensor(n)
        if(n.ndim != 1):
            raise ValueError(f"Torch tensor of 'n' values for 'HypergeometricLayer' is expected to be one-dimensional, but is {n.ndim}-dimensional.")
        if(n.shape[0] != self.n_out):
            raise ValueError(f"Length of torch tensor of 'n' values for 'HypergeometricLayer' must match number of output nodes {self.n_out}, but is {n.shape[0]}")

        if torch.any(N < 0) or not torch.all(torch.isfinite(N)):
            raise ValueError(
                f"Value of 'N' for 'HypergeometricLayer' must be greater of equal to 0, but was: {N}"
            )
        if not torch.all(torch.remainder(N, 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Value of 'N' for 'HypergeometricLayer' must be (equal to) an integer value, but was: {N}"
            )

        if torch.any(M < 0) or torch.any(M > N) or not torch.all(torch.isfinite(M)):
            raise ValueError(
                f"Values of 'M' for 'HypergeometricLayer' must be greater of equal to 0 and less or equal to 'N', but was: {M}"
            )
        if not torch.all(torch.remainder(M, 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Values of 'M' for 'HypergeometricLayer' must be (equal to) an integer value, but was: {M}"
            )

        if torch.any(n < 0) or torch.any(n > N) or not torch.all(torch.isfinite(n)):
            raise ValueError(
                f"Value of 'n' for 'HypergeometricLayer' must be greater of equal to 0 and less or equal to 'N', but was: {n}"
            )
        if not torch.all(torch.remainder(n, 1.0) == torch.tensor(0.0)):
            raise ValueError(
                f"Value of 'n' for 'HypergeometricLayer' must be (equal to) an integer value, but was: {n}"
            )

        node_scopes = torch.tensor([s.query[0] for s in self.scopes_out])

        for node_scope in torch.unique(node_scopes):
            # at least one such element exists
            N_values = N[node_scopes == node_scope]
            if not torch.all(N_values == N_values[0]):
                raise ValueError("All values of 'N' for 'HypergeometricLayer' over the same scope must be identical.")
            # at least one such element exists
            M_values = M[node_scopes == node_scope]
            if not torch.all(M_values == M_values[0]):
                raise ValueError("All values of 'M' for 'HypergeometricLayer' over the same scope must be identical.")
            # at least one such element exists
            n_values = n[node_scopes == node_scope]
            if not torch.all(n_values == n_values[0]):
                raise ValueError("All values of 'n' for 'HypergeometricLayer' over the same scope must be identical.")

        self.N.data = N
        self.M.data = M
        self.n.data = n
    
    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.N, self.M, self.n
    
    def check_support(self, data: torch.Tensor, node_ids: Optional[List[int]]=None) -> torch.Tensor:
        r"""Checks if instances are part of the support of the Hypergeometric distribution.

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

        valid = torch.ones(scope_data.shape, dtype=torch.bool)

        # check for infinite values
        valid &= ~torch.isinf(scope_data)

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        # check if all values are valid integers
        valid[~nan_mask] &= torch.remainder(scope_data[~nan_mask], 1) == 0

        node_ids_tensor = torch.tensor(node_ids)
        N_nodes = self.N[node_ids_tensor]
        M_nodes = self.M[node_ids_tensor]
        n_nodes = self.n[node_ids_tensor]

        # check if values are in valid range
        valid[~nan_mask & valid] &= ((scope_data >= torch.max(torch.vstack([torch.zeros(scope_data.shape[1]), n_nodes + M_nodes - N_nodes]), dim=0)[0].unsqueeze(0)) & (  # type: ignore
            scope_data <= torch.min(torch.vstack([n_nodes, M_nodes]), dim=0)[0].unsqueeze(0)  # type: ignore
        ))[~nan_mask & valid]

        return valid
    
    def log_prob(self, k: torch.Tensor, node_ids: Optional[List[int]]=None) -> torch.Tensor:

        if node_ids is None:
            node_ids = list(range(self.n_out))
        
        node_ids_tensor = torch.tensor(node_ids)

        N = self.N[node_ids_tensor]
        M = self.M[node_ids_tensor]
        n = self.n[node_ids_tensor]

        N_minus_M = N - M  # type: ignore
        n_minus_k = n - k  # type: ignore

        # ----- (M over m) * (N-M over n-k) / (N over n) -----

        # log_M_over_k = torch.lgamma(self.M+1) - torch.lgamma(self.M-k+1) - torch.lgamma(k+1)
        # log_NM_over_nk = torch.lgamma(N_minus_M+1) - torch.lgamma(N_minus_M-n_minus_k+1) - torch.lgamma(n_minus_k+1)
        # log_N_over_n = torch.lgamma(self.N+1) - torch.lgamma(self.N-self.n+1) - torch.lgamma(self.n+1)
        # result = log_M_over_k + log_NM_over_nk - log_N_over_n

        # ---- alternatively (more precise according to SciPy) -----
        # betaln(good+1, 1) + betaln(bad+1,1) + betaln(total-draws+1, draws+1) - betaln(k+1, good-k+1) - betaln(draws-k+1, bad-draws+k+1) - betaln(total+1, 1)

        lgamma_1 = torch.lgamma(torch.ones(len(node_ids)))
        lgamma_M_p_2 = torch.lgamma(M + 2)
        lgamma_N_p_2 = torch.lgamma(N + 2)
        lgamma_N_m_M_p_2 = torch.lgamma(N_minus_M + 2)

        result = (
            torch.lgamma(M + 1)  # type: ignore
            + lgamma_1
            - lgamma_M_p_2  # type: ignore
            + torch.lgamma(N_minus_M + 1)  # type: ignore
            + lgamma_1
            - lgamma_N_m_M_p_2  # type: ignore
            + torch.lgamma(N - n + 1)  # type: ignore
            + torch.lgamma(n + 1)  # type: ignore
            - lgamma_N_p_2  # type: ignore
            - torch.lgamma(k + 1) # .float()
            - torch.lgamma(M - k + 1)
            + lgamma_M_p_2  # type: ignore
            - torch.lgamma(n_minus_k + 1)
            - torch.lgamma(N_minus_M - n + k + 1)
            + lgamma_N_m_M_p_2  # type: ignore
            - torch.lgamma(N + 1)  # type: ignore
            - lgamma_1
            + lgamma_N_p_2  # type: ignore
        )

        return result


@dispatch(memoize=True)
def marginalize(layer: HypergeometricLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[HypergeometricLayer, Hypergeometric, None]:
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
        return Hypergeometric(scope=marginalized_scopes[0], N=layer.N[node_id].item(), M=layer.M[node_id].item(), n=layer.n[node_id].item())
    else:
        return HypergeometricLayer(scope=marginalized_scopes, N=layer.N[marginalized_node_ids], M=layer.M[marginalized_node_ids], n=layer.n[marginalized_node_ids])


@dispatch(memoize=True)
def toTorch(layer: BaseHypergeometricLayer, dispatch_ctx: Optional[DispatchContext]=None) -> HypergeometricLayer:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return HypergeometricLayer(scope=layer.scopes_out, N=layer.N, M=layer.M, n=layer.n)


@dispatch(memoize=True)
def toBase(torch_layer: HypergeometricLayer, dispatch_ctx: Optional[DispatchContext]=None) -> BaseHypergeometricLayer:
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseHypergeometricLayer(scope=torch_layer.scopes_out, N=torch_layer.N.numpy(), M=torch_layer.M.numpy(), n=torch_layer.n.numpy())