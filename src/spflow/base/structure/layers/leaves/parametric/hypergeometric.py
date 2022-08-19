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
from spflow.base.structure.nodes.leaves.parametric.hypergeometric import Hypergeometric


class HypergeometricLayer(Module):
    """Layer representing multiple (univariate) hypergeometric leaf nodes.

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
        
        super(HypergeometricLayer, self).__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [Hypergeometric(s, 1, 1, 1) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(N, M, n)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module."""
        return self._n_out

    @property
    def N(self) -> np.ndarray:
        return np.array([node.N for node in self.nodes])
    
    @property
    def M(self) -> np.ndarray:
        return np.array([node.M for node in self.nodes])

    @property
    def n(self) -> np.ndarray:
        return np.array([node.n for node in self.nodes])

    def set_params(self, N: Union[int, List[int], np.ndarray], M: Union[int, List[int], np.ndarray], n: Union[int, List[int], np.ndarray]) -> None:

        if isinstance(N, int):
            N = np.array([N for _ in range(self.n_out)])
        if isinstance(N, list):
            N = np.array(N)
        if(N.ndim != 1):
            raise ValueError(f"Numpy array of 'N' values for 'HypergeometricLayer' is expected to be one-dimensional, but is {N.ndim}-dimensional.")
        if(N.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'N' values for 'HypergeometricLayer' must match number of output nodes {self.n_out}, but is {N.shape[0]}")

        if isinstance(M, int):
            M = np.array([M for _ in range(self.n_out)])
        if isinstance(M, list):
            M = np.array(M)
        if(M.ndim != 1):
            raise ValueError(f"Numpy array of 'M' values for 'HypergeometricLayer' is expected to be one-dimensional, but is {M.ndim}-dimensional.")
        if(M.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'M' values for 'HypergeometricLayer' must match number of output nodes {self.n_out}, but is {M.shape[0]}")

        if isinstance(n, int):
            n = np.array([n for _ in range(self.n_out)])
        if isinstance(n, list):
            n = np.array(n)
        if(n.ndim != 1):
            raise ValueError(f"Numpy array of 'n' values for 'HypergeometricLayer' is expected to be one-dimensional, but is {n.ndim}-dimensional.")
        if(n.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'n' values for 'HypergeometricLayer' must match number of output nodes {self.n_out}, but is {n.shape[0]}")

        node_scopes = np.array([s.query[0] for s in self.scopes_out])

        for node_scope in np.unique(node_scopes):
            # at least one such element exists
            N_values = N[node_scopes == node_scope]
            if not np.all(N_values == N_values[0]):
                raise ValueError("All values of 'N' for 'HypergeometricLayer' over the same scope must be identical.")
            # at least one such element exists
            M_values = M[node_scopes == node_scope]
            if not np.all(M_values == M_values[0]):
                raise ValueError("All values of 'M' for 'HypergeometricLayer' over the same scope must be identical.")
            # at least one such element exists
            n_values = n[node_scopes == node_scope]
            if not np.all(n_values == n_values[0]):
                raise ValueError("All values of 'n' for 'HypergeometricLayer' over the same scope must be identical.")

        for node_N, node_M, node_n, node in zip(N, M, n, self.nodes):
            node.set_params(node_N, node_M, node_n)
    
    def get_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.N, self.M, self.n
    
    def check_support(self, scope_data: np.ndarray, node_ids: List[int]) -> np.ndarray:
        "TODO"
        valid = np.ones(scope_data.shape, dtype=bool)

        # check for infinite values
        valid &= ~np.isinf(scope_data)

        # check if all values are valid integers
        # TODO: runtime warning due to nan values
        valid &= np.remainder(scope_data, 1) == 0

        # check if values are in valid range
        valid &= (scope_data >= max(0, self.n + self.M - self.N)) & (  # type: ignore
            scope_data <= min(self.n, self.M)  # type: ignore
        )

        return valid


@dispatch(memoize=True)
def marginalize(layer: HypergeometricLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[HypergeometricLayer, Hypergeometric, None]:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # marginalize nodes
    marg_scopes = []
    marg_params = []

    for node in layer.nodes:
        marg_node = marginalize(node, marg_rvs, prune=prune)

        if marg_node is not None:
            marg_scopes.append(marg_node.scope)
            marg_params.append(marg_node.get_params())

    if len(marg_scopes) == 0:
        return None
    elif len(marg_scopes) == 1 and prune:
        new_node = Hypergeometric(marg_scopes[0], *marg_params[0])
        return new_node
    else:
        new_layer = HypergeometricLayer(marg_scopes, *[np.array(p) for p in zip(*marg_params)])
        return new_layer