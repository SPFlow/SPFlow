"""
Created on October 21, 2022

@authors: Philipp Deibert
"""
from typing import List, Union, Optional, Iterable, Tuple, Callable
from functools import reduce
import numpy as np
import torch
import torch.distributions as D

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.scope.scope import Scope
from spflow.torch.structure.module import Module
from spflow.torch.structure.nodes.leaves.parametric.cond_negative_binomial import CondNegativeBinomial
from spflow.base.structure.layers.leaves.parametric.cond_negative_binomial import CondNegativeBinomialLayer as BaseCondNegativeBinomialLayer


class CondNegativeBinomialLayer(Module):
    """Layer representing multiple conditional (univariate) negative binomial leaf nodes in the Torch backend.

    Args:
        scope: TODO
        n: TODO
        cond_f: TODO
        n_nodes: number of output nodes.
    """
    def __init__(self, scope: Union[Scope, List[Scope]], n: Union[int, List[int], np.ndarray, torch.Tensor], cond_f: Optional[Union[Callable,List[Callable]]]=None, n_nodes: int=1, **kwargs) -> None:
        """TODO"""
        
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(f"Number of nodes for 'CondNegativeBinomialLayer' must be greater or equal to 1, but was {n_nodes}")

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'CondNegativeBinomialLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")

        super(CondNegativeBinomialLayer, self).__init__(children=[], **kwargs)
    
        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("n", torch.empty(size=[]))

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.union(s2), self.scopes_out)

        # parse weights
        self.set_params(n)

        self.set_cond_f(cond_f)
    
    @property
    def n_out(self) -> int:
        return self._n_out

    def set_cond_f(self, cond_f: Optional[Union[List[Callable], Callable]]=None) -> None:

        if isinstance(cond_f, List) and len(cond_f) != self.n_out:
            raise ValueError("'CondNegativeBinomialLayer' received list of 'cond_f' functions, but length does not not match number of conditional nodes.")

        self.cond_f = cond_f

    def dist(self, p: torch.Tensor, node_ids: Optional[List[int]]=None) -> D.Distribution:

        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.NegativeBinomial(total_count=self.n[node_ids], probs=torch.ones(len(node_ids))-p[node_ids])

    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> torch.Tensor:

        p, cond_f = None, None

        # check dispatch cache for required conditional parameter 'p'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if a value for 'p' is specified (highest priority)
            if "p" in args:
                p = args["p"]
            # check if alternative function to provide 'p' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'p' specified (lowest priority)
            cond_f = self.cond_f
        
        # if neither 'p' nor 'cond_f' is specified (via node or arguments)
        if p is None and cond_f is None:
            raise ValueError("'CondNegativeBinomialLayer' requires either 'p' or 'cond_f' to retrieve 'p' to be specified.")

        # if 'p' was not already specified, retrieve it
        if p is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, List):
                p = torch.tensor([f(data)['p'] for f in cond_f])
            else:
                p = cond_f(data)['p']

        if isinstance(p, float) or isinstance(p, int):
            p = torch.tensor([p for _ in range(self.n_out)])
        elif isinstance(p, list) or isinstance(p, np.ndarray):
            p = torch.tensor(p)
        if p.ndim != 1:
            raise ValueError(f"Numpy array of 'p' values for 'CondNegativeBinomialLayer' is expected to be one-dimensional, but is {p.ndim}-dimensional.")
        if p.shape[0] == 1:
            p = torch.hstack([p for _ in range(self.n_out)])
        if(p.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'p' values for 'CondNegativeBinomialLayer' must match number of output nodes {self.n_out}, but is {p.shape[0]}")
        if torch.any(p < 0.0) or torch.any(p > 1.0) or not all(torch.isfinite(p)):
            raise ValueError(
                f"Values of 'p' for 'CondNegativeBinomialLayer' distribution must to be between 0.0 and 1.0, but are: {p}"
            )

        return p

    def set_params(self, n: Union[int, List[int], np.ndarray, torch.Tensor]) -> None:
    
        if isinstance(n, int) or isinstance(n, float):
            n = torch.tensor([n for _ in range(self.n_out)])
        elif isinstance(n, list) or isinstance(n, np.ndarray):
            n = torch.tensor(n)
        if(n.ndim != 1):
            raise ValueError(f"Numpy array of 'n' values for 'NegativeBinomialLayer' is expected to be one-dimensional, but is {n.ndim}-dimensional.")
        if(n.shape[0] != self.n_out):
            raise ValueError(f"Length of numpy array of 'n' values for 'NegativeBinomialLayer' must match number of output nodes {self.n_out}, but is {n.shape[0]}")
        
        if torch.any(n < 0) or not torch.any(torch.isfinite(n)):
            raise ValueError(
                f"Values for 'n' of 'NegativeBinomialLayer' must to greater of equal to 0, but was: {n}"
            )

        if not torch.all((torch.remainder(n, 1.0) == torch.tensor(0.0))):
            raise ValueError(
                f"Values for 'n' of 'NegativeBinomialLayer' must be (equal to) an integer value, but was: {n}"
            )
        
        node_scopes = torch.tensor([s.query[0] for s in self.scopes_out])

        for node_scope in torch.unique(node_scopes):
            # at least one such element exists
            n_values = n[node_scopes == node_scope]
            if not torch.all(n_values == n_values[0]):
                raise ValueError("All values of 'n' for 'NegativeBinomialLayer' over the same scope must be identical.")

        self.n.data = n

    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.n,)

    def check_support(self, data: torch.Tensor, node_ids: Optional[List[int]]=None) -> torch.Tensor:
        r"""Checks if instances are part of the support of the NegativeBinomial distribution.

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
        valid = self.dist(torch.ones(self.n_out), node_ids).support.check(scope_data)  # type: ignore

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        # set nan_entries back to True
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf()

        return valid


@dispatch(memoize=True)
def marginalize(layer: CondNegativeBinomialLayer, marg_rvs: Iterable[int], prune: bool=True, dispatch_ctx: Optional[DispatchContext]=None) -> Union[CondNegativeBinomialLayer, CondNegativeBinomial, None]:
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
        return CondNegativeBinomial(scope=marginalized_scopes[0], n=layer.n[node_id].item())
    else:
        return CondNegativeBinomialLayer(scope=marginalized_scopes, n=layer.n[marginalized_node_ids].detach())


@dispatch(memoize=True)
def toTorch(layer: BaseCondNegativeBinomialLayer, dispatch_ctx: Optional[DispatchContext]=None) -> CondNegativeBinomialLayer:
    return CondNegativeBinomialLayer(scope=layer.scopes_out, n=layer.n)


@dispatch(memoize=True)
def toBase(torch_layer: CondNegativeBinomialLayer, dispatch_ctx: Optional[DispatchContext]=None) -> BaseCondNegativeBinomialLayer:
    return BaseCondNegativeBinomialLayer(scope=torch_layer.scopes_out, n=torch_layer.n.numpy())