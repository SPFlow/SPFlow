# -*- coding: utf-8 -*-
"""Contains conditional Exponential leaf layer for SPFlow in the ``torch`` backend.
"""
from typing import List, Union, Optional, Iterable, Tuple, Callable
from functools import reduce
import numpy as np
import torch
import torch.distributions as D

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.contexts.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.scope.scope import Scope
from spflow.torch.structure.module import Module
from spflow.torch.structure.nodes.leaves.parametric.cond_exponential import (
    CondExponential,
)
from spflow.base.structure.layers.leaves.parametric.cond_exponential import (
    CondExponentialLayer as BaseCondExponentialLayer,
)


class CondExponentialLayer(Module):
    r"""Layer of multiple conditional (univariate) Exponential distribution leaf nodes in the ``torch`` backend.

    Represents multiple conditional univariate Exponential distributions with independent scopes, each with the following probability distribution function (PDF):

    .. math::
        
        \text{PDF}(x) = \begin{cases} \lambda e^{-\lambda x} & \text{if } x > 0\\
                                      0                      & \text{if } x <= 0\end{cases}
    
    where
        - :math:`x` is the input observation
        - :math:`\lambda` is the rate parameter
    
    Attributes:
        cond_f:
            Optional callable or list of callables to retrieve parameters for the leaf nodes.
            If a single callable, its output should be a dictionary containing ``l`` as a key, and the value should be
            a floating point, a list of floats or a one-dimensional NumPy array or PyTorch tensor, containing the rate parameters (:math:`\lambda`), greater than 0.0.
            If it is a single floating point value, the same value is reused for all leaf nodes.
            If a list of callables, each one should return a dictionary containing ``l`` as a key, and the value should
            be a floating point value greater than 0.0.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        cond_f: Optional[Union[Callable, List[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``CondExponentialLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            cond_f:
                Optional callable or list of callables to retrieve parameters for the leaf nodes.
                If a single callable, its output should be a dictionary containing ``l`` as a key, and the value should be
                a floating point, a list of floats or a one-dimensional NumPy array or PyTorch tensor, containing the rate parameters (:math:`\lambda`), greater than 0.0.
                If it is a single floating point value, the same value is reused for all leaf nodes.
                If a list of callables, each one should return a dictionary containing ``l`` as a key, and the value should
                be a floating point value greater than 0.0.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'CondExponentialLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError(
                    "List of scopes for 'CondExponentialLayer' was empty."
                )

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")

        super(CondExponentialLayer, self).__init__(children=[], **kwargs)

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(
            lambda s1, s2: s1.union(s2), self.scopes_out
        )

        self.set_cond_f(cond_f)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    def set_cond_f(
        self, cond_f: Optional[Union[List[Callable], Callable]] = None
    ) -> None:
        r"""Sets the ``cond_f`` property.

        Args:
            cond_f:
                Optional callable or list of callables to retrieve parameters for the leaf nodes.
                If a single callable, its output should be a dictionary containing ``l`` as a key, and the value should be
                a floating point, a list of floats or a one-dimensional NumPy array or PyTorch tensor, containing the rate parameters (:math:`\lambda`), greater than 0.0.
                If it is a single floating point value, the same value is reused for all leaf nodes.
                If a list of callables, each one should return a dictionary containing ``l`` as a key, and the value should
                be a floating point value greater than 0.0.

        Raises:
            ValueError: If list of callables does not match number of nodes represented by the layer.
        """
        if isinstance(cond_f, List) and len(cond_f) != self.n_out:
            raise ValueError(
                "'CondExponentialLayer' received list of 'cond_f' functions, but length does not not match number of conditional nodes."
            )

        self.cond_f = cond_f

    def dist(
        self, l: torch.Tensor, node_ids: Optional[List[int]] = None
    ) -> D.Distribution:
        r"""Returns the PyTorch distributions represented by the leaf layer.

        Args:
            l:
                One-dimensional PyTorch tensor representing the rate parameters of all distributions (not just the ones specified by ``node_ids``).
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            ``torch.distributions.Exponential`` instance.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Exponential(rate=l[node_ids])

    def retrieve_params(
        self, data: np.ndarray, dispatch_ctx: DispatchContext
    ) -> torch.Tensor:
        r"""Retrieves the conditional parameters of the leaf layer.

        First, checks if conditional parameter (``l``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function or list of functions (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameters.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameter.

        Args:
            data:
                Two-dimensional NumPy array containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            One-dimensional PyTorch tensor representing the rate parameters.

        Raises:
            ValueError: No way to retrieve conditional parameters or invalid conditional parameters.
        """
        l, cond_f = None, None

        # check dispatch cache for required conditional parameter 'l'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if a value for 'l' is specified (highest priority)
            if "l" in args:
                l = args["l"]
            # check if alternative function to provide 'l' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'l' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'l' nor 'cond_f' is specified (via node or arguments)
        if l is None and cond_f is None:
            raise ValueError(
                "'CondExponentialLayer' requires either 'l' or 'cond_f' to retrieve 'l' to be specified."
            )

        # if 'l' was not already specified, retrieve it
        if l is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, List):
                l = torch.tensor([f(data)["l"] for f in cond_f])
            else:
                l = cond_f(data)["l"]

        if isinstance(l, int) or isinstance(l, float):
            l = torch.tensor([l for _ in range(self.n_out)])
        elif isinstance(l, list) or isinstance(l, np.ndarray):
            l = torch.tensor(l)
        if l.ndim != 1:
            raise ValueError(
                f"Numpy array of 'l' values for 'CondExponentialLayer' is expected to be one-dimensional, but is {l.ndim}-dimensional."
            )
        if l.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'l' values for 'CondExponentialLayer' must match number of output nodes {self.n_out}, but is {l.shape[0]}"
            )

        if torch.any(l <= 0) or not torch.any(torch.isfinite(l)):
            raise ValueError(
                f"Values for 'l' of 'CondExponentialLayer' must to greater of equal to 0, but was: {l}"
            )

        return l

    def check_support(
        self,
        data: torch.Tensor,
        node_ids: Optional[List[int]] = None,
        is_scope_data: bool = False,
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Exponential distributions, which are:

        .. math::

            \text{supp}(\text{Exponential})=[0,+\infty)

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leafs' scope in the correct order (True) or if it needs to be extracted from the full data set.
                Note, that this should already only contain only the data according (and in order of) ``node_ids``.
                Defaults to False.

        Returns:
            Two dimensional PyTorch tensor indicating for each instance and node, whether they are part of the support (True) or not (False).
            Each row corresponds to an input sample.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        if is_scope_data:
            scope_data = data
        else:
            # all query scopes are univariate
            scope_data = data[
                :, [self.scopes_out[node_id].query[0] for node_id in node_ids]
            ]

        # NaN values do not throw an error but are simply flagged as False
        valid = self.dist(torch.ones(self.n_out), node_ids).support.check(scope_data)  # type: ignore

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        # set nan_entries back to True
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf()

        return valid


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: CondExponentialLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[CondExponentialLayer, CondExponential, None]:
    """Structural marginalization for ``CondExponentialLayer`` objects in the ``torch`` backend.

    Structurally marginalizes the specified layer module.
    If the layer's scope contains non of the random variables to marginalize, then the layer is returned unaltered.
    If the layer's scope is fully marginalized over, then None is returned.

    Args:
        layer:
            Layer module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect here. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Unaltered leaf layer or None if it is completely marginalized.
    """
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
        return CondExponential(scope=marginalized_scopes[0])
    else:
        return CondExponentialLayer(scope=marginalized_scopes)


@dispatch(memoize=True)  # type: ignore
def toTorch(
    layer: BaseCondExponentialLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> CondExponentialLayer:
    """Conversion for ``CondExponentialLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondExponentialLayer(scope=layer.scopes_out)


@dispatch(memoize=True)  # type: ignore
def toBase(
    torch_layer: CondExponentialLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> BaseCondExponentialLayer:
    """Conversion for ``CondExponentialLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondExponentialLayer(scope=torch_layer.scopes_out)
