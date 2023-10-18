"""Contains conditional Categorical leaf layer for SPFlow in the ``torch`` backend.
"""
from functools import reduce
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as D

from spflow.base.structure.general.layers.leaves.parametric.cond_categorical import (
    CondCategoricalLayer as BaseCondCategoricalLayer,
)
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaves.parametric.cond_categorical import (
    CondCategorical,
)
from spflow.torch.structure.module import Module


class CondCategoricalLayer(Module):
    r"""Layer of multiple conditional (univariate) Categorical distribution leaf nodes in the ``torch`` backend.

    Represents multiple conditional univariate Categorical distributions with independent scopes, each with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k)= p_k
        
    where
        - :math:`k` is an integer associated with a category
        - :math:`p_k` is the success probability of the associated category k in :math:`[0,1]`

    Attributes:
        cond_f:
            Optional callable or list of callables to retrieve parameters for the leaf nodes.
            If a single callable, its output should be a dictionary containing ``p`` as a key, and the value should be
            a floating point, a list of floats or o one-dimensional NumPy array or PyTorch tensor, containing the success probabilities between zero and one.
            If it is a single floating point value, the same value is reused for all leaf nodes.
            If a list of callables, each one should return a dictionary containing ``p`` as a key, and the value should
            be a floating point value between zero and one.
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
        r"""Initializes ``CondCategoricalLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            cond_f:
                Optional callable or list of callables to retrieve parameters for the leaf nodes.
                If a single callable, its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point, a list of floats or o one-dimensional NumPy array or PyTorch tensor, containing the success probabilities between zero and one.
                If it is a single floating point value, the same value is reused for all leaf nodes.
                If a list of callables, each one should return a dictionary containing ``p`` as a key, and the value should
                be a floating point value between zero and one.
            n_nodes:
                Integer specifying the number of nodes the layer should represent.

        Raises:
            ValueError: Invalid arguments.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'CondCategoricalLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'CondCategoricalLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")
            if len(s.evidence) == 0:
                raise ValueError(f"Evidence scope for 'CondCategoricalLayer' should not be empty.")

        super().__init__(children=[], **kwargs)

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.join(s2), self.scopes_out)

        self.set_cond_f(cond_f)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``CondCategoricalLayer`` can represent one or more univariate nodes with ``MetaType.Discrete`` or ``CategoricalType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not CondCategorical.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "CondCategoricalLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``CondCategoricalLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'CondCategoricalLayer' cannot be instantiated from the following signatures: {signatures}."
            )

        scopes = []

        for feature_ctx in signatures:

            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if (
                domain == MetaType.Discrete
                or domain == FeatureTypes.Categorical
                or isinstance(domain, FeatureTypes.Categorical)
            ):
                pass
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'CondCategoricalLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return CondCategoricalLayer(scopes)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    def set_cond_f(self, cond_f: Optional[Union[List[Callable], Callable]] = None) -> None:
        r"""Sets the ``cond_f`` property.

        Args:
            cond_f:
                Optional callable or list of callables to retrieve parameters for the leaf nodes.
                If a single callable, its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point, a list of floats or o one-dimensional NumPy array or PyTorch tensor, containing the success probabilities between zero and one.
                If it is a single floating point value, the same value is reused for all leaf nodes.
                If a list of callables, each one should return a dictionary containing ``p`` as a key, and the value should
                be a floating point value between zero and one.

        Raises:
            ValueError: If list of callables does not match number of nodes represented by the layer.
        """
        if isinstance(cond_f, List) and len(cond_f) != self.n_out:
            raise ValueError(
                "'CondCategoricalLayer' received list of 'cond_f' functions, but length does not not match number of conditional nodes."
            )

        self.cond_f = cond_f

    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Retrieves the conditional parameters of the leaf layer.

        First, checks if conditional parameter (``p``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function or list of functions (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameters.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameter.

        Args:
            data:
                Two-dimensional PyTorch tensor containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            One-dimensional PyTorch tensor representing the success probabilities.

        Raises:
            ValueError: No way to retrieve conditional parameters or invalid conditional parameters.
        """
        k, p, cond_f = None, None, None

        # check dispatch cache for required conditional parameters 'k', 'p'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if values for 'k' and 'p' are specified (highest priority)
            if "k" in args:
                k = args["k"]
            if "p" in args:
                p = args["p"]
            # check if alternative function to provide 'k' and 'p' are specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'k', 'p' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'k' nor 'p' nor 'cond_f' is specified (via node or arguments)
        if k is None and p is None and cond_f is None:
            raise ValueError("'CondCategoricalLayer' requires either 'p' or 'cond_f' to retrieve 'p' to be specified.")

        # if 'k' and 'p' were not already specified, retrieve them
        if k is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, List):
                k = torch.tensor([f(data)["k"] for f in cond_f])
            else:
                k = cond_f(data)["k"]    
        if p is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, List):
                p = torch.tensor([f(data)["p"] for f in cond_f])
            else:
                p = cond_f(data)["p"]


        if isinstance(k, int):
            k = torch.tensor([k for _ in range(self.n_out)])
        elif isinstance(k, list):
            k = torch.tensor(k)
        if k.ndim != 1:
            raise ValueError(f"Tensor of 'k' values for 'CategoricalLayer' is expected to be one-dimensional, but is {k.ndim}-dimensional")
        if k.shape[0] == 1:
            k = torch.hstack([k for _ in range(self.n_out)])
        if k.shape[0] != self.n_out:
            raise ValueError(f"Length of numpy array of 'k' values for 'CategoricalLayer' must match number of output nodes {self.n_out}, but is {k.shape[0]}")
        if torch.any(k<1) or not all(torch.isfinite(k)):
            raise ValueError(f"Values of 'k' for CategoricalLayer' distribution must be positive, but are: {k}")


        if isinstance(p, (list, np.ndarray)) and isinstance(p[0], float):
            p = torch.tensor([p for _ in range(self.n_out)])
        elif isinstance(p, (list, np.ndarray)) and isinstance(p[0], (list, np.ndarray)):
            p = torch.tensor(p)
        if p.ndim != 2:
            raise ValueError(
                f"Tensor of 'p' values for 'CategoricalLayer' is expected to be two-dimensional, but is {p.ndim}-dimensional."
            )
        if p.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'p' values for 'CategoricalLayer' must match number of output nodes {self.n_out}, but is {p.shape[0]}"
            )
        if torch.any((p < 0.0).float()) or torch.any((p > 1.0).float()) or not torch.all(torch.isfinite(p)):
            raise ValueError(
                f"Values of 'p' for 'CategoricalLayer' distribution must to be between 0.0 and 1.0, but are: {p}"
            )

        return k, p

    def dist(self, p: torch.Tensor, node_ids: Optional[List[int]] = None) -> D.Distribution:
        r"""Returns the PyTorch distributions represented by the leaf layer.

        Args:
            p:
                Two-dimensional PyTorch tensor representing the selection probabilities of all distributions between zero and one (not just the ones specified by ``node_ids``).
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            ``torch.distributions.Categorical`` instance.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Categorical(probs=p[node_ids])

    def check_support(
        self,
        data: torch.Tensor,
        node_ids: Optional[List[int]] = None,
        is_scope_data: bool = False,
        dispatch_ctx: DispatchContext() = DispatchContext()
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or not instances are part of the supports of the Categorical distributions, which are:

        .. math::

            \text{supp}(\text{Categorical})=\{0, 1, ..., k-1\}

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
            scope_data = data[:, [self.scopes_out[node_id].query[0] for node_id in node_ids]]

        # NaN values do not throw an error but are simply flagged as False
        k, p = self.retrieve_params(torch.tensor([1.]), dispatch_ctx=dispatch_ctx)
        valid = self.dist(p, node_ids).support.check(scope_data)  # type: ignore

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        # set nan_entries back to True
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf()

        return valid


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: CondCategoricalLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[CondCategoricalLayer, CondCategorical, None]:
    """Structural marginalization for ``CondCategoricalLayer`` objects in the ``torch`` backend.

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
        return CondCategorical(scope=marginalized_scopes[0])
    else:
        return CondCategoricalLayer(scope=marginalized_scopes)


@dispatch(memoize=True)  # type: ignore
def toTorch(
    layer: BaseCondCategoricalLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> CondCategoricalLayer:
    """Conversion for ``CondCategoricalLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondCategoricalLayer(scope=layer.scopes_out)


@dispatch(memoize=True)  # type: ignore
def toBase(
    torch_layer: CondCategoricalLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> BaseCondCategoricalLayer:
    """Conversion for ``CondCategoricalLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondCategoricalLayer(scope=torch_layer.scopes_out)
