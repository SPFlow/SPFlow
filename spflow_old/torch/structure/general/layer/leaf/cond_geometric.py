"""Contains conditional Geometric leaf layer for SPFlow in the ``torch`` backend.
"""
from functools import reduce
from typing import Callable, List, Optional, Union
from collections.abc import Iterable

import numpy as np
import torch
import tensorly as tl
import torch.distributions as D

from spflow.base.structure.general.layer.leaf.cond_geometric import (
    CondGeometricLayer as BaseCondGeometricLayer,
)
from spflow.structure.general.layer.leaf import CondGeometricLayer as GeneralCondGeometricLayer
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.node.leaf.cond_geometric import (
    CondGeometric,
)
from spflow.modules.module import Module


class CondGeometricLayer(Module):
    r"""Layer of multiple conditional (univariate) Geometric distribution leaf nodes in the ``torch`` backend.

    Represents multiple conditional univariate Geometric distributions with independent scopes, each with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) =  p(1-p)^{k-1}

    where
        - :math:`k` is the number of trials
        - :math:`p` is the success probability of each trial

    Attributes:
        cond_f:
            Optional callable or list of callables to retrieve parameters for the leaf nodes.
            If a single callable, its output should be a dictionary containing ``p`` as a key, and the value should be
            a floating point, a list of floats or a one-dimensional NumPy array or PyTorch tensor, containing the success probabilities in the range :math:`(0,1]`.
            If it is a single floating point value, the same value is reused for all leaf nodes.
            If a list of callables, each one should return a dictionary containing ``p`` as a key, and the value should
            be a floating point value in the range :math:`(0,1]`.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(
        self,
        scope: Union[Scope, list[Scope]],
        cond_f: Optional[Union[Callable, list[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``CondGeometricLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            cond_f:
                Optional callable or list of callables to retrieve parameters for the leaf nodes.
                If a single callable, its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point, a list of floats or a one-dimensional NumPy array or PyTorch tensor, containing the success probabilities in the range :math:`(0,1]`.
                If it is a single floating point value, the same value is reused for all leaf nodes.
                If a list of callables, each one should return a dictionary containing ``p`` as a key, and the value should
                be a floating point value in the range :math:`(0,1]`.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'CondGeometricLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'CondGeometricLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")
            if len(s.evidence) == 0:
                raise ValueError(f"Evidence scope for 'CondGeometricLayer' should not be empty.")

        super().__init__(children=[], **kwargs)

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.join(s2), self.scopes_out)

        self.set_cond_f(cond_f)
        self.backend = "pytorch"

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``CondGeometricLayer`` can represent one or more univariate nodes with ``MetaType.Discrete`` or ``GeometricType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not CondGeometric.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "CondGeometricLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``CondGeometricLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'CondGeometricLayer' cannot be instantiated from the following signatures: {signatures}."
            )

        scopes = []

        for feature_ctx in signatures:
            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if (
                domain == MetaType.Discrete
                or domain == FeatureTypes.Geometric
                or isinstance(domain, FeatureTypes.Geometric)
            ):
                pass
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'CondGeometricLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return CondGeometricLayer(scopes)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    def set_cond_f(self, cond_f: Optional[Union[list[Callable], Callable]] = None) -> None:
        r"""Sets the ``cond_f`` property.

        Args:
            cond_f:
                Optional callable or list of callables to retrieve parameters for the leaf nodes.
                If a single callable, its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point, a list of floats or a one-dimensional NumPy array or PyTorch tensor, containing the success probabilities in the range :math:`(0,1]`.
                If it is a single floating point value, the same value is reused for all leaf nodes.
                If a list of callables, each one should return a dictionary containing ``p`` as a key, and the value should
                be a floating point value in the range :math:`(0,1]`.

        Raises:
            ValueError: If list of callables does not match number of nodes represented by the layer.
        """
        if isinstance(cond_f, list) and len(cond_f) != self.n_out:
            raise ValueError(
                "'CondGeometricLayer' received list of 'cond_f' functions, but length does not not match number of conditional nodes."
            )

        self.cond_f = cond_f

    def dist(self, p: torch.Tensor, node_ids: Optional[list[int]] = None) -> D.Distribution:
        r"""Returns the PyTorch distributions represented by the leaf layer.

        Args:
            p:
                One-dimensional PyTorch tensor representing the success probabilities of all distributions in the range :math:`(0,1]` (not just the ones specified by ``node_ids``).
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            ``torch.distributions.Geometric`` instance.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Geometric(probs=p[node_ids])

    def retrieve_params(self, data: np.ndarray, dispatch_ctx: DispatchContext) -> torch.Tensor:
        r"""Retrieves the conditional parameters of the leaf layer.

        First, checks if conditional parameter (``p``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function or list of functions (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameters.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameter.

        Args:
            data:
                Two-dimensional NumPy array containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            One-dimensional PyTorch tensor representing the success probabilities.

        Raises:
            ValueError: No way to retrieve conditional parameters or invalid conditional parameters.
        """
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
            raise ValueError(
                "'CondGeometricLayer' requires either 'p' or 'cond_f' to retrieve 'p' to be specified."
            )

        # if 'p' was not already specified, retrieve it
        if p is None:
            # there is a different function for each conditional node
            if isinstance(cond_f, list):
                p = torch.tensor([f(data)["p"] for f in cond_f], dtype=self.dtype, device=self.device)
            else:
                p = cond_f(data)["p"]

        if isinstance(p, int) or isinstance(p, float):
            p = torch.tensor([p for _ in range(self.n_out)], dtype=self.dtype, device=self.device)
        elif isinstance(p, list) or isinstance(p, np.ndarray):
            p = torch.tensor(p, dtype=self.dtype, device=self.device)
        if p.ndim != 1:
            raise ValueError(
                f"Numpy array of 'p' values for 'CondGeometricLayer' is expected to be one-dimensional, but is {p.ndim}-dimensional."
            )
        if p.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'p' values for 'CondGeometricLayer' must match number of output nodes {self.n_out}, but is {p.shape[0]}"
            )

        if torch.any(p <= 0) or not torch.any(torch.isfinite(p)):
            raise ValueError(
                f"Values for 'p' of 'CondGeometricLayer' must to greater of equal to 0, but was: {p}"
            )

        return p

    def check_support(
        self,
        data: torch.Tensor,
        node_ids: Optional[list[int]] = None,
        is_scope_data: bool = False,
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Geometric distributions, which are:

        .. math::

            \text{supp}(\text{Geometric})=\mathbb{N}\setminus\{0\}

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
        # data needs to be offset by -1 due to the different definitions between SciPy and PyTorch
        valid = self.dist(torch.ones(self.n_out), node_ids).support.check(scope_data - 1)  # type: ignore

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        # set nan_entries back to True
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf()

        return valid


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: CondGeometricLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[CondGeometricLayer, CondGeometric, None]:
    """Structural marginalization for ``CondGeometricLayer`` objects in the ``torch`` backend.

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
        return CondGeometric(scope=marginalized_scopes[0])
    else:
        return CondGeometricLayer(scope=marginalized_scopes)


@dispatch(memoize=True)  # type: ignore
def toTorch(
    layer: BaseCondGeometricLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> CondGeometricLayer:
    """Conversion for ``CondGeometricLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondGeometricLayer(scope=layer.scopes_out)


@dispatch(memoize=True)  # type: ignore
def toBase(
    torch_layer: CondGeometricLayer,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> BaseCondGeometricLayer:
    """Conversion for ``CondGeometricLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondGeometricLayer(scope=torch_layer.scopes_out)


@dispatch(memoize=True)  # type: ignore
def updateBackend(leaf_node: CondGeometricLayer, dispatch_ctx: Optional[DispatchContext] = None):
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    data = tl.tensor([])
    cond_f = None
    if leaf_node.cond_f != None:
        params = leaf_node.cond_f(data)

        for key in leaf_node.cond_f(params):
            # Update the value for each key
            params[key] = tl.tensor(params[key])
        cond_f = lambda data: params
    return GeneralCondGeometricLayer(scope=leaf_node.scopes_out, cond_f=cond_f)
