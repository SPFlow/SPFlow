"""Contains Poisson leaf layer for SPFlow in the ``torch`` backend.
"""
from functools import reduce
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter

from spflow.base.structure.general.layer.leaf.poisson import (
    PoissonLayer as BasePoissonLayer,
)
from spflow.tensorly.structure.general.layer.leaf.general_poisson import PoissonLayer as GeneralPoissonLayer
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.node.leaf.poisson import Poisson
from spflow.tensorly.structure.module import Module
from spflow.torch.utils.projections import proj_bounded_to_real, proj_real_to_bounded


class PoissonLayer(Module):
    r"""Layer of multiple (univariate) Poisson distribution leaf node in the ``torch`` backend.

    Represents multiple univariate Poisson distributions with independent scopes, each with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \lambda^k\frac{e^{-\lambda}}{k!}

    where
        - :math:`k` is the number of occurrences
        - :math:`\lambda` is the rate parameter

    Internally :math:`l` are represented as unbounded parameters that are projected onto the bounded range :math:`[0,\infty)` for representing the actual rate probabilities.

    Attributes:
        l_aux:
            Unbounded one-dimensional PyTorch parameter that is projected to yield the actual rate parameter.
        l:
            One-dimensional PyTorch tensor representing the rate parameters (:math:`\lambda`) of the Poisson distributions (projected from ``l_aux``).
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        l: Union[int, float, List[float], np.ndarray, torch.Tensor] = 1.0,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``PoissonLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            l:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor containing the rate parameters (:math:`\lambda`) for each of the independent Poisson distributions (greater than or equal to 0.0).
                If a single value is given, it is broadcast to all nodes.
                Defaults to 1.0.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
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
            if len(s.evidence) != 0:
                raise ValueError(f"Evidence scope for 'PoissonLayer' should be empty, but was {s.evidence}.")

        super().__init__(children=[], **kwargs)

        # register auxiliary torch parameter for rate l of each implicit node
        self.l_aux = Parameter()

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.join(s2), self.scopes_out)

        # parse weights
        self.set_params(l)
        self.backend = "pytorch"

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``PoissonLayer`` can represent one or more univariate nodes with ``MetaType.Discrete`` or ``PoissonType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not Poisson.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "PoissonLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``PoissonLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'PoissonLayer' cannot be instantiated from the following signatures: {signatures}.")

        l = []
        scopes = []

        for feature_ctx in signatures:

            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if domain == MetaType.Discrete:
                l.append(1.0)
            elif domain == FeatureTypes.Poisson:
                # instantiate object
                l.append(domain().l)
            elif isinstance(domain, FeatureTypes.Poisson):
                l.append(domain.l)
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'PoissonLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return PoissonLayer(scopes, l=l)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def l(self) -> torch.Tensor:
        """Returns the rate parameters of the represented distributions."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.l_aux, lb=0.0)  # type: ignore

    def dist(self, node_ids: Optional[List[int]] = None) -> D.Distribution:
        r"""Returns the PyTorch distributions represented by the leaf layer.

        Args:
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            ``torch.distributions.Poisson`` instances.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Poisson(rate=self.l[node_ids])

    def set_params(self, l: Union[int, float, List[float], np.ndarray, torch.Tensor]) -> None:
        """Sets the parameters for the represented distributions in the ``base`` backend.

        Args:
            start:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor containing the start of the intervals (including).
                If a single floating point value is given, it is broadcast to all nodes.
            end:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor containing the end of the intervals (including). Must be larger than 'start'.
                If a single floating point value is given, it is broadcast to all nodes.
            support_outside:
                Boolean, list of booleans or one-dimensional NumPy array or PyTorch tensor containing booleans indicating whether or not values outside of the intervals are part of the support.
                If a single boolean value is given, it is broadcast to all nodes.
                Defaults to True.
        """
        if isinstance(l, int) or isinstance(l, float):
            l = torch.tensor([l for _ in range(self.n_out)], dtype=self.dtype, device=self.device)
        elif isinstance(l, list) or isinstance(l, np.ndarray):
            l = torch.tensor(l, dtype=self.dtype, device=self.device)
        if l.ndim != 1:
            raise ValueError(
                f"Numpy array of 'l' values for 'PoissonLayer' is expected to be one-dimensional, but is {l.ndim}-dimensional."
            )
        if l.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'l' values for 'PoissonLayer' must match number of output nodes {self.n_out}, but is {l.shape[0]}"
            )

        if torch.any(l < 0) or not torch.any(torch.isfinite(l)):
            raise ValueError(f"Values for 'l' of 'PoissonLayer' must to greater of equal to 0, but was: {l}")

        self.l_aux.data = proj_bounded_to_real(l, lb=0.0)

    def get_trainable_params(self) -> List[torch.Tensor]:
        """Returns the parameters of the represented distribution.

        Returns:
            One-dimensional PyTorch tensor representing the rate parameters.
        """
        return [self.l_aux]

    def get_params(self) -> List[torch.Tensor]:
        """Returns the parameters of the represented distribution.

        Returns:
            One-dimensional PyTorch tensor representing the rate parameters.
        """
        return [self.l.cpu().detach().numpy()]

    def check_support(
        self,
        data: torch.Tensor,
        node_ids: Optional[List[int]] = None,
        is_scope_data: bool = False,
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Poisson distributions, which are:

        .. math::

            \text{supp}(\text{Poisson})=\mathbb{N}\cup\{0\}

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.
                Assumes that relevant data is located in the columns corresponding to the scope indices.
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
        valid = self.dist(node_ids).support.check(scope_data)  # type: ignore

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        # set nan_entries back to True
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf()

        return valid

    def to_dtype(self, dtype):
        self.dtype = dtype
        self.l_aux.data = self.l_aux.data.type(dtype)

    def to_device(self, device):
        self.device = device
        self.l_aux.data = self.l_aux.data.to(device)


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: PoissonLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[PoissonLayer, Poisson, None]:
    """Structural marginalization for ``PoissonLayer`` objects in the ``torch`` backend.

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
        node_id = marginalized_node_ids.pop()
        return Poisson(scope=marginalized_scopes[0], l=layer.l[node_id].item())
    else:
        return PoissonLayer(scope=marginalized_scopes, l=layer.l[marginalized_node_ids].detach())


@dispatch(memoize=True)  # type: ignore
def toTorch(layer: BasePoissonLayer, dispatch_ctx: Optional[DispatchContext] = None) -> PoissonLayer:
    """Conversion for ``PoissonLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return PoissonLayer(scope=layer.scopes_out, l=layer.l)


@dispatch(memoize=True)  # type: ignore
def toBase(layer: PoissonLayer, dispatch_ctx: Optional[DispatchContext] = None) -> BasePoissonLayer:
    """Conversion for ``PoissonLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BasePoissonLayer(scope=layer.scopes_out, l=layer.l.detach().numpy())

@dispatch(memoize=True)  # type: ignore
def updateBackend(leaf_node: PoissonLayer, dispatch_ctx: Optional[DispatchContext] = None):
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return GeneralPoissonLayer(scope=leaf_node.scopes_out, l=leaf_node.l.detach().numpy())