"""Contains Bernoulli leaf layer for SPFlow in the ``torch`` backend.
"""
from functools import reduce
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter

from spflow.base.structure.general.layers.leaves.parametric.bernoulli import (
    BernoulliLayer as BaseBernoulliLayer,
)
from spflow.tensorly.structure.spn.layers.leaves.parametric import BernoulliLayer as GeneralBernoulli
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaves.parametric.bernoulli import Bernoulli
from spflow.tensorly.structure.module import Module
from spflow.torch.utils.projections import proj_bounded_to_real, proj_real_to_bounded


class BernoulliLayer(Module):
    r"""Layer of multiple (univariate) Bernoulli distribution leaf nodes in the ``torch`` backend.

    Represents multiple univariate Bernoulli distributions with independent scopes, each with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k)=\begin{cases} p   & \text{if } k=1\\
                                    1-p & \text{if } k=0\end{cases}
        
    where
        - :math:`p` is the success probability in :math:`[0,1]`
        - :math:`k` is the outcome of the trial (0 or 1)

    Internally :math:`p` are represented as unbounded parameters that are projected onto the bounded range :math:`[0,1]` for representing the actual success probabilities.

    Attributes:
        p_aux:
            Unbounded one-dimensional PyTorch parameter that is projected to yield the actual success probabilities.
        p:
            One-dimensional PyTorch tensor representing the success probabilities of the Bernoulli distributions (projected from ``p_aux``).
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        p: Union[int, float, List[float], np.ndarray, torch.Tensor] = 0.5,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``BernoulliLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            p:
                Floating point, list of floats, one-dimensional NumPy array or PyTorch tensor representing the success probabilities of the Bernoulli distributions between zero and one.
                If a single value is given it is broadcast to all nodes.
                Defaults to 0.5.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'BernoulliLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'BernoulliLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")
            if len(s.evidence) != 0:
                raise ValueError(f"Evidence scope for 'BernoulliLayer' should be empty, but was {s.evidence}.")

        super().__init__(children=[], **kwargs)

        # register auxiliary torch parameter for the success probabilities p for each implicit node
        self.p_aux = Parameter()

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.join(s2), self.scopes_out)

        # parse weights
        self.set_params(p)
        self.backend = "pytorch"

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``BernoulliLayer`` can represent one or more univariate nodes with ``MetaType.discrete`` or ``BernoulliType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not Bernoulli.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "BernoulliLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``BernoulliLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'BernoulliLayer' cannot be instantiated from the following signatures: {signatures}.")

        p = []
        scopes = []

        for feature_ctx in signatures:

            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if domain == MetaType.Discrete:
                p.append(0.5)
            elif domain == FeatureTypes.Bernoulli:
                # instantiate object
                p.append(domain().p)
            elif isinstance(domain, FeatureTypes.Bernoulli):
                p.append(domain.p)
            else:
                raise ValueError(
                    f"Unknown signature domain {domain} for 'BernoulliLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return BernoulliLayer(scopes, p=p)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def p(self) -> torch.Tensor:
        """Returns the success probabilities."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)  # type: ignore

    @p.setter
    def p(self, p: Union[int, float, List[float], np.ndarray, torch.Tensor]) -> None:
        r"""Sets the success probability.

        Args:
            p:
                Floating point, list of floats, one-dimensional NumPy array or PyTorch tensor representing the success probabilities of the Bernoulli distributions between zero and one.
                If a single value is given it is broadcast to all nodes.

        Raises:
            ValueError: Invalid arguments.
        """
        if isinstance(p, float) or isinstance(p, int):
            p = torch.tensor([p for _ in range(self.n_out)], dtype=self.dtype, device=self.device)
        elif isinstance(p, list) or isinstance(p, np.ndarray):
            p = torch.tensor(p, dtype=self.dtype, device=self.device)
        if p.ndim != 1:
            raise ValueError(
                f"Numpy array of 'p' values for 'BernoulliLayer' is expected to be one-dimensional, but is {p.ndim}-dimensional."
            )
        if p.shape[0] == 1:
            p = torch.hstack([p for _ in range(self.n_out)])
        if p.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'p' values for 'BernoulliLayer' must match number of output nodes {self.n_out}, but is {p.shape[0]}"
            )
        if torch.any(p < 0.0) or torch.any(p > 1.0) or not all(torch.isfinite(p)):
            raise ValueError(
                f"Values of 'p' for 'BernoulliLayer' distribution must to be between 0.0 and 1.0, but are: {p}"
            )

        self.p_aux.data = proj_bounded_to_real(p, lb=0.0, ub=1.0)

    def dist(self, node_ids: Optional[List[int]] = None) -> D.Distribution:
        r"""Returns the PyTorch distributions represented by the leaf layer.

        Args:
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            ``torch.distributions.Bernoulli`` instance.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Bernoulli(probs=self.p[node_ids])

    def set_params(self, p: Union[int, float, List[float], np.ndarray, torch.Tensor] = 0.5) -> None:
        """Sets the parameters for the represented distributions.

        Bounded parameter ``p`` is projected onto the unbounded parameter ``p_aux``.

        TODO: projection function

        Args:
            p:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor representing the success probabilities of the Bernoulli distributions between zero and one.
                If a single value is given it is broadcast to all nodes.
                Defaults to 0.5.
        """
        self.p = p

    def get_trainable_params(self) -> List[torch.Tensor]:
        """Returns the parameters of the represented distribution.

        Returns:
            One-dimensional PyTorch tensor representing the success probabilities.
        """
        return [self.p_aux]

    def get_params(self):
        return self.p.data.cpu().numpy()

    def check_support(
        self,
        data: torch.Tensor,
        node_ids: Optional[List[int]] = None,
        is_scope_data: bool = False,
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or not instances are part of the supports of the Bernoulli distributions, which are:

        .. math::

            \text{supp}(\text{Bernoulli})=\{0,1\}

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
        self.p_aux.data = self.p_aux.data.type(dtype)

    def to_device(self, device):
        self.device = device
        self.p_aux.data = self.p_aux.data.to(device)


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: BernoulliLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[BernoulliLayer, Bernoulli, None]:
    """Structural marginalization for ``BernoulliLayer`` objects in the ``torch`` backend.

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
        return Bernoulli(scope=marginalized_scopes[0], p=layer.p[node_id].item())
    else:
        return BernoulliLayer(scope=marginalized_scopes, p=layer.p[marginalized_node_ids].detach())


@dispatch(memoize=True)  # type: ignore
def toTorch(layer: BaseBernoulliLayer, dispatch_ctx: Optional[DispatchContext] = None) -> BernoulliLayer:
    """Conversion for ``BernoulliLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BernoulliLayer(scope=layer.scopes_out, p=layer.p)


@dispatch(memoize=True)  # type: ignore
def toBase(layer: BernoulliLayer, dispatch_ctx: Optional[DispatchContext] = None) -> BaseBernoulliLayer:
    """Conversion for ``BernoulliLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseBernoulliLayer(scope=layer.scopes_out, p=layer.p.detach().numpy())

@dispatch(memoize=True)  # type: ignore
def updateBackend(leaf_node: BernoulliLayer, dispatch_ctx: Optional[DispatchContext] = None):
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return GeneralBernoulli(scope=leaf_node.scopes_out, p=leaf_node.p.data)


