"""Contains Binomial leaf layer for SPFlow in the ``torch`` backend.
"""
from functools import reduce
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter

from spflow.base.structure.general.layers.leaves.parametric.binomial import (
    BinomialLayer as BaseBinomialLayer,
)
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaves.parametric.binomial import Binomial
from spflow.torch.structure.module import Module
from spflow.torch.utils.projections import proj_bounded_to_real, proj_real_to_bounded


class BinomialLayer(Module):
    r"""Layer of multiple (univariate) Binomial distribution leaf nodes in the 'base' backend.

    Represents multiple univariate Binomial distributions with independent scopes, each with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \binom{n}{k}p^k(1-p)^{n-k}

    where
        - :math:`p` is the success probability of each trial in :math:`[0,1]`
        - :math:`n` is the number of total trials
        - :math:`k` is the number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Internally :math:`p` are represented as unbounded parameters that are projected onto the bounded range :math:`[0,1]` for representing the actual success probabilities.

    Attributes:
        n:
            One-dimensional PyTorch tensor containing the number of i.i.d. Bernoulli trials (greater or equal to 0) for each independent Binomial distribution.
        p_aux:
            Unbounded one-dimensional PyTorch parameter that is projected to yield the actual success probabilities.
        p:
            One-dimensional PyTorch tensor representing the success probabilities of the Bernoulli distributions (projected from ``p_aux``).
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        n: Union[int, List[int], np.ndarray, torch.Tensor],
        p: Union[int, float, List[float], np.ndarray, torch.Tensor] = 0.5,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``BinomialLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            n:
                Integer, list of integers, one-dimensional NumPy array or PyTorch tensor containing the number of i.i.d. Bernoulli trials (greater or equal to 0) for each independent Binomial distribution.
                If a single value is given it is broadcast to all nodes.
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
                    f"Number of nodes for 'BinomialLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'BinomialLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")
            if len(s.evidence) != 0:
                raise ValueError(f"Evidence scope for 'BinomialLayer' should be empty, but was {s.evidence}.")

        super().__init__(children=[], **kwargs)

        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("n", torch.empty(size=[]))

        # register auxiliary torch parameter for the success probabilities p for each implicit node
        self.p_aux = Parameter()

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.join(s2), self.scopes_out)

        # parse weights
        self.set_params(n, p)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``BinomialLayer`` can represent one or more univariate nodes with ``BinomialType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not Binomial.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "BinomialLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``BinomialLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'BinomialLayer' cannot be instantiated from the following signatures: {signatures}.")

        n = []
        p = []
        scopes = []

        for feature_ctx in signatures:

            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if isinstance(domain, FeatureTypes.Binomial):
                n.append(domain.n)
                p.append(domain.p)
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'BinomialLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return BinomialLayer(scopes, n=n, p=p)

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
            p = torch.tensor([p for _ in range(self.n_out)])
        elif isinstance(p, list) or isinstance(p, np.ndarray):
            p = torch.tensor(p)
        if p.ndim != 1:
            raise ValueError(
                f"Numpy array of 'p' values for 'BinomialLayer' is expected to be one-dimensional, but is {p.ndim}-dimensional."
            )
        if p.shape[0] == 1:
            p = torch.hstack([p for _ in range(self.n_out)])
        if p.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'p' values for 'BinomialLayer' must match number of output nodes {self.n_out}, but is {p.shape[0]}"
            )
        if torch.any(p < 0.0) or torch.any(p > 1.0) or not all(torch.isfinite(p)):
            raise ValueError(
                f"Values of 'p' for 'BinomialLayer' distribution must to be between 0.0 and 1.0, but are: {p}"
            )

        self.p_aux.data = proj_bounded_to_real(p, lb=0.0, ub=1.0)

    def dist(self, node_ids: Optional[List[int]] = None) -> D.Distribution:
        r"""Returns the PyTorch distributions represented by the leaf layer.

        Args:
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            ``torch.distributions.Binomial`` instance.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Binomial(total_count=self.n[node_ids], probs=self.p[node_ids])

    def set_params(
        self,
        n: Union[int, List[int], np.ndarray, torch.Tensor],
        p: Union[int, float, List[float], np.ndarray, torch.Tensor] = 0.5,
    ) -> None:
        """Sets the parameters for the represented distributions.

        Bounded parameter ``p`` is projected onto the unbounded parameter ``p_aux``.

        TODO: projection function

        Args:
            p:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor representing the success probabilities of the Bernoulli distributions between zero and one.
                If a single value is given it is broadcast to all nodes.
                Defaults to 0.5.
        """
        if isinstance(n, int) or isinstance(n, float):
            n = torch.tensor([n for _ in range(self.n_out)])
        elif isinstance(n, list) or isinstance(n, np.ndarray):
            n = torch.tensor(n)
        if n.ndim != 1:
            raise ValueError(
                f"Numpy array of 'n' values for 'BinomialLayer' is expected to be one-dimensional, but is {n.ndim}-dimensional."
            )
        if n.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'n' values for 'BinomialLayer' must match number of output nodes {self.n_out}, but is {n.shape[0]}"
            )

        if torch.any(n < 0) or not torch.any(torch.isfinite(n)):
            raise ValueError(f"Values for 'n' of 'BinomialLayer' must to greater of equal to 0, but was: {n}")

        if not torch.all(torch.remainder(n, 1.0) == torch.tensor(0.0)):
            raise ValueError(f"Values for 'n' of 'BinomialLayer' must be (equal to) an integer value, but was: {n}")

        node_scopes = torch.tensor([s.query[0] for s in self.scopes_out])

        for node_scope in torch.unique(node_scopes):
            # at least one such element exists
            n_values = n[node_scopes == node_scope]
            if not torch.all(n_values == n_values[0]):
                raise ValueError("All values of 'n' for 'BinomialLayer' over the same scope must be identical.")

        self.p = p
        self.n.data = n

    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of two one-dimensional PyTorch tensors representing the number of i.i.d. Bernoulli trials and the success probabilities, respectively.
        """
        return (self.n, self.p)

    def check_support(
        self,
        data: torch.Tensor,
        node_ids: Optional[List[int]] = None,
        is_scope_data: bool = False,
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Binomial distributions, which are:

        .. math::

            \text{supp}(\text{Binomial})=\{0,\hdots,n\}

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


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: BinomialLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[BinomialLayer, Binomial, None]:
    """Structural marginalization for ``BinomialLayer`` objects in the ``torch`` backend.

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
        return Binomial(
            scope=marginalized_scopes[0],
            n=layer.n[node_id].item(),
            p=layer.p[node_id].item(),
        )
    else:
        return BinomialLayer(
            scope=marginalized_scopes,
            n=layer.n[marginalized_node_ids].detach(),
            p=layer.p[marginalized_node_ids].detach(),
        )


@dispatch(memoize=True)  # type: ignore
def toTorch(layer: BaseBinomialLayer, dispatch_ctx: Optional[DispatchContext] = None) -> BinomialLayer:
    """Conversion for ``BinomialLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BinomialLayer(scope=layer.scopes_out, n=layer.n, p=layer.p)


@dispatch(memoize=True)  # type: ignore
def toBase(layer: BinomialLayer, dispatch_ctx: Optional[DispatchContext] = None) -> BaseBinomialLayer:
    """Conversion for ``BinomialLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseBinomialLayer(scope=layer.scopes_out, n=layer.n.numpy(), p=layer.p.detach().numpy())
