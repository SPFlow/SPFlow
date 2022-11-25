"""Contains Binomial leaf node for SPFlow in the ``torch`` backend.
"""
import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import List, Tuple, Optional
from spflow.torch.utils.projections import (
    proj_bounded_to_real,
    proj_real_to_bounded,
)
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaf_node import LeafNode
from spflow.base.structure.general.nodes.leaves.parametric.binomial import (
    Binomial as BaseBinomial,
)


class Binomial(LeafNode):
    r"""(Univariate) Binomial distribution leaf node in the ``torch`` backend.

    Represents an univariate Binomial distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \binom{n}{k}p^k(1-p)^{n-k}

    where
        - :math:`p` is the success probability of each trial in :math:`[0,1]`
        - :math:`n` is the number of total trials
        - :math:`k` is the number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Internally :math:`p` is represented as an unbounded parameter that is projected onto the bounded range :math:`[0,1]` for representing the actual success probability.

    Attributes:
        n:
            Scalar PyTorch tensor representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
        p_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual success probability.
        p:
            Scalar PyTorch tensor representing the success probability of the Bernoulli distribution (projected from ``p_aux``).
    """

    def __init__(self, scope: Scope, n: int, p: float = 0.5) -> None:
        r"""Initializes ``Binomial`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            n:
                Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
            p:
                Floating point value representing the success probability of each trial between zero and one.
                Defaults to 0.5.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'Binomial' should be 1, but was {len(scope.query)}."
            )
        if len(scope.evidence) != 0:
            raise ValueError(
                f"Evidence scope for 'Binomial' should be empty, but was {scope.evidence}."
            )

        super().__init__(scope=scope)

        # register number of trials n as torch buffer (should not be changed)
        self.register_buffer("n", torch.empty(size=[]))

        # register auxiliary torch parameter for the success probability p
        self.p_aux = Parameter()

        # set parameters
        self.set_params(n, p)

    @property
    def p(self) -> torch.Tensor:
        """Returns the success proability."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)  # type: ignore

    @p.setter
    def p(self, p: float) -> None:
        r"""Sets the success probability.

        Args:
            p:
                Floating point representing the success probability in :math:`[0,1]`.

        Raises:
            ValueError: Invalid arguments.
        """
        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of 'p' for 'Binomial' distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        self.p_aux.data = proj_bounded_to_real(torch.tensor(float(p)), lb=0.0, ub=1.0)  # type: ignore

    @classmethod
    def accepts(self, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Binomial`` can represent a single univariate node with ``BinomialType`` domain.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf only has one output
        if len(signatures) != 1:
            return False

        # get single output signature
        feature_ctx = signatures[0]
        domains = feature_ctx.get_domains()

        # leaf is a single non-conditional univariate node
        if (
            len(domains) != 1
            or len(feature_ctx.scope.query) != len(domains)
            or len(feature_ctx.scope.evidence) != 0
        ):
            return False

        # leaf is a discrete Binomial distribution
        # NOTE: only accept instances of 'FeatureTypes.Binomial', otherwise required parameter 'n' is not specified. Reject 'FeatureTypes.Discrete' for the same reason.
        if not isinstance(domains[0], FeatureTypes.Binomial):
            return False

        return True

    @classmethod
    def from_signatures(self, signatures: List[FeatureContext]) -> "Binomial":
        """Creates an instance from a specified signature.

        Returns:
            ``Binomial`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not self.accepts(signatures):
            raise ValueError(
                f"'Binomial' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if isinstance(domain, FeatureTypes.Binomial):
            n = domain.n
            p = domain.p
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Binomial' that was not caught during acception checking."
            )

        return Binomial(feature_ctx.scope, n=n, p=p)

    @property
    def dist(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.Binomial`` instance.
        """
        return D.Binomial(total_count=self.n, probs=self.p)

    def set_params(self, n: int, p: float) -> None:
        """Sets the parameters for the represented distribution.

        Bounded parameter ``p`` is projected onto the unbounded parameter ``p_aux``.

        TODO: projection function

        Args:
            n:
                Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
            p:
                Floating point value representing the success probability of the Binomial distribution between zero and one.
        """
        if isinstance(n, float):
            if not n.is_integer():
                raise ValueError(
                    f"Value of 'n' for 'Binomial' must be (equal to) an integer value, but was: {n}"
                )
            n = torch.tensor(int(n))
        elif isinstance(n, int):
            n = torch.tensor(n)
        if n < 0 or not torch.isfinite(n):
            raise ValueError(
                f"Value of 'n' for 'Binomial' must to greater of equal to 0, but was: {n}"
            )

        self.p = torch.tensor(float(p))
        self.n.data = torch.tensor(int(n))  # type: ignore

    def get_params(self) -> Tuple[int, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Integer number representing the number of i.i.d. Bernoulli trials and the floating point value representing the success probability.
        """
        return self.n.data.cpu().numpy(), self.p.data.cpu().numpy()  # type: ignore

    def check_support(
        self, data: torch.Tensor, is_scope_data: bool = False
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Binomial distribution, which is:

        .. math::

            \text{supp}(\text{Binomial})=\{0,\hdots,n\}

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leaf's scope in the correct order (True) or if it needs to be extracted from the full data set.
                Defaults to False.

        Returns:
            Two dimensional PyTorch tensor indicating for each instance, whether they are part of the support (True) or not (False).
        """
        if is_scope_data:
            scope_data = data
        else:
            # select relevant data for scope
            scope_data = data[:, self.scope.query]

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope.query):
            raise ValueError(
                f"Expected 'scope_data' to be of shape (n,{len(self.scope.query)}), but was: {scope_data.shape}"
            )

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)
        valid[~nan_mask] = self.dist.support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check for infinite values
        valid[~nan_mask & valid] &= (
            ~scope_data[~nan_mask & valid].isinf().squeeze(-1)
        )

        return valid


@dispatch(memoize=True)  # type: ignore
def toTorch(
    node: BaseBinomial, dispatch_ctx: Optional[DispatchContext] = None
) -> Binomial:
    """Conversion for ``Binomial`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return Binomial(node.scope, *node.get_params())


@dispatch(memoize=True)  # type: ignore
def toBase(
    node: Binomial, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseBinomial:
    """Conversion for ``Binomial`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseBinomial(node.scope, *node.get_params())
