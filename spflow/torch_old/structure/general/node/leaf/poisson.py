"""Contains Poisson leaf node for SPFlow in the ``torch`` backend.
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter

from spflow.base.structure.general.node.leaf.poisson import (
    Poisson as BasePoisson,
)
from spflow.modules.node import Poisson as GeneralPoisson
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.node.leaf_node import LeafNode
from spflow.torch.utils.projections import proj_bounded_to_real, proj_real_to_bounded


class Poisson(LeafNode):
    r"""(Univariate) Poisson distribution leaf node in the ``torch`` backend.

    Represents a univariate Poisson distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \lambda^k\frac{e^{-\lambda}}{k!}

    where
        - :math:`k` is the number of occurrences
        - :math:`\lambda` is the rate parameter

    Internally :math:`l` is represented as an unbounded parameter that is projected onto the bounded range :math:`[0,\infty)` for representing the actual rate probability.

    Attributes:
        l_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual rate parameter.
        l:
            Scalar PyTorch tensor representing the rate parameters (:math:`\lambda`) of the Poisson distribution (projected from ``l_aux``).
    """

    def __init__(self, scope: Scope, l: Optional[float] = 1.0) -> None:
        r"""Initializes ``Poisson`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            l:
                Floating point value representing the rate parameter (:math:`\lambda`), expected value and variance of the Poisson distribution (must be greater than or equal to 0).
                Defaults to 1.0.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Poisson' should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Poisson' should be empty, but was {scope.evidence}.")

        super().__init__(scope=scope)

        # register auxiliary torch parameter for lambda l
        self.l_aux = Parameter()

        # set parameters
        self.set_params(l)
        self.backend = "pytorch"

    @property
    def l(self) -> torch.Tensor:
        """Returns the rate parameter."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.l_aux, lb=0.0)  # type: ignore

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Poisson`` can represent a single univariate node with ``MetaType.Discrete`` or ``PoissonType`` domain.

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

        # leaf is a discrete Poisson distribution
        if not (
            domains[0] == FeatureTypes.Discrete
            or domains[0] == FeatureTypes.Poisson
            or isinstance(domains[0], FeatureTypes.Poisson)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Poisson":
        """Creates an instance from a specified signature.

        Returns:
            ``Poisson`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'Poisson' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Discrete:
            l = 1.0
        elif domain == FeatureTypes.Poisson:
            # instantiate object
            l = domain().l
        elif isinstance(domain, FeatureTypes.Poisson):
            l = domain.l
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Poisson' that was not caught during acception checking."
            )

        return Poisson(feature_ctx.scope, l=l)

    @property
    def dist(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.Poisson`` instance.
        """
        return D.Poisson(rate=self.l)

    def set_params(self, l: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            l:
                Floating point value representing the rate parameter (:math:`\lambda`), expected value and variance of the Poisson distribution (must be greater than or equal to 0).
        """
        if not np.isfinite(l):
            raise ValueError(f"Value of l for Poisson distribution must be finite, but was: {l}")

        if l < 0:
            raise ValueError(f"Value of l for Poisson distribution must be non-negative, but was: {l}")

        self.l_aux.data = proj_bounded_to_real(
            torch.tensor(float(l), dtype=self.dtype, device=self.device), lb=0.0
        )

    def get_trainable_params(self) -> tuple[float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the rate parameter, expected value and variance.
        """
        # return (self.l.data.cpu().numpy(),)  # type: ignore
        return [self.l_aux]  # type: ignore

    def get_params(self) -> tuple[float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the rate parameter, expected value and variance.
        """
        return (self.l.data.cpu().numpy(),)  # type: ignore

    def check_support(self, data: torch.Tensor, is_scope_data: bool = False) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Poisson distribution, which is:

        .. math::

            \text{supp}(\text{Poisson})=\mathbb{N}\cup\{0\}

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
            Two-dimensional PyTorch tensor indicating for each instance, whether they are part of the support (True) or not (False).
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

        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool, device=self.device)
        valid[~nan_mask] = self.dist.support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check if all values are valid integers
        valid[~nan_mask & valid] &= (
            torch.remainder(scope_data[~nan_mask & valid], torch.tensor(1)).squeeze(-1) == 0
        )

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf().squeeze(-1)

        return valid

    def to_dtype(self, dtype):
        self.dtype = dtype
        self.set_params(self.l.data)

    def to_device(self, device):
        self.device = device
        self.set_params(self.l.data)


@dispatch(memoize=True)  # type: ignore
def toTorch(node: BasePoisson, dispatch_ctx: Optional[DispatchContext] = None) -> Poisson:
    """Conversion for ``Poisson`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return Poisson(node.scope, *node.get_trainable_params())


@dispatch(memoize=True)  # type: ignore
def toBase(node: Poisson, dispatch_ctx: Optional[DispatchContext] = None) -> BasePoisson:
    """Conversion for ``Poisson`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BasePoisson(node.scope, *node.get_trainable_params())


@dispatch(memoize=True)  # type: ignore
def updateBackend(leaf_node: Poisson, dispatch_ctx: Optional[DispatchContext] = None):
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return GeneralPoisson(scope=leaf_node.scope, l=leaf_node.l.data.item())
