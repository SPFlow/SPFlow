# -*- coding: utf-8 -*-
"""Contains Exponential leaf node for SPFlow in the ``torch`` backend.
"""
import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import Tuple, Optional, List, Union, Type
from .projections import proj_bounded_to_real, proj_real_to_bounded
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import MetaType, FeatureType, FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.spn.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.exponential import (
    Exponential as BaseExponential,
)


class Exponential(LeafNode):
    r"""(Univariate) Exponential distribution leaf node in the ``torch`` backend.

    Represents an univariate Exponential distribution, with the following probability distribution function (PDF):

    .. math::
        
        \text{PDF}(x) = \begin{cases} \lambda e^{-\lambda x} & \text{if } x > 0\\
                                      0                      & \text{if } x <= 0\end{cases}
    
    where
        - :math:`x` is the input observation
        - :math:`\lambda` is the rate parameter
    
    Internally :math:`l` is represented as an unbounded parameter that is projected onto the bounded range :math:`(0,\infty)` for representing the actual rate parameters.

    Attributes:
        l_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual rate parameter.
        l:
            Scalar PyTorch tensor representing the rate parameter (:math:`\lambda`) of the Exponential distribution (projected from ``l_aux``).
    """

    def __init__(self, scope: Scope, l: float = 1.0) -> None:
        r"""Initializes ``Exponential`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            l:
                Floating point value representing the rate parameter (:math:`\lambda`) of the Exponential distribution (must be greater than 0).
                Defaults to 1.0.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'Exponential' should be 1, but was {len(scope.query)}."
            )
        if len(scope.evidence) != 0:
            raise ValueError(
                f"Evidence scope for 'Exponential' should be empty, but was {scope.evidence}."
            )

        super(Exponential, self).__init__(scope=scope)

        # register auxiliary torch parameter for parameter l
        self.l_aux = Parameter()

        # set parameters
        self.set_params(l)

    @property
    def l(self) -> torch.Tensor:
        """Returns the rate parameter."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.l_aux, lb=0.0)  # type: ignore

    @classmethod
    def accepts(self, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Exponential`` can represent a single univariate node with ``MetaType.Continuous`` or ``ExponentialType`` domain.

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

        # leaf is a discrete Exponential distribution
        if not (
            domains[0] == FeatureTypes.Continuous
            or domains[0] == FeatureTypes.Exponential
            or isinstance(domains[0], FeatureTypes.Exponential)
        ):
            return False

        return True

    @classmethod
    def from_signatures(
        self, signatures: List[FeatureContext]
    ) -> "Exponential":
        """Creates an instance from a specified signature.

        Returns:
            ``Exponential`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not self.accepts(signatures):
            raise ValueError(
                f"'Exponential' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Continuous:
            l = 1.0
        elif domain == FeatureTypes.Exponential:
            # instantiate object
            l = domain().l
        elif isinstance(domain, FeatureTypes.Exponential):
            l = domain.l
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Exponential' that was not caught during acception checking."
            )

        return Exponential(feature_ctx.scope, l=l)

    @property
    def dist(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.Exponential`` instance.
        """
        return D.Exponential(rate=self.l)

    def set_params(self, l: float) -> None:
        r"""Sets the parameters for the represented distribution.

        TODO: projection function

        Args:
            l:
                Floating point value representing the rate parameter (:math:`\lambda`) of the Exponential distribution (must be greater than 0).
        """
        if l <= 0.0 or not np.isfinite(l):
            raise ValueError(
                f"Value of 'l' for 'Exponential' must be greater than 0, but was: {l}"
            )

        self.l_aux.data = proj_bounded_to_real(torch.tensor(float(l)), lb=0.0)

    def get_params(self) -> Tuple[float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the rate parameter.
        """
        return (self.l.data.cpu().numpy(),)  # type: ignore

    def check_support(
        self, data: torch.Tensor, is_scope_data: bool = False
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Exponential distribution, which is:

        .. math::

            \text{supp}(\text{Exponential})=[0,+\infty)

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

        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)
        valid[~nan_mask] = self.dist.support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check for infinite values
        valid[~nan_mask & valid] &= (
            ~scope_data[~nan_mask & valid].isinf().squeeze(-1)
        )

        return valid


@dispatch(memoize=True)  # type: ignore
def toTorch(
    node: BaseExponential, dispatch_ctx: Optional[DispatchContext] = None
) -> Exponential:
    """Conversion for ``Exponential`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return Exponential(node.scope, *node.get_params())


@dispatch(memoize=True)  # type: ignore
def toBase(
    node: Exponential, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseExponential:
    """Conversion for ``Exponential`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseExponential(node.scope, *node.get_params())
