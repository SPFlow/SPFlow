# -*- coding: utf-8 -*-
"""Contains Uniform leaf node for SPFlow in the ``torch`` backend.
"""
import numpy as np
import torch
import torch.distributions as D
from typing import Tuple, Optional, List, Union, Type
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import MetaType, FeatureType, FeatureTypes
from spflow.meta.data.feature_context import FeatureContext

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.uniform import (
    Uniform as BaseUniform,
)


class Uniform(LeafNode):
    r"""(Univariate) continuous Uniform distribution leaf node in the ``torch`` backend.

    Represents an univariate Uniform distribution, with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\text{end} - \text{start}}\mathbf{1}_{[\text{start}, \text{end}]}(x)

    where
        - :math:`x` is the input observation
        - :math:`\mathbf{1}_{[\text{start}, \text{end}]}` is the indicator function for the given interval (evaluating to 0 if x is not in the interval)

    Attributes:
        dist:
            ``torch.distributions.Uniform`` instance of the PyTorch distribution represented by the leaf node.
        start:
            Scalar PyTorch tensor representing the start of the interval (including).
        end:
            Scalar PyTorch tensor representing the end of the interval (including). Must be larger than 'start'.
        end_next:
            Scalary PyTorch tensor containing the next largest floating point value to ``end``.
            Used for the PyTorch distribution which does not include the specified end of the interval.
        support_outside:
            Scalar PyTorch tensor indicating whether or not values outside of the interval are part of the support.
    """

    def __init__(
        self,
        scope: Scope,
        start: float,
        end: float,
        support_outside: bool = True,
    ) -> None:
        r"""Initializes ``Uniform`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            start:
                Floating point value representing the start of the interval (including).
            end:
                Floating point value representing the end of the interval (including). Must be larger than 'start'.
            support_outside:
                Boolean indicating whether or not values outside of the interval are part of the support.
                Defaults to True.
        """
        if len(scope.query) != 1:
            raise ValueError(
                f"Query scope size for 'Uniform' should be 1, but was: {len(scope.query)}."
            )
        if len(scope.evidence) != 0:
            raise ValueError(
                f"Evidence scope for 'Uniform' should be empty, but was {scope.evidence}."
            )

        super(Uniform, self).__init__(scope=scope)

        # register interval bounds as torch buffers (should not be changed)
        self.register_buffer("start", torch.empty(size=[]))
        self.register_buffer("end", torch.empty(size=[]))
        self.register_buffer("end_next", torch.empty(size=[]))

        # set parameters
        self.set_params(start, end, support_outside)

    @classmethod
    def accepts(self, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Uniform`` can represent a single univariate node with with ``UniformType`` domain.

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

        # leaf is a continuous Uniform distribution
        # NOTE: only accept instances of 'FeatureTypes.Uniform', otherwise required parameters 'start','end' are not specified. Reject 'FeatureTypes.Continuous' for the same reason.
        if not isinstance(domains[0], FeatureTypes.Uniform):
            return False

        return True

    @classmethod
    def from_signatures(self, signatures: List[FeatureContext]) -> "Uniform":
        """Creates an instance from a specified signature.

        Returns:
            ``Uniform`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not self.accepts(signatures):
            raise ValueError(
                f"'Uniform' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if isinstance(domain, FeatureTypes.Uniform):
            start, end = domain.start, domain.end
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Uniform' that was not caught during acception checking."
            )

        return Uniform(feature_ctx.scope, start=start, end=end)

    def set_params(
        self, start: float, end: float, support_outside: bool = True
    ) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            start:
                Floating point value representing the start of the interval (including).
            end:
                Floating point value representing the end of the interval (including). Must be larger than ``start``.
            support_outside:
                Boolean indicating whether or not values outside of the interval are part of the support.
                Defaults to True.
        """
        if not start < end:
            raise ValueError(
                f"Value of 'start' for 'Uniform' must be less than value of 'end', but were: {start}, {end}"
            )
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(
                f"Values of 'start' and 'end' for 'Uniform' must be finite, but were: {start}, {end}"
            )

        # since torch Uniform distribution excludes the upper bound, compute next largest number
        end_next = torch.nextafter(torch.tensor(end), torch.tensor(float("Inf")))  # type: ignore

        self.start.data = torch.tensor(float(start))  # type: ignore
        self.end.data = torch.tensor(float(end))  # type: ignore
        self.end_next.data = torch.tensor(float(end_next))
        self.support_outside = support_outside

        # create Torch distribution with specified parameters
        self.dist = D.Uniform(low=self.start, high=end_next)

    def get_params(self) -> Tuple[float, float, bool]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of the floating point values representing the start and end of the interval and the boolean indicating whether or not values outside of the interval are part of the support.
        """
        return self.start.cpu().numpy(), self.end.cpu().numpy(), self.support_outside  # type: ignore

    def check_support(
        self, data: torch.Tensor, is_scope_data: bool = False
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Uniform distribution, which is:

        .. math::

            \text{supp}(\text{Uniform})=\begin{cases} [start,end] & \text{if support\_outside}=\text{false}\\
                                                 (-\infty,\infty) & \text{if support\_outside}=\text{true} \end{cases}
        where
            - :math:`start` is the start of the interval
            - :math:`end` is the end of the interval
            - :math:`\text{support\_outside}` is a truth value indicating whether values outside of the interval are part of the support

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leaf's scope in the correct order (True) or if it needs to be extracted from the full data set.
                Defaults to False

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

        # torch distribution support is an interval, despite representing a distribution over a half-open interval
        # end is adjusted to the next largest number to make sure that desired end is part of the distribution interval
        # may cause issues with the support check; easier to do a manual check instead
        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)

        # check for infinite values
        valid[~nan_mask & valid] &= (
            ~scope_data[~nan_mask & valid].isinf().squeeze(-1)
        )

        # check if values are within valid range
        if not self.support_outside:
            valid[~nan_mask & valid] &= (
                (scope_data[~nan_mask & valid] >= self.start)
                & (scope_data[~nan_mask & valid] < self.end_next)
            ).squeeze(-1)

        return valid


@dispatch(memoize=True)  # type: ignore
def toTorch(
    node: BaseUniform, dispatch_ctx: Optional[DispatchContext] = None
) -> Uniform:
    """Conversion for ``Uniform`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return Uniform(node.scope, node.start, node.end)


@dispatch(memoize=True)  # type: ignore
def toBase(
    node: Uniform, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseUniform:
    """Conversion for ``Uniform`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseUniform(
        node.scope, node.start.cpu().numpy(), node.end.cpu().numpy()
    )
