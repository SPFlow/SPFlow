from spflow.modules.node.leaf_node import LeafNode
import torch
from torch import nn, Tensor
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
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
        low: float,
        high: float,
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
            raise ValueError(f"Query scope size for 'Uniform' should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Uniform' should be empty, but was {scope.evidence}.")
        if low >= high:
            raise ValueError(
                f"Value of 'start' for 'Uniform' must be smaller than 'end', but was: {low} >= {high}."
            )

        super().__init__(scope=scope)

        self.register_buffer("low", torch.tensor(low))
        self.register_buffer("high", torch.tensor(high))
        self.support_outside = support_outside

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the PyTorch distribution represented by the leaf node."""
        return torch.distributions.Uniform(self.low, self.high)

    @property
    def high_next(self) -> Tensor:
        return torch.nextafter(self.high, torch.tensor(float("inf")))

    @property
    def device(self) -> torch.device:
        """Overwrite since this leaf has no parameters and the Module device method iterates over the parameters."""
        return self.low.device

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
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
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Uniform":
        """Creates an instance from a specified signature.

        Returns:
            ``Uniform`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'Uniform' cannot be instantiated from the following signatures: {signatures}.")

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

        return Uniform(feature_ctx.scope, low=start, high=end)


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: Uniform,
    data: Tensor,
    weights: Optional[Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``Uniform`` node parameters in the ``torch`` backend.

    All parameters of the Uniform distribution are regarded as fixed and will not be estimated.
    Therefore, this method does nothing, but check for the validity of the data.

    Args:
        leaf:
            Leaf node to estimate parameters of.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        weights:
            Optional one-dimensional PyTorch tensor containing non-negative weights for all data samples.
            Must match number of samples in ``data``.
            Defaults to None in which case all weights are initialized to ones.
        bias_corrections:
            Boolen indicating whether or not to correct possible biases.
            Has no effects for ``Uniform`` nodes.
            Defaults to True.
        nan_strategy:
            Optional string or callable specifying how to handle missing data.
            If 'ignore', missing values (i.e., NaN entries) are ignored.
            If a callable, it is called using ``data`` and should return another PyTorch tensor of same size.
            Defaults to None.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Raises:
        ValueError: Invalid arguments.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    if check_support:
        if torch.any(~leaf.check_support(data[:, leaf.scope.query])):
            raise ValueError("Encountered values outside of the support for 'Uniform'.")
