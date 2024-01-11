"""Contains Geometric leaf node for SPFlow in the ``torch`` backend.
"""
from spflow.modules.node.leaf.utils import apply_nan_strategy
from spflow.modules.node.leaf_node import LeafNode
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter

from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded


from typing import Callable, Optional, Union

import torch
from torch import nn, Tensor

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class Geometric(LeafNode):
    r"""(Univariate) Geometric distribution leaf node in the ``torch`` backend.

    Represents an univariate Geometric distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) =  p(1-p)^{k-1}

    where
        - :math:`k` is the number of trials
        - :math:`p` is the success probability of each trial

    Internally :math:`p` is represented as an unbounded parameter that is projected onto the bounded range :math:`(0,1]` for representing the actual success probability.

    Attributes:
        p_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual success probability.
        p:
            Scalar PyTorch tensor representing the success probability in the range :math:`(0,1]` (projected from ``p_aux``).
    """

    def __init__(self, scope: Scope, p: float = 0.5) -> None:
        r"""Initializes ``Geometric`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            p:
                Floating points representing the probability of success in the range :math:`(0,1]`.
                Defaults to 0.5.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Geometric' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Geometric' should be empty, but was {scope.evidence}.")

        super().__init__(scope=scope)

        # register auxiliary torch parameter for the success probability p
        self.p_aux = torch.nn.Parameter(torch.empty(1))
        self.p = torch.tensor(p)

    @property
    def p(self) -> Tensor:
        """Returns the success proability."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.p_aux, 0.0, 1.0)

    @p.setter
    def p(self, p: Tensor) -> None:
        """Sets the success probability."""
        # project auxiliary parameter onto actual parameter range
        if p < 0.0 or p > 1.0 or not torch.isfinite(p):
            raise ValueError(f"Value of 'p' for 'Geometric' must to be between 0.0 and 1.0, but was: {p}")
        self.p_aux.data = proj_bounded_to_real(p, 0.0, 1.0)

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Geometric`` can represent a single univariate node with ``MetaType.Discrete`` or ``GeometricType`` domain.

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

        # leaf is a discrete Geometric distribution
        if not (
            domains[0] == FeatureTypes.Discrete
            or domains[0] == FeatureTypes.Geometric
            or isinstance(domains[0], FeatureTypes.Geometric)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Geometric":
        """Creates an instance from a specified signature.

        Returns:
            ``Geometric`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'Geometric' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Discrete:
            p = 0.5
        elif domain == FeatureTypes.Geometric:
            # instantiate object
            p = domain().p
        elif isinstance(domain, FeatureTypes.Geometric):
            p = domain.p
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Geometric' that was not caught during acception checking."
            )

        return Geometric(feature_ctx.scope, p=p)

    @property
    def distribution(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Note, that the Geometric distribution as implemented in PyTorch uses :math:`k-1` as input.
        Therefore values are offset by 1 if used directly.

        Returns:
            ``torch.distributions.Geometric`` instance.
        """
        return D.Geometric(probs=self.p)


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: Geometric,
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``Geometric`` node parameters in the ``torch`` backend.

    Estimates the success probability :math:`p` of a Gaussian distribution from data, as follows:

    .. math::

        p^{\*}=\begin{cases} \frac{1}{\sum_{i=1}^N w_i}\sum_{i=1}^{N}w_ix_i & \text{if } \sum_{i=1}^Nw_ix_i\ne0\\
                             0 & \text{if } \sum_{i=1}^Nw_ix_i=0 \end{cases}

    or

    .. math::

        p^{\*}=\begin{cases} \frac{1}{(\sum_{i=1}^N w_i)-1}\sum_{i=1}^{N}w_ix_i & \text{if } \sum_{i=1}^Nw_ix_i\ne0\\
                             0 & \text{if } \sum_{i=1}^Nw_ix_i=0 \end{cases}

    if bias correction is used, where
        - :math:`N` is the number of samples in the data set
        - :math:`x_i` is the data of the relevant scope for the `i`-th sample of the data set
        - :math:`w_i` is the weight for the `i`-th sample of the data set

    Weights are normalized to sum up to :math:`N`.

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

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    # Apply NaN strategy
    scope_data, weights = apply_nan_strategy(nan_strategy, scope_data, leaf, weights, check_support)

    # total number of instances
    n_total = weights.sum()

    # total number of trials in data
    scope_data = scope_data + 1  # Shift by one since torch.distributions.Geometric starts counting at 0
    n_trials = (weights * scope_data).sum()

    # avoid division by zero
    p_est = 1e-8 if n_trials == 0 else n_total / n_trials

    if bias_correction:
        p_est = p_est - p_est * (1 - p_est) / n_total

    # edge case: if prob. 1 (or 0), set to smaller (or larger) value
    if torch.isclose(p_est, torch.tensor(0.0)):
        p_est = torch.tensor(1e-8)
    elif torch.isclose(p_est, torch.tensor(1.0)):
        p_est = torch.tensor(1 - 1e-8)

    # set parameters of leaf node
    leaf.p = p_est
