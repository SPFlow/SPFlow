"""Contains Categorical leaf node for SPFlow in the ``torch`` backend.
"""
from spflow.utils.projections import proj_convex_to_real
from typing import Callable, Optional, Union

import torch
from torch import nn, Tensor
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
from spflow.modules.node.leaf.utils import apply_nan_strategy
from spflow.modules.node.leaf_node import LeafNode


class Categorical(LeafNode):
    def __init__(self, scope: Scope, probs: list[float]) -> None:
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Categorical' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Categorical' should be empty, but was {scope.evidence}.")

        super().__init__(scope=scope)

        # register logits
        self.logits = nn.Parameter(torch.empty(len(probs), dtype=torch.float32))
        self.probs = probs

    @property
    def probs(self) -> Tensor:
        return torch.softmax(self.logits, dim=0)

    @probs.setter
    def probs(self, probs: list[float]) -> None:
        if isinstance(probs, list):
            probs = torch.tensor(probs, device=self.device)
        if probs.ndim != 1:
            raise ValueError(
                f"Numpy array of weight probs for 'Categorical' is expected to be one-dimensional, but is {probs.ndim}-dimensional."
            )
        if not torch.all(probs > 0):
            raise ValueError("Probabilities for 'Categorical' must be all positive.")
        if not torch.isclose(torch.sum(probs), torch.tensor(1.0)):
            raise ValueError("Probabilities for 'Categorical' must sum up to one.")

        values = proj_convex_to_real(probs)
        self.logits.data = values

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Categorical(logits=self.logits)

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Categorical`` can represent a single univariate node with ``CategoricalType`` domain.

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

        # leaf is a discrete Categorical distribution
        # NOTE: only accept instances of 'FeatureTypes.Categorical', otherwise required parameter 'n' is not specified. Reject 'FeatureTypes.Discrete' for the same reason.
        if not isinstance(domains[0], FeatureTypes.Categorical):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Categorical":
        """Creates an instance from a specified signature.

        Returns:
            ``Categorical`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'Categorical' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if isinstance(domain, FeatureTypes.Categorical):
            probs = domain.probs
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Categorical' that was not caught during acception checking."
            )

        return Categorical(feature_ctx.scope, probs=probs)

    def describe_node(self) -> str:
        formatted_probs = [f"{num:.3f}" for num in self.probs.tolist()]
        return f"probs=[{', '.join(formatted_probs)}]"


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: Categorical,
    data: Tensor,
    weights: Optional[Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``Categorical`` node parameters in the ``torch`` backend.

    Estimates the success probabilities :math:`p` of a Categorical distribution from data, as follows:

    .. math::

        TODO

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
            Has no effect for ``Categorical`` nodes.
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

    # handle nans
    scope_data, weights = apply_nan_strategy(nan_strategy, scope_data, leaf, weights, check_support)

    # normalize weights to sum to n_samples
    weights /= weights.sum() / scope_data.shape[0]

    # compute weighted counts
    weighted_counts = torch.bincount(
        scope_data.reshape(-1), weights=weights.reshape(-1), minlength=len(leaf.probs)
    )

    # update parameters
    leaf.probs = weighted_counts / weighted_counts.sum()
