#!/usr/bin/env python3

from spflow.meta.dispatch.dispatch import dispatch
from spflow.modules.node.leaf.utils import apply_nan_strategy
from torch import Tensor
from spflow.meta.dispatch.dispatch_context import DispatchContext, init_default_dispatch_context
from typing import Callable, Iterable, Optional, Union
from abc import ABC, abstractmethod

import torch

from spflow.meta.data.scope import Scope
from spflow.modules.module import Module
from spflow.distributions.distribution import Distribution


class LeafModule(Module, ABC):
    def __init__(self, scope: Scope, event_shape: tuple[int, ...] = None):
        r"""Initializes ``Normal`` leaf node.

        Args:
            scope: Scope object specifying the scope of the distribution.
            event_shape: Tuple of integers specifying the shape of the distribution.
                E.g. for a single node this would be (1,), for a vector this would be (n,) and for a layer/matrix
                this would be (n, m).
        """
        super().__init__()
        self.scope = scope.copy()

        # Check if event_shape is a tuple of positive integers
        assert all(e > 0 for e in event_shape), "Event shape must be a tuple of positive integers."
        self.event_shape = event_shape

    @property
    def distribution(self) -> Distribution:
        self._distribution

    @distribution.setter
    def distribution(self, distribution: Distribution):
        self._distribution = distribution


@dispatch(memoize=True)  # type: ignore
def em(
    leaf: LeafModule,
    data: torch.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for the given leaf module.

    Args:
        leaf:
            Leaf module to perform EM step for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
    """
    # TODO: resolve this circular import somehow
    from spflow import maximum_likelihood_estimation

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    with torch.no_grad():
        # ----- expectation step -----

        # get cached log-likelihood gradients w.r.t. module log-likelihoods
        expectations = dispatch_ctx.cache["log_likelihood"][leaf].grad
        # normalize expectations for better numerical stability
        # Reduce expectations to shape [batch_size, 1]
        dims = list(range(1, len(expectations.shape)))
        expectations = expectations.sum(dims)
        expectations /= expectations.sum(keepdim=True)

        # ----- maximization step -----

        # update parameters through maximum weighted likelihood estimation
        maximum_likelihood_estimation(
            leaf,
            data,
            weights=expectations,
            bias_correction=False,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
        )

    # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    leaf: LeafModule,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    r"""Computes log-likelihoods for the leaf module given the data.

    Missing values (i.e., NaN) are marginalized over.

    Args:
        node:
            Leaf to perform inference for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the distribution.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.

    Raises:
        ValueError: Data outside of support.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # get information relevant for the scope
    data = data[:, leaf.scope.query]

    log_prob = torch.zeros_like(data, dtype=torch.float)

    # ----- marginalization -----
    marg_mask = torch.isnan(data)

    # If there are any marg_ids, set them to 0.0 to ensure that distribution.log_prob call is succesfull and doesn't throw errors
    # due to NaNs
    if marg_mask.any():
        data[marg_mask] = 0.0 # ToDo in-support value

    # ----- log probabilities -----

    # Unsqueeze scope_data to make space for num_nodes dimension
    data = data.unsqueeze(2)

    if check_support:
        # create mask based on distribution's support
        valid_mask = leaf.distribution.check_support(data)

        if not torch.all(valid_mask):
            raise ValueError(
                f"Encountered data instances that are not in the support of the distribution."
            )





    # compute probabilities for values inside distribution support
    log_prob = leaf.distribution.log_prob(data.float())

    # Marginalize entries
    log_prob[marg_mask] = 0.0

    # Set marginalized scope data back to NaNs
    if marg_mask.any():
        data[marg_mask] = torch.nan

    return log_prob


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: LeafModule,
    data: Tensor,
    weights: Optional[Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of a leaf module.

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

    # apply NaN strategy
    scope_data, weights = apply_nan_strategy(nan_strategy, scope_data, leaf, weights, check_support)

    # Forward to the actual distribution
    leaf.distribution.maximum_likelihood_estimation(scope_data, weights, bias_correction)
