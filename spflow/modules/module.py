"""Contains the abstract ``Module`` class for SPFlow modules in the ``base`` backend.

All valid SPFlow modules in the ``base`` backend should inherit from this class or a subclass of it.
"""

from abc import ABC, abstractmethod
from functools import reduce
from typing import Optional

import torch
from torch import Tensor, nn

from spflow.meta.data.scope import Scope
from spflow.meta.dispatch import (
    SamplingContext,
    dispatch,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.meta.dispatch.sampling_context import init_default_sampling_context


class Module(nn.Module, ABC):
    """Abstract module class for building graph-based models."""

    def __init__(self) -> None:
        """Initializes the module."""
        super().__init__()
        self._scope = Scope()

    @property
    @abstractmethod
    def out_features(self) -> int:
        pass

    @property
    @abstractmethod
    def out_channels(self) -> int:
        pass

    @property
    def scope(self) -> Scope:
        """Returns the scope of the module."""
        return self._scope

    @scope.setter
    def scope(self, scope: Scope):
        """Sets the scope of the module."""
        self._scope = scope

    @property
    def device(self):
        """
        Returns the device of the model. If the model parameters are on different devices,
        it returns the device of the first parameter. If the model has no parameters,
        it returns 'cpu' as the default device.
        """
        return next(iter(self.parameters())).device

    def forward(
        self, data: Tensor, check_support: bool = True, dispatch_ctx: Optional[DispatchContext] = None
    ):
        """Forward pass is simply the log-likelihood function."""
        return log_likelihood(self, data, check_support=check_support, dispatch_ctx=dispatch_ctx)

    def extra_repr(self) -> str:
        return f"D={self.out_features}, C={self.out_channels}"


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: Module,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    """Raises ``NotImplementedError`` for modules in the ``base`` backend that have not dispatched a log-likelihood inference routine.

    Args:
        module:
            Sum node to perform inference for.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional NumPy array containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    raise NotImplementedError(
        f"'log_likelihood' is not defined for modules of type {type(module)}. Check if dispatched functions are correctly declared or imported."
    )


@dispatch  # type: ignore
def likelihood(
    module: Module,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    """Computes likelihoods for modules in the ``base`` backend given input data.

    Likelihoods are per default computed from the infered log-likelihoods of a module.

    Args:
        module:
            Module to perform inference for.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional NumPy array containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return torch.exp(log_likelihood(module, data, check_support=check_support, dispatch_ctx=dispatch_ctx))


@dispatch  # type: ignore
def sample(
    module: Module,
    is_mpe: bool = False,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    r"""Samples from modules in the ``base`` backend without any evidence.

    Samples a single instance from the module.

    Args:
        module:
            Module to sample from.
        is_mpe:
            Boolean value indicating whether to perform maximum a posteriori estimation (MPE).
            Defaults to False.
        check_support:
            Boolean value indicating whether if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
        sampling_ctx:
            Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

    Returns:
        Two-dimensional NumPy array containing the sampled values.
        Each row corresponds to a sample.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return sample(
        module,
        1,
        is_mpe=is_mpe,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
        sampling_ctx=sampling_ctx,
    )


@dispatch  # type: ignore
def sample(
    module: Module,
    num_samples: int = 1,
    is_mpe: bool = False,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    r"""Samples specified numbers of instances from modules in the ``base`` backend without any evidence.

    Samples a specified number of instance from the module by creating an empty two-dimensional NumPy array (i.e., filled with NaN values) of appropriate size and filling it.

    Args:
        module:
            Module to sample from.
        num_samples:
            Number of samples to generate.
        is_mpe:
            Boolean value indicating whether to perform maximum a posteriori estimation (MPE).
            Defaults to False.
        check_support:
            Boolean value indicating whether if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
        sampling_ctx:
            Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

    Returns:
        Two-dimensional NumPy array containing the sampled values.
        Each row corresponds to a sample.
    """
    combined_module_scope = reduce(
        lambda s1, s2: s1.join(s2), module.scopes_out
    )  # TODO: why is scopes_out expected to be in a module but is currently only defined in the node class?

    data = torch.full(
        (num_samples, int(max(combined_module_scope.query) + 1)), torch.nan, device=module.device
    )

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    return sample(
        module,
        data,
        is_mpe=is_mpe,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
        sampling_ctx=sampling_ctx,
    )


@dispatch(memoize=True)  # type: ignore
def em(
    module: Module,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximization (EM) step for this module.

    Args:
        module:
            Layer to perform EM step for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # recursively call EM on inputs

    if isinstance(module.inputs, Module):
        em(module.inputs, data, check_support=check_support, dispatch_ctx=dispatch_ctx)
    else:
        for inp in module.inputs:
            em(inp, data, check_support=check_support, dispatch_ctx=dispatch_ctx)
