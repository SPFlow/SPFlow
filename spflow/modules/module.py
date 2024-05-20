"""Contains the abstract ``Module`` class for SPFlow modules in the ``base`` backend.

All valid SPFlow modules in the ``base`` backend should inherit from this class or a subclass of it.
"""
from abc import ABC, abstractmethod
from functools import reduce
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, Tensor

from spflow.meta.dispatch import (
    DispatchContext,
    SamplingContext,
    dispatch,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.meta.dispatch.sampling_context import init_default_sampling_context
from spflow.meta.data.scope import Scope


class Module(nn.Module, ABC):
    r"""Abstract module class for building graph-based models in the ``base`` backend.

    Attributes:
        inputs:
            List of modules that are inputs to the module in a directed graph.
        n_out:
            Integer indicating the number of outputs.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(self, inputs: Optional[list["Module"]] = None) -> None:
        r"""Initializes ``Module`` object.

        Initializes module by correctly setting its inputs.

        Args:
            inputs:
                List of modules that are inputs to the module.

        Raises:
            ValueError: inputs of invalid type.
        """
        super().__init__()
        if inputs is None:
            inputs = []

        self.inputs = nn.ModuleList(inputs)

    @property
    @abstractmethod
    def n_out(self) -> int:
        """Returns the number of outputs for this module."""
        pass

    @property
    # @abstractmethod
    def scope(self) -> Scope:
        """Returns the scope of the module."""
        return self._scope

    @scope.setter
    def scope(self, scope: Scope):
        """Sets the scope of the module."""
        self._scope = scope

    @property
    def scopes_out(self) -> list[Scope]:
        """Returns the output scopes this module represents."""
        return [self.scope]

    @property
    def device(self):
        """
        Returns the device of the model. If the model parameters are on different devices,
        it returns the device of the first parameter. If the model has no parameters,
        it returns 'cpu' as the default device.
        """
        return next(iter(self.parameters())).device

    def input_to_output_ids(self, input_ids: Union[list[int], Tensor]) -> tuple[list[int], list[int]]:
        """Translates input indices into corresponding child module indices and child module output indices.

        For a given sequence of input indices (taking the inputs of all child modules into account), computes
        the corresponding child module indices and child module output indices.

        Args:
            input_ids:
                List of integers or one-dimensional NumPy array of integers specifying input indices to the module.

        Returns:
            A tuple of two lists of integers. The first list contains indices of child modules and the
            second list contains the corresponding output indices of the respective child modules.
        """
        if len(input_ids) == 0:
            input_ids = list(range(self.n_out))

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int)

        # remember original shape
        shape = input_ids.shape
        # flatten tensor
        input_ids = torch.ravel(input_ids)

        # infer number of inputs from inputs (and their numbers of outputs)
        child_num_outputs = torch.tensor([child.n_out for child in self.inputs])
        child_cum_outputs = torch.cumsum(child_num_outputs, -1)

        # get child module for corresponding input
        child_ids = torch.sum(child_cum_outputs <= input_ids.view(-1, 1), dim=1)
        # get output id of child module for corresponding input
        output_ids = input_ids - (child_cum_outputs[child_ids] - child_num_outputs[child_ids])

        # restore original shape
        child_ids = child_ids.view(shape)
        output_ids = output_ids.view(shape)

        return child_ids, output_ids

    def modules(self):
        modules = []
        for child in self.inputs:
            modules.extend(list(child.modules()))
        modules.insert(0, self)
        return modules

    def forward(
        self, data: Tensor, check_support: bool = True, dispatch_ctx: Optional[DispatchContext] = None
    ):
        """Forward pass is simply the log-likelihood function."""
        return log_likelihood(self, data, check_support=check_support, dispatch_ctx=dispatch_ctx)


@dispatch(memoize=True)  # type: ignore
def toNodeBased(mod: Module, dispatch_ctx: Optional[DispatchContext] = None):
    return mod


@dispatch(memoize=True)  # type: ignore
def toLayerBased(mod: Module, dispatch_ctx: Optional[DispatchContext] = None):
    return mod


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: Module,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    """Raises ``NotImplementedError`` for modules in the ``base`` backend that have not dispatched a log-likelihood inference routine.

    Args:
        sum_node:
            Sum node to perform inference for.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
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
        modules:
            Module to perform inference for.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
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
        data:
            Two-dimensional NumPy array containing potential evidence.
            Each row corresponds to a sample.
        is_mpe:
            Boolean value indicating whether or not to perform maximum a posteriori estimation (MPE).
            Defaults to False.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
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
        data:
            Two-dimensional NumPy array containing potential evidence.
            Each row corresponds to a sample.
        is_mpe:
            Boolean value indicating whether or not to perform maximum a posteriori estimation (MPE).
            Defaults to False.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
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
    layer: Module,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``ProductLayer`` in the ``torch`` backend.

    Args:
        layer:
            Layer to perform EM step for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # recursively call EM on inputs

    em(layer.inputs[0], data, check_support=check_support, dispatch_ctx=dispatch_ctx)