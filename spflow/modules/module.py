"""Contains the abstract ``Module`` class for SPFlow modules in the ``base`` backend.

All valid SPFlow modules in the ``base`` backend should inherit from this class or a subclass of it.
"""
from functools import reduce
from abc import ABC
from typing import List, Optional, Tuple, Union

import numpy as np

from spflow.meta.dispatch import dispatch, DispatchContext, init_default_dispatch_context, SamplingContext
from spflow.meta.dispatch.sampling_context import init_default_sampling_context
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
)
from spflow import tensor as T
from spflow.tensor import Tensor
from spflow.tensor.ops import Tensor


class Module(ABC):
    r"""Abstract module class for building graph-based models in the ``base`` backend.

    Attributes:
        children:
            List of modules that are children to the module in a directed graph.
        n_out:
            Integer indicating the number of outputs.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(self, children: Optional[list["Module"]] = None) -> None:
        r"""Initializes ``Module`` object.

        Initializes module by correctly setting its children.

        Args:
            children:
                List of modules that are children to the module.

        Raises:
            ValueError: children of invalid type.
        """
        if children is None:
            children = []

        self.backend = T.get_backend()
        if any(child.backend != self.backend for child in children):
            raise ValueError("Children must all have the same backend as the parent")
        self.children = children

        # as soon as there are more possible backend with gpu compatibility, a backend distinction is necessary
        # TODO: make this torch agnostic (should we make our own device enum with CPU, GPU, ...?)
        self.device = "cpu"

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
            input_ids = T.tensor(input_ids, dtype=T.int32())

        # remember original shape
        shape = T.shape(input_ids)
        # flatten tensor
        input_ids = T.ravel(input_ids)

        # infer number of inputs from children (and their numbers of outputs)
        child_num_outputs = T.tensor([child.n_out for child in self.children])
        child_cum_outputs = T.cumsum(child_num_outputs, -1)

        # get child module for corresponding input
        child_ids = T.sum(child_cum_outputs <= T.reshape(input_ids, (-1, 1)), axis=1)
        # get output id of child module for corresponding input
        output_ids = input_ids - (child_cum_outputs[child_ids] - child_num_outputs[child_ids])

        # restore original shape
        child_ids = T.reshape(child_ids, shape)
        output_ids = T.reshape(output_ids, shape)

        return T.tensor(child_ids, dtype=T.int32()), T.tensor(output_ids, dtype=T.int32())

    def modules(self):
        modules = []
        for child in self.children:
            modules.extend(list(child.modules()))
        modules.insert(0, self)
        return modules

    def parameters(self) -> list[Tensor]:
        parameters = []
        for child in self.children:
            parameters.extend(list(child.parameters()))
        return parameters

    def to(self, dtype=None, device=None):
        if device is not None:
            self.device = device

        for child in self.children:
            child.to(dtype=dtype, device=device)


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
    return T.exp(log_likelihood(module, data, check_support=check_support, dispatch_ctx=dispatch_ctx))


@dispatch  # type: ignore
def sample(
    module: Module,
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
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
        sampling_ctx=sampling_ctx,
    )


@dispatch  # type: ignore
def sample(
    module: Module,
    num_samples: int = 1,
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
    combined_module_scope = reduce(lambda s1, s2: s1.join(s2), module.scopes_out)

    data = T.tensor(
        T.full((num_samples, int(max(combined_module_scope.query) + 1)), T.NAN),
        device=module.device,
    )

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, T.shape(data)[0])

    return sample(
        module,
        data,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
        sampling_ctx=sampling_ctx,
    )
