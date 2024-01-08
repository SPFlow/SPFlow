"""Contains the abstract ``NestedModule`` class for SPFlow modules in the ``base`` backend.
"""
from abc import ABC
from typing import List, Optional, Tuple, Union

import numpy as np
from spflow import tensor as T

from spflow.meta.dispatch import (
    dispatch,
    SamplingContext,
    init_default_dispatch_context,
    init_default_sampling_context,
)
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.tensor.ops import Tensor
from spflow.modules.module import Module


class NestedModule(Module, ABC):
    """Convenient abstract module class for modules in the ``base`` backend that nest non-terminal modules.

    Attributes:
        children:
            List of modules that are children to the module in a directed graph.
        n_out:
            Integer indicating the number of outputs.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(self, children: Optional[list[Module]] = None, **kwargs) -> None:
        """Initializes ``NestedModule`` object.

        Initializes module by correctly setting its children.

        Args:
            children:
                List of modules that are children to the module.
        """
        if children is None:
            children = []

        super().__init__(children=children, **kwargs)
        self.placeholders = []

    def create_placeholder(self, input_ids: list[int]) -> "Placeholder":
        """Creates an internal placeholder module.

        Creates and registers a placeholder module that can be used for internal non-terminal modules.

        Args:
            input_ids:
                List of integers specifying the input indices of the host module that this placeholder should represent internally.

        Returns:
            'Placeholder' object (to pass as child to internal non-terminal modules).
        """
        # create and register placeholder
        ph = self.Placeholder(host=self, input_ids=input_ids)
        self.placeholders.append(ph)

        return ph

    def set_placeholders(
        self,
        f_name: str,
        inputs: Tensor,
        dispatch_ctx: DispatchContext,
        overwrite=True,
    ) -> None:
        """Fills the cache for all registered placeholder modules given a function name and specified input values.

        Args:
            f_name:
                String of the function name to set the cache of the placeholders for.
            inputs:
                NumPy array of all inputs. Inputs to be cached are selected based on input indices the placeholders represent.
            dispatch_ctx:
                Dispatch context to use cache of.
            overwrite:
                Boolean indicating whether or not to overwrite potentially existing cached values.
        """
        for ph in self.placeholders:
            # fill placeholder cache with specified input values
            dispatch_ctx.cache_value(f_name, ph, inputs[:, ph.input_ids], overwrite=overwrite)

    class Placeholder(Module):
        """Placeholder module as an intermediary module between nested non-terminal modules and actual child modules in the ``base`` backend.

        Since all non-terminal modules need their children to be specified at creation, internal non-terminal modules would
        have to have the same children as the outer host module. This is not ideal, therefore placeholders can be used instead
        that simply act as mediators between the actual host module's children and the internal non-terminal modules.
        Furthermore, placeholders can be used to select parts of modules' outputs and use them internally in arbirary ways.

        Attributes:
            n_out:
                Integer indicating the number of outputs (equal to the number of inputs it represents).
            scopes_out:
                List of scopes representing the output scopes (equal to the scopes of the inputs it represents).
        """

        def __init__(self, host: Module, input_ids: list[int]) -> None:
            """Initializes ``Placeholder`` object.

            Initializes module by correctly setting its children.

            Args:
                host:
                    Host module, that the placeholder is part of.
                input_ids:
                    List of integers specifying the input indices of the host module that this placeholder should represent internally.
            """
            super().__init__()

            self.host = host
            self.input_ids = input_ids

            (
                self.child_ids_actual,
                self.output_ids_actual,
            ) = self.input_to_output_ids(list(range(len(input_ids))))

            # get child scopes
            child_scopes = sum([child.scopes_out for child in host.chs], [])

            # compute scope for placeholder
            self.scopes_out = [child_scopes[i] for i in input_ids]

        def input_to_output_ids(self, input_ids: Union[list[int], Tensor]) -> tuple[list[int], list[int]]:
            """Translates input indices to the host module into corresponding child module indices and child module output indices.

            For a given sequence of input indices to the host module (taking the inputs of all child modules into account), computes
            the corresponding child module indices and child module output indices.

            Args:
                input_ids:
                    List of integers or one-dimensional NumPy array of integers specifying input indices to the module.

            Returns:
                A tuple of two lists of integers. The first list contains indices of child modules and the
                second list contains the corresponding output indices of the respective child modules.
            """
            if len(input_ids) == 0:
                input_ids = list(range(len(self.input_ids)))

            return self.host.input_to_output_ids([self.input_ids[i] for i in input_ids])

        @property
        def n_out(self) -> int:
            """Returns the number of output for this module."""
            return len(self.input_ids)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    nesting_module: NestedModule.Placeholder,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    """Raises ``LookupError`` for placeholder-modules in the ``base`` backend.

    The log-likelihoods for placeholder-modules should be set in the dispatch context cache by the host module.
    This method is only called if the cache is not filled properly, due to memoization.

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
    raise LookupError(
        "Log-likelihood values for 'NestedModule.Placeholder' must not have been found in dispatch cache. Check if these are correctly set by the host module."
    )


@dispatch  # type: ignore
def sample(
    placeholder: NestedModule.Placeholder,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    r"""Samples from a placeholder modules in the ``base`` with potential evidence.

    Samples from the actual inputs represented by the placeholder module.

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
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, T.shape(data)[0])

    # dictionary to hold the
    sampling_ids_per_child = [([], []) for _ in placeholder.host.children]

    for instance_id, output_ids in zip(sampling_ctx.instance_ids, sampling_ctx.output_ids):
        # convert ids to actual child and output ids of host module
        child_ids_actual, output_ids_actual = placeholder.input_to_output_ids(output_ids)

        # for child_id in T.unique(child_ids_actual):
        for child_id in np.unique(child_ids_actual):
            # child_id = T.tensor(child_id, dtype=int)
            sampling_ids_per_child[child_id][0].append(instance_id)
            # sampling_ids_per_child[child_id][1].append(
            #    T.tensor(output_ids_actual, dtype=int)[child_ids_actual == child_id].tolist()
            # )
            sampling_ids_per_child[child_id][1].append(
                np.array(output_ids_actual)[child_ids_actual == child_id].tolist()
            )

    # sample from children
    for child_id, (instance_ids, output_ids) in enumerate(sampling_ids_per_child):
        if len(instance_ids) == 0:
            continue
        sample(
            placeholder.host.children[child_id],
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=SamplingContext(instance_ids, output_ids),
        )

    return data
