from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterCombinationError, ScopeError
from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.modules.module import Module


class BaseProduct(Module, ABC):
    r"""
    TODO
    """

    def __init__(
        self,
        inputs: Union[Module, tuple[Module, Module], list[Module]],
        split_method: Optional[str] = None,
        split_indices: Optional[
            Union[tuple[list[int], list[int]], torch.IntTensor, tuple[torch.IntTensor, torch.IntTensor]]
        ] = None,
    ) -> None:
        r"""Initializes ``BaseProduct`` object.

        Args:
            inputs:
                Can be either a Module or a list of Modules.
                The scopes for all child modules need to be pair-wise disjoint.

            split_method:
                Method to split the input into two equal-sized subsets.
                Possible values are "split_indices" and "random".
                Defaults to "split_indices".

            split_indices:
                Indices to split the input into two equal-sized subsets.
                If split_method is set to "split_indices", this argument is required.
                Defaults to None.

        Raises:
            ValueError: Invalid arguments.
        """
        super().__init__()
        if isinstance(inputs, list):
            self.inputs = nn.ModuleList(inputs)
        else:
            self.inputs = inputs

        # Save split method
        self.split_method = split_method

        self.has_single_input = not isinstance(inputs, list)

        if self.has_single_input:

            # Check that out_features is even
            if not self.inputs.out_features % 2 == 0:
                raise ValueError(f"Number of input out_features must be even but was {self.inputs.out_features}.")

            if split_method is None:
                raise InvalidParameterCombinationError(
                    "If inputs is a single Module, split_method must be specified."
                )
            elif split_method == "split_indices" and split_indices is None:
                raise InvalidParameterCombinationError(
                    "If split_method is 'split_indices', split_indices must be specified."
                )

            if split_method == "split_indices":
                # Check that splits_indices covers all features
                if not len(split_indices[0]) + len(split_indices[1]) == self.inputs.out_features:
                    raise ValueError(f"Split indices must be the cover range(0, {self.inputs.out_features}).")

                # Convert split indices to torch tensor
                if isinstance(split_indices, tuple):
                    if isinstance(split_indices[0], list):
                        split_indices = torch.stack(
                            [torch.tensor(split_indices[0]), torch.tensor(split_indices[1])], dim=0
                        )
                    elif isinstance(split_indices[0], torch.IntTensor):
                        split_indices = torch.stack(split_indices, dim=0)
                    elif isinstance(split_indices, torch.Tensor):
                        split_indices = split_indices

            elif split_method == "random":
                # Randomly split the input into two equal-sized subsets
                if split_indices is not None:
                    raise InvalidParameterCombinationError(
                        "If split_method is 'random', split_indices must be set to None, but was not."
                    )

                split_indices = torch.randperm(self.inputs.out_features).view(2, -1)
            else:
                raise ValueError(
                    f"Invalid split method. Must be either 'split_indices' or 'random' but was {split_method}."
                )

            # Register indices as buffer
            self.register_buffer("split_indices", split_indices)

            # Invert permutation
            split_indices_inverted = torch.argsort(split_indices.flatten())
            self.register_buffer("split_indices_inverted", split_indices_inverted)

            # Derive output shape from inputs
            self._out_features = self.inputs.out_features // 2
        else:
            if split_method is not None or split_indices is not None:
                raise InvalidParameterCombinationError(
                    "If inputs is a list of Modules, split_method and split_indices must be set to None."
                )

            # Check if inputs have equal number of features
            if self.inputs[0].out_features != self.inputs[1].out_features:
                raise ValueError(
                    f"Inputs must have equal number of features, but were "
                    f"{self.inputs[0].out_features} and {self.inputs[1].out_features}."
                )

            # Check that scopes are disjoint
            if not self.inputs[0].scope.isdisjoint(self.inputs[1].scope):
                raise ScopeError("Input scopes must be disjoint.")

            # Derive output shape from inputs
            self._out_features = self.inputs[0].out_features

        # Obtain scope
        if self.has_single_input:
            self.scope = self.inputs.scope
        else:
            self.scope = self.inputs[0].scope.join(self.inputs[1].scope)

    @abstractmethod
    def map_out_channels_to_in_channels(self, output_ids: Tensor) -> Tensor:
        r"""Map output ids to input ids.

        Args:
            output_ids: Output ids.

        Returns:
            Mapped input ids.
        """
        pass

    @property
    def out_features(self) -> int:
        """Returns the number of output features for this module."""
        return self._out_features

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, split_method={self.split_method}"


@dispatch(memoize=True)  # type: ignore
def marginalize(
    module: BaseProduct,
    marg_rvs: list[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[BaseProduct, Module, None]:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # This is not yet implemented Reasons: Marginalization over the element-product has a couple of challenges - if
    # the input is a single module, we need to ensure, that the splits are still equally sized and we need to
    # eventually update the split_indices which seems non-trivial (mapping from scopes to features where one feature
    # can contain multiple scopes) - if the input are two modules, we need to ensure, that both inputs have the same
    # number of features after marginalization

    raise NotImplementedError("Not implemented yet.")


@dispatch  # type: ignore
def sample(
    module: BaseProduct,
    data: Tensor,
    is_mpe: bool = False,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # Map to (i, j) to index left/right inputs
    oids = module.map_out_channels_to_in_channels(sampling_ctx.output_ids)

    if module.has_single_input:
        sampling_ctx.output_ids = oids.reshape(oids.size(0), module.inputs.out_features)

        # Invert permutation given by split_indices
        sampling_ctx.output_ids = sampling_ctx.output_ids[:, module.split_indices_inverted]

        # Sample from input module
        sample(
            module.inputs,
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=sampling_ctx,
        )
    else:
        # Sample from left
        sampling_ctx.output_ids = oids[:, :, 0]
        sample(
            module.inputs[0],
            data,
            is_mpe=is_mpe,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=sampling_ctx,
        )

        # Sample from right
        sampling_ctx.output_ids = oids[:, :, 1]
        sample(
            module.inputs[1],
            data,
            is_mpe=is_mpe,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=sampling_ctx,
        )

    return data


def _get_input_log_likelihoods(
    module: BaseProduct,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
):
    """
    Prepare the input log-likelihoods for the product module.

    Args:
        module: The product module.
        data: The data tensor.
        check_support: Whether to check the support of the input module.
        dispatch_ctx: The dispatch context.
    """
    if module.has_single_input:
        ll = log_likelihood(
            module.inputs,
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
        )

        # Split the input according to `split_indices` at feature dimension
        ll_left = ll[:, module.split_indices[0]]
        ll_right = ll[:, module.split_indices[1]]
    else:
        ll_left = log_likelihood(
            module.inputs[0],
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
        )
        ll_right = log_likelihood(
            module.inputs[1],
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
        )

    return [ll_left, ll_right]


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: BaseProduct,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    raise NotImplementedError(
        "Not implemented for BaseProduct -- needs to be called on subclasses of BaseProduct."
    )
