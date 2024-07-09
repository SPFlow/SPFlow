from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterCombinationError, ScopeError
from spflow.meta.data import Scope
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
        inputs: list[Module],
    ) -> None:
        r"""Initializes ``BaseProduct`` object.

        Args:
            inputs:
                List of Modules. The scopes for all child modules need to be pair-wise disjoint.

        Raises:
            ValueError: Invalid arguments.
        """
        super().__init__()

        if not input:
            raise ValueError(f"'{self.__class__.__name__}' requires at least one input to be specified.")

        self.inputs = nn.ModuleList(inputs)

        # Check if all inputs have equal number of features
        if not all(inp.out_features == self.inputs[0].out_features for inp in self.inputs):
            raise ValueError(
                f"Inputs must have equal number of features, but were "
                f"{[inp.out_features for inp in self.inputs]}."
            )

        # Check that scopes are disjoint
        if not Scope.all_pairwise_disjoint([inp.scope for inp in self.inputs]):
            raise ScopeError("Input scopes must be disjoint.")

        # Derive output shape from inputs
        self._out_features = self.inputs[0].out_features

        self._max_out_channels = max(inp.out_channels for inp in self.inputs)

        # Join all scopes
        scope = self.inputs[0].scope
        for inp in self.inputs[1:]:
            scope = scope.join(inp.scope)

        self.scope = scope

    @abstractmethod
    def map_out_channels_to_in_channels(self, output_ids: Tensor) -> Tensor:
        r"""Map output ids to input ids.

        Args:
            output_ids: Output ids.

        Returns:
            Mapped input ids.
        """
        pass

    @abstractmethod
    def map_out_mask_to_in_mask(self, mask: Tensor) -> Tensor:
        r"""Map output mask to input mask.

        Args:
            mask: Output mask.

        Returns:
            Mapped input mask.
        """
        pass

    @property
    def out_features(self) -> int:
        """Returns the number of output features for this module."""
        return self._out_features

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}"


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
    channel_index = module.map_out_channels_to_in_channels(sampling_ctx.channel_index)
    mask = module.map_out_mask_to_in_mask(sampling_ctx.mask)  # TODO: is this correct?

    cid_per_module = []
    mask_per_module = []

    inputs = module.inputs
    for i in range(len(module.inputs)):
        cid_per_module.append(channel_index[..., i])
        mask_per_module.append(mask[..., i])

    # Iterate over inputs, their channel indices and masks
    for inp, cid, mask in zip(inputs, cid_per_module, mask_per_module):
        sampling_ctx.update(channel_index=cid, mask=mask)
        sample(
            inp,
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
) -> list[Tensor]:
    """
    Prepare the input log-likelihoods for the product module.

    Args:
        module: The product module.
        data: The data tensor.
        check_support: Whether to check the support of the input module.
        dispatch_ctx: The dispatch context.
    """
    lls = []
    for inp in module.inputs:
        ll = log_likelihood(
            inp,
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
        )
        lls.append(ll)

    return lls


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
