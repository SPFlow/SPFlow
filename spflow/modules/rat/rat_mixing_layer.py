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
from spflow.utils.projections import (
    proj_convex_to_real,
)
from spflow.modules import Sum


class MixingLayer(Sum):
    """
A mixing layer that sums over the input channels, which is used for the RAT model.
    """

    def __init__(
            self,
            inputs: Module,
            out_channels: Optional[int] = None,
            num_repetitions: Optional[int] = None,
            weights: Optional[Tensor] = None,
            sum_dim: Optional[int] = 1,
    ) -> None:
        """
        Args:
            inputs: Single input module or list of modules. The sum is over the sum dimension of the input.
            out_channels: Optional number of output nodes for each sum, if weights are not given.
            num_repetitions: Optional number of repetitions for the sum module. If not provided, it will be inferred from the weights.
            weights: Optional weights for the sum module. If not provided, weights will be initialized randomly.
            sum_dim: The dimension over which to sum the inputs. Default is 1 (channel dimension).
        """

        super().__init__(inputs, out_channels, num_repetitions, weights, sum_dim)
        if not input:
            raise ValueError("'Sum' requires at least one input to be specified.")

        if weights is not None:
            if out_channels is not None:
                raise InvalidParameterCombinationError(
                    f"Cannot specify both 'out_channels' and 'weights' for 'Sum' module."
                )

            out_channels = weights.shape[2]

        if out_channels < 1:
            raise ValueError(
                f"Number of nodes for 'Sum' must be greater of equal to 1 but was {out_channels}."
            )

        self.inputs = inputs

        # Single input, sum over in_channel dimension
        self.sum_dim = sum_dim
        self._out_features = self.inputs.out_features
        self._out_channels_total = out_channels

        assert out_channels == inputs.out_channels, "out_channels must match the out_channels of the input module."
        assert self._out_features == 1, "MixingLayer represents the first layer of the RatSPN, so it must have a single output feature."

        if num_repetitions is not None:
            self.num_repetitions = num_repetitions
        else:
            raise ValueError(
                "num_repetitions must be specified for 'MixingLayer' module."
            )

        # sum up all repetitions
        self._in_channels = self.num_repetitions

        self.weights_shape = (self._out_features, self._in_channels, self._out_channels_total)

        self.scope = self.inputs.scope

        # If weights are not provided, initialize them randomly
        if weights is None:
            weights = (
                # weights has shape (n_nodes, n_scopes, n_inputs) to prevent permutation at ll and sample
                torch.rand(self.weights_shape)
                + 1e-08
            )  # avoid zeros

            # Normalize
            weights /= torch.sum(weights, axis=self.sum_dim, keepdims=True)

        # Register unnormalized log-probabilities for weights as torch parameters
        self.logits = torch.nn.Parameter()

        # Initialize weights (which sets self.logits under the hood accordingly)
        self.weights = weights

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def out_channels(self) -> int:
        return self._out_channels_total

    @property
    def feature_to_scope(self) -> list[Scope]:
        return self.inputs.feature_to_scope


@dispatch  # type: ignore
def sample(
    module: MixingLayer,
    data: Tensor,
    is_mpe: bool = False,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:

    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    logits = module.logits

    logits = logits.unsqueeze(0).expand(
        sampling_ctx.channel_index.shape[0], -1, -1, -1)  # shape [b , n_features , in_c, out_c]

    if (
            "log_likelihood" in dispatch_ctx.cache
            and dispatch_ctx.cache["log_likelihood"][module.inputs] is not None
    ):
        input_lls = dispatch_ctx.cache["log_likelihood"][module.inputs]

        # Compute log posterior by reweighing logits with input lls
        log_prior = logits
        log_posterior = log_prior + input_lls.unsqueeze(3)
        log_posterior = log_posterior.log_softmax(dim=2)
        logits = log_posterior

    if is_mpe:
        # Take the argmax of the logits to obtain the most probable index
        repetition_idx = torch.argmax(logits.sum(-1), dim=-1).squeeze(-1)
    else:
        # Sample from categorical distribution defined by weights to obtain indices for repetitions
        # sum up the input channel for distribution
        repetition_idx = torch.distributions.Categorical(logits=logits.sum(-1)).sample()

    # get repetition index for the given channels
    # repetition_idx = repetition_idx.gather(dim=1,index=sampling_ctx.channel_index).squeeze()

    sampling_ctx.repetition_idx = repetition_idx

    # Sample from input module
    sample(
        module.inputs,
        data,
        is_mpe=is_mpe,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
        sampling_ctx=sampling_ctx,
    )

    return data


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: MixingLayer,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    ll = log_likelihood(
        module.inputs,
        data,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
    )

    log_weights = module.log_weights.unsqueeze(0)  # shape: (1, F, IC, OC)

    # Weighted log-likelihoods
    weighted_lls = ll.permute(0,1,3,2) + log_weights  # shape: (B, F, R, OC) + (1, F, IC, OC) = (B, F, R = IC, OC)

    # Sum over input channels (sum_dim + 1 since here the batch dimension is the first dimension)
    output = torch.logsumexp(weighted_lls, dim=module.sum_dim + 1) # shape: (B, F, OC, R)

    return output.view(-1, module.out_features, module.out_channels)



