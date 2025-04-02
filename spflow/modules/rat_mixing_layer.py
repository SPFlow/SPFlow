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
    A sum module that represents the sum operation over inputs.

    The sum module can be used to sum over the channel dimension of a single input or over the stacked inputs.
    """

    def __init__(
            self,
            inputs: Module,
            out_channels: Optional[int] = None,
            num_repetitions: Optional[int] = None,
            weights: Optional[Tensor] = None,
            sum_dim: Optional[int] = 3,
    ) -> None:
        """
        Create a Sum module.

        Args:
            inputs: Single input module. The sum is over the channel dimension of the input.
            out_channels: Optional number of output nodes for each sum, if weights are not given.
            weights: Optional weights for the sum module. If not provided, weights will be initialized randomly.
        """
        super().__init__(inputs, out_channels, num_repetitions, weights, sum_dim)


@dispatch(memoize=True)  # type: ignore
def marginalize(
    module: MixingLayer,
    marg_rvs: list[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[None, Sum]:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # compute module scope (same for all outputs)
    module_scope = module.scope
    marg_input = None

    # for idx,s in enumerate(module_scope):
    mutual_rvs = set(module_scope.query).intersection(set(marg_rvs))
    module_weights = module.weights

    # module scope is being fully marginalized over
    if len(mutual_rvs) == len(module_scope.query):
        # passing this loop means marginalizing over the whole scope of this branch
        pass

    # node scope is being partially marginalized
    elif mutual_rvs:
        # marginalize input modules
        marg_input = marginalize(module.inputs, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)



        # if marginalized input is not None
        if marg_input:
            feature_to_scope = module.inputs.feature_to_scope
            # remove mutual_rvs from feature_to_scope list
            for rv in mutual_rvs:
                for idx, scope in enumerate(feature_to_scope):
                    if rv in scope:
                        feature_to_scope[idx] = scope.remove_from_query(rv)

            # construct mask with empty scopes
            mask = [scope.isempty() for scope in feature_to_scope]

            module_weights = module_weights[mask]

    else:
        marg_input = module.inputs

    if marg_input is None:
        return None

    else:
        return MixingLayer(inputs=marg_input, weights=module_weights)


@dispatch  # type: ignore
def sample(
    module: MixingLayer,
    data: Tensor,
    is_mpe: bool = False,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    # initialize contexts
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # Index into the correct weight channels given by parent module
    # (stay in logits space since Categorical distribution accepts logits directly)


    logits = module.logits.unsqueeze(0).expand(sampling_ctx.channel_index.shape[0], -1, -1, -1)
    idxs = sampling_ctx.channel_index[..., None, None]
    in_channels_total = logits.shape[2]
    #idxs = idxs.expand(-1, -1, module._in_channels_total, -1)
    idxs = idxs.expand(-1, -1, in_channels_total, -1)
    logits = logits.gather(dim=3, index=idxs).squeeze(3)

    if (
        "log_likelihood" in dispatch_ctx.cache
        and dispatch_ctx.cache["log_likelihood"][module.inputs] is not None
    ):
        input_lls = dispatch_ctx.cache["log_likelihood"][module.inputs]

        if sampling_ctx.repetition_idx is not None:
            indices = sampling_ctx.repetition_idx.expand(-1, input_lls.shape[1], input_lls.shape[2],-1)
            input_lls = torch.gather(input_lls, dim=-1, index=indices).squeeze(-1)
            log_prior = logits
            log_posterior = log_prior + input_lls#[..., sampling_ctx.repetition_idx]
            log_posterior = log_posterior.log_softmax(dim=2)
            logits = log_posterior
        else:
            log_prior = logits
            log_posterior = log_prior + input_lls
            log_posterior = log_posterior.log_softmax(dim=2)
            logits = log_posterior


    # Sample from categorical distribution defined by weights to obtain indices into input channels
    if is_mpe:
        # Take the argmax of the logits to obtain the most probable index
        sampling_ctx.channel_index = torch.argmax(logits, dim=-1)
    else:
        # Sample from categorical distribution defined by weights to obtain indices into input channels
        sampling_ctx.channel_index = torch.distributions.Categorical(logits=logits).sample()

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

    log_weights = module.log_weights.squeeze(2).unsqueeze(0) # shape: (1, F, IC, R)
    weighted_lls = ll + log_weights # shape: (B, F, input_OC, R) + (1, F, IC, R) = (B, F, OC, R)
    output = torch.logsumexp(weighted_lls, dim=-1) # shape: (B, F, OC)
    return output


@dispatch(memoize=True)  # type: ignore
def em(
    module: MixingLayer,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    with torch.no_grad():
        # ----- expectation step -----

        # Get input LLs
        input_lls = dispatch_ctx.cache["log_likelihood"][
            module.inputs
        ]  # shape: (batch_size, 1, num_scopes, num_nodes_child)

        # Get module lls
        module_lls = dispatch_ctx.cache["log_likelihood"][module]

        log_weights = module.log_weights.unsqueeze(0)
        log_grads = torch.log(module_lls.grad).unsqueeze(2)
        # input_lls = input_lls.unsqueeze(-1)
        input_lls = input_lls.unsqueeze(3)
        module_lls = module_lls.unsqueeze(2)

        log_expectations = log_weights + log_grads + input_lls - module_lls
        log_expectations = log_expectations.logsumexp(0)  # Sum over batch dimension
        log_expectations = log_expectations.log_softmax(module.sum_dim)  # Normalize

        # ----- maximization step -----
        module.log_weights = log_expectations

        # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients
        # TODO: Check if the above is still true after the whole reimplementation (don't we set param.data = ...?)

    em(module.inputs, data, check_support=check_support, dispatch_ctx=dispatch_ctx)


