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


class Sum(Module):
    """
    A sum module that represents the sum operation over inputs.

    The sum module can be used to sum over the channel dimension of a single input or over the stacked inputs.
    """

    def __init__(
        self,
        inputs: Union[Module, list[Module]],
        out_channels: Optional[int] = None,
        weights: Optional[Tensor] = None,
    ) -> None:
        """
        Create a Sum module.

        Args:
            inputs: Can be a single input or a list of inputs.
                - If a single input is provided, the sum module will sum over the channel dimension of the input.
                - If a list of inputs is provided, the sum will be performed over the stacked inputs.
            out_channels: Optional number of output nodes for each sum, if weights are not given.
            weights: Optional weights for the sum module. If not provided, weights will be initialized randomly.
        """
        super().__init__()

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

        # Flag to allow conditional code based on single/multiple inputs
        self.has_single_input = not isinstance(inputs, list)

        if self.has_single_input:
            # Single input, sum over in_channel dimension
            self.inputs = inputs
            self.sum_dim = 1
            self._out_features = self.inputs.out_features
            self._in_channels_total = self.inputs.out_channels
            self._out_channels_total = out_channels
            self.weights_shape = (self._out_features, self._in_channels_total, self._out_channels_total)
            self.scope = self.inputs.scope
        else:
            # Check, that all inputs have the same number of features
            if not all([module.out_features == inputs[0].out_features for module in inputs]):
                raise ValueError("All inputs must have the same number of features.")

            # Check, that all inputs have the same number of channels or 1 channel (broadcast)
            if not all(
                [module.out_channels in (1, max(m.out_channels for m in inputs)) for module in inputs]
            ):
                raise ValueError(
                    "All inputs must have the same number of channels or 1 channel (in which case the "
                    "operation is broadcast)."
                )

            # Check that all input modules have the same scope
            if not Scope.all_equal([module.scope for module in inputs]):
                raise ScopeError("All input modules must have the same scope.")

            self.scope = inputs[0].scope

            # Multiple inputs, stack and sum over stacked dimension
            self.inputs = nn.ModuleList(inputs)
            self._out_features = self.inputs[0].out_features
            self._num_sums = out_channels

            # out_channels will be flattened and thus multiplied by the number of inputs
            self._in_channels_per_input = max([module.out_channels for module in self.inputs])
            self._out_channels_total = self._num_sums * self._in_channels_per_input
            self.sum_dim = 3
            self.weights_shape = (
                self._out_features,
                self._in_channels_per_input,
                self._num_sums,
                len(inputs),
            )

            # Store unraveled in- and out-channel indices
            # E.g. for 2 inputs with 3 in_channels_per_input, the mapping should be:
            # [   0  ,    1  ,    2  ,    3  ,    4  ,    5  ]
            #     |       |       |       |       |       |
            #     v       v       v       v       v       v
            # [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
            #
            # This is necessary to map the output indices to the correct input indices
            unraveled_channel_indices = torch.tensor(
                [(i, j) for i in range(self._in_channels_per_input) for j in range(self._num_sums)]
            )
            self.register_buffer(name="unraveled_channel_indices", tensor=unraveled_channel_indices)

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
    def log_weights(self) -> Tensor:
        """Returns the weights of all nodes as a two-dimensional PyTorch tensor."""
        # project auxiliary weights onto weights that sum up to one
        return torch.nn.functional.log_softmax(self.logits, dim=self.sum_dim)

    @property
    def weights(self) -> Tensor:
        """Returns the weights of all nodes as a two-dimensional PyTorch tensor."""
        # project auxiliary weights onto weights that sum up to one
        return torch.nn.functional.softmax(self.logits, dim=self.sum_dim)

    @weights.setter
    def weights(
        self,
        values: Tensor,
    ) -> None:
        """
        Set weights of all nodes.

        Args:
            values: Three-dimensional PyTorch tensor containing weights for each input and node.
        """
        if values.shape != self.weights_shape:
            raise ValueError(f"Invalid shape for weights: {values.shape}.")
        if not torch.all(values > 0):
            raise ValueError("Weights for 'Sum' must be all positive.")
        if not torch.allclose(torch.sum(values, dim=self.sum_dim), torch.tensor(1.0)):
            raise ValueError("Weights for 'Sum' must sum up to one.")
        self.logits.data = proj_convex_to_real(values)

    @log_weights.setter
    def log_weights(
        self,
        values: Tensor,
    ) -> None:
        """
        Set weights of all nodes.

        Args:
            values: Three-dimensional PyTorch tensor containing weights for each input and node.
        """
        if values.shape != self.log_weights.shape:
            raise ValueError(f"Invalid shape for weights: {values.shape}.")
        self.logits.data = values

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, weights={self.weights_shape}"


@dispatch(memoize=True)  # type: ignore
def marginalize(
    module: Sum,
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
        if module.has_single_input:
            marg_input = marginalize(module.inputs, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)
        else:
            marg_input = [
                marginalize(inp, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx) for inp in module.inputs
            ]

            if all(mi is None for mi in marg_input):
                marg_input = None

        # if marginalized input is not None
        if marg_input:
            indices = [module.scope.query.index(el) for el in list(mutual_rvs)]
            mask = torch.ones_like(torch.tensor(module_scope.query), dtype=torch.bool)
            mask[indices] = False
            module_weights = module_weights[mask]

    else:
        marg_input = module.inputs

    if marg_input is None:
        return None

    else:
        if module.has_single_input:
            return Sum(inputs=marg_input, weights=module_weights)
        else:
            return Sum(inputs=[inp for inp in marg_input], weights=module_weights)


@dispatch  # type: ignore
def sample(
    module: Sum,
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
    if module.has_single_input:
        logits = module.logits.unsqueeze(0).expand(sampling_ctx.channel_index.shape[0], -1, -1, -1)
        idxs = sampling_ctx.channel_index[..., None, None]
        idxs = idxs.expand(-1, -1, module._in_channels_total, -1)
        logits = logits.gather(dim=3, index=idxs).squeeze(3)

        if (
            "log_likelihood" in dispatch_ctx.cache
            and dispatch_ctx.cache["log_likelihood"][module.inputs] is not None
        ):
            input_lls = dispatch_ctx.cache["log_likelihood"][module.inputs]

            # Compute log posterior by reweighing logits with input lls
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
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
            sampling_ctx=sampling_ctx,
        )

    else:
        logits = module.logits.unsqueeze(0).expand(sampling_ctx.channel_index.shape[0], -1, -1, -1, -1)
        oids_mapped = module.unraveled_channel_indices[sampling_ctx.channel_index]

        # Take the first element of the tuple (input_channel_idx, output_channel_idx)
        # This is the out_channels index for all inputs in the Stack module
        cids_in_channels_per_input = oids_mapped[..., 0]
        oids_num_sums = oids_mapped[..., 1]

        # Index weights with oids_num_sums (selects the correct output channel)
        oids_num_sums = oids_num_sums[..., None, None, None].expand(
            -1, -1, logits.shape[-3], -1, logits.shape[-1]
        )
        logits = logits.gather(dim=3, index=oids_num_sums).squeeze(3)

        # Index logits with oids_in_channels_per_input to get the correct logits for each input
        logits = logits.gather(
            dim=2, index=cids_in_channels_per_input[..., None, None].expand(-1, -1, -1, logits.shape[-1])
        ).squeeze(2)

        if module.has_single_input:
            if (
                "log_likelihood" in dispatch_ctx.cache
                and dispatch_ctx.cache["log_likelihood"][module.inputs] is not None
            ):
                input_lls = dispatch_ctx.cache["log_likelihood"][module.inputs]
                is_conditional = True
            else:
                is_conditional = False

        else:
            if "log_likelihood" in dispatch_ctx.cache and all(
                dispatch_ctx.cache["log_likelihood"][inp] is not None for inp in module.inputs
            ):
                input_lls = [dispatch_ctx.cache["log_likelihood"][inp] for inp in module.inputs]
                input_lls = torch.stack(input_lls, dim=-1)
                is_conditional = True
            else:
                is_conditional = False

        if is_conditional:
            cids_in_channels_input_lls = (
                cids_in_channels_per_input.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, input_lls.shape[3])
            )
            input_lls = input_lls.gather(dim=2, index=cids_in_channels_input_lls).squeeze(2)

            # Compute log posterior by reweighing logits with input lls
            log_prior = logits
            log_posterior = log_prior + input_lls
            log_posterior = log_posterior.log_softmax(dim=2)
            logits = log_posterior

        # Sample/MPE from categorical distribution defined by weights to obtain indices into the Stack dimension
        if is_mpe:
            cids_stack = torch.argmax(logits, dim=-1)
        else:
            cids_stack = torch.distributions.Categorical(logits=logits).sample()

        # Sample from input module
        sampling_ctx.channel_index = cids_in_channels_per_input

        for i, inp in enumerate(module.inputs):
            # Update feature_mask
            mask = sampling_ctx.mask & (cids_stack == i)

            sampling_ctx_cpy = sampling_ctx.copy()
            sampling_ctx_cpy.mask = mask

            # Sample from input module
            sample(
                inp,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
                sampling_ctx=sampling_ctx_cpy,
            )

    return data


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    module: Sum,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    if module.has_single_input:
        ll = log_likelihood(
            module.inputs,
            data,
            check_support=check_support,
            dispatch_ctx=dispatch_ctx,
        )
    else:
        # Get input log-likelihoods
        lls = []
        for inp in module.inputs:
            ll = log_likelihood(
                inp,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )

            # Prepare for broadcasting
            if inp.out_channels == 1 and module._in_channels_per_input > 1:
                ll = ll.expand(data.shape[0], module.out_features, module._in_channels_per_input)

            lls.append(ll)

        # Stack input log-likelihoods
        ll = torch.stack(lls, dim=module.sum_dim)

    # if module.has_single_input:
    ll = ll.unsqueeze(3)  # shape: (B, F, IC, 1)

    log_weights = module.log_weights.unsqueeze(0)  # shape: (1, F, IC, OC)

    # Weighted log-likelihoods
    weighted_lls = ll + log_weights  # shape: (B, F, IC, OC)

    # Sum over input channels (sum_dim + 1 since here the batch dimension is the first dimension)
    output = torch.logsumexp(weighted_lls, dim=module.sum_dim + 1)  # shape: (B, F, OC)

    output = output.view(data.shape[0], module.out_features, module.out_channels)

    return output


@dispatch(memoize=True)  # type: ignore
def em(
    module: Sum,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    with torch.no_grad():
        # ----- expectation step -----

        # Get input LLs
        if module.has_single_input:
            input_lls = dispatch_ctx.cache["log_likelihood"][
                module.inputs
            ]  # shape: (batch_size, 1, num_scopes, num_nodes_child)
        else:
            input_lls = [dispatch_ctx.cache["log_likelihood"][inp] for inp in module.inputs]
            input_lls = torch.stack(input_lls, dim=3)

        # Get module lls
        module_lls = dispatch_ctx.cache["log_likelihood"][module]

        if module.has_single_input:
            log_weights = module.log_weights.unsqueeze(0)
            log_grads = torch.log(module_lls.grad).unsqueeze(2)
            input_lls = input_lls.unsqueeze(-1)
            module_lls = module_lls.unsqueeze(2)
        else:
            log_weights = module.log_weights.unsqueeze(0)
            input_lls = input_lls.unsqueeze(3)

            s = (
                module_lls.shape[0],
                module.out_features,
                module._in_channels_per_input,
                module._num_sums,
                1,
            )
            log_grads = torch.log(module_lls.grad).view(s)
            module_lls = module_lls.view(s)

        log_expectations = log_weights + log_grads + input_lls - module_lls
        log_expectations = log_expectations.logsumexp(0)  # Sum over batch dimension
        log_expectations = log_expectations.log_softmax(module.sum_dim)  # Normalize

        # ----- maximization step -----
        module.log_weights = log_expectations

        # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients
        # TODO: Check if the above is still true after the whole reimplementation (don't we set param.data = ...?)

    if module.has_single_input:
        em(module.inputs, data, check_support=check_support, dispatch_ctx=dispatch_ctx)
    else:
        for inp in module.inputs:
            em(inp, data, check_support=check_support, dispatch_ctx=dispatch_ctx)
