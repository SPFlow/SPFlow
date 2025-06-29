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


class ElementwiseSum(Module):
    """
    A sum module that the elementwise sum over inputs.

    The sum module can be used to sum over the channel dimension of a single input or over the stacked inputs.
    """

    def __init__(
        self,
        inputs: list[Module],
        out_channels: Optional[int] = None,
        weights: Optional[Tensor] = None,
        num_repetitions: Optional[int] = None,
        sum_dim: int = 3,
    ) -> None:
        """
        Create a Sum module.

        Args:
            inputs: Single input module or list of modules. The sum is over the sum dimension of the input.
            out_channels: Optional number of output nodes for each sum, if weights are not given.
            num_repetitions: Optional number of repetitions for the sum module. If not provided, it will be inferred from the weights.
            weights: Optional weights for the sum module. If not provided, weights will be initialized randomly.
            sum_dim: The dimension over which to sum the inputs. Default is 1 (channel dimension).
        """
        super().__init__()

        if not input:
            raise ValueError("'Sum' requires at least one input to be specified.")

        self.inputs = nn.ModuleList(inputs)

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

        # Save number of sums
        self._num_sums = out_channels
        self.sum_dim = sum_dim

        # Check, that all inputs have the same number of features
        if not all([module.out_features == inputs[0].out_features for module in inputs]):
            raise ValueError("All inputs must have the same number of features.")

        # Check, that all inputs have the same number of channels or 1 channel (broadcast)
        if not all([module.out_channels in (1, max(m.out_channels for m in inputs)) for module in inputs]):
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

        if num_repetitions is None:
            self.num_repetitions = self.inputs[0].num_repetitions
        else:
            self.num_repetitions = num_repetitions

        self._num_inputs = len(inputs)

        # out_channels will be flattened and thus multiplied by the number of inputs
        self._in_channels_per_input = max([module.out_channels for module in self.inputs])
        self._out_channels_total = self._num_sums * self._in_channels_per_input

        if self.num_repetitions is not None:
            self.weights_shape = (
                self._out_features,
                self._in_channels_per_input,
                self._num_sums,
                self._num_inputs,
                self.num_repetitions
            )
        else:
            self.weights_shape = (
                self._out_features,
                self._in_channels_per_input,
                self._num_sums,
                self._num_inputs
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
    def feature_to_scope(self) -> list[Scope]:
        return self.inputs.feature_to_scope

    @property
    def log_weights(self) -> Tensor:
        # project auxiliary weights onto weights that sum up to one
        return torch.nn.functional.log_softmax(self.logits, dim=self.sum_dim)

    @property
    def weights(self) -> Tensor:
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
            values: PyTorch tensor containing weights for each input and node.
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
    module: ElementwiseSum,
    marg_rvs: list[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[None, ElementwiseSum]:
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
        return ElementwiseSum(inputs=[inp for inp in marg_input], weights=module_weights)


@dispatch  # type: ignore
def sample(
    module: ElementwiseSum,
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
    if sampling_ctx.repetition_idx is not None:
        logits = module.logits.unsqueeze(0).expand(
            sampling_ctx.channel_index.shape[0], -1, -1, -1, -1, -1)

        indices = sampling_ctx.repetition_idx  # Shape (30000, 1, 1)

        # Use gather to select the correct repetition
        # Repeat indices to match the target dimension for gathering

        indices = indices.view(-1, 1, 1, 1, 1, 1).expand(-1, logits.shape[1], logits.shape[2], logits.shape[3], logits.shape[4],
                                                      -1)
        logits = torch.gather(logits, dim=-1, index=indices).squeeze(-1)
    else:
        logits = module.logits.unsqueeze(0).expand(sampling_ctx.channel_index.shape[0], -1, -1, -1, -1)
    cids_mapped = module.unraveled_channel_indices[sampling_ctx.channel_index]

    # Take the first element of the tuple (input_channel_idx, output_channel_idx)
    # This is the out_channels index for all inputs in the Stack module
    cids_in_channels_per_input = cids_mapped[..., 0]
    cids_num_sums = cids_mapped[..., 1]

    # Index weights with cids_num_sums (selects the correct output channel)
    cids_num_sums = cids_num_sums[..., None, None, None].expand(
        -1, -1, logits.shape[-3], -1, logits.shape[-1]
    )
    logits = logits.gather(dim=3, index=cids_num_sums).squeeze(3)

    # Index logits with oids_in_channels_per_input to get the correct logits for each input
    logits = logits.gather(
        dim=2, index=cids_in_channels_per_input[..., None, None].expand(-1, -1, -1, logits.shape[-1])
    ).squeeze(2)

    if "log_likelihood" in dispatch_ctx.cache and all(
        dispatch_ctx.cache["log_likelihood"][inp] is not None for inp in module.inputs
    ):
        input_lls = [dispatch_ctx.cache["log_likelihood"][inp] for inp in module.inputs]
        input_lls = torch.stack(input_lls, dim=module.sum_dim)#torch.stack(input_lls, dim=-1)
        if sampling_ctx.repetition_idx is not None:
            indices = sampling_ctx.repetition_idx.view(-1,1,1,1,1).expand(-1, input_lls.shape[1], input_lls.shape[2],input_lls.shape[3],-1)
            input_lls = torch.gather(input_lls, dim=-1, index=indices).squeeze(-1)
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
    module: ElementwiseSum,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

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

    ll = ll.unsqueeze(3)  # shape: (B, F, IC, 1)

    log_weights = module.log_weights.unsqueeze(0)  # shape: (1, F, IC, OC)

    # Weighted log-likelihoods
    weighted_lls = ll + log_weights  # shape: (B, F, IC, OC)

    # Sum over input channels (sum_dim + 1 since here the batch dimension is the first dimension)
    output = torch.logsumexp(weighted_lls, dim=module.sum_dim + 1)
    if module.num_repetitions is not None:
        output = output.view(data.shape[0], module.out_features, module.out_channels, module.num_repetitions)
    else:
        output = output.view(data.shape[0], module.out_features, module.out_channels)

    return output


@dispatch(memoize=True)  # type: ignore
def em(
    module: ElementwiseSum,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    with torch.no_grad():
        # ----- expectation step -----

        # Get input LLs
        input_lls = [dispatch_ctx.cache["log_likelihood"][inp] for inp in module.inputs]
        input_lls = torch.stack(input_lls, dim=3)

        # Get module lls
        module_lls = dispatch_ctx.cache["log_likelihood"][module]

        log_weights = module.log_weights.unsqueeze(0)
        input_lls = input_lls.unsqueeze(3)
        # Get input channel indices
        s = (
            module_lls.shape[0],
            module.out_features,
            module._in_channels_per_input,
            module._num_sums,
            1,
        )
        if module.num_repetitions is not None:
            s = s + (module.num_repetitions,)
        log_grads = torch.log(module_lls.grad).view(s)
        module_lls = module_lls.view(s)

        log_expectations = log_weights + log_grads + input_lls - module_lls
        log_expectations = log_expectations.logsumexp(0)  # Sum over batch dimension
        log_expectations = log_expectations.log_softmax(module.sum_dim)  # Normalize

        # ----- maximization step -----
        module.log_weights = log_expectations

        # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients
        # TODO: Check if the above is still true after the whole reimplementation (don't we set param.data = ...?)

    for inp in module.inputs:
        em(inp, data, check_support=check_support, dispatch_ctx=dispatch_ctx)
