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
        inputs: Module,
        out_channels: Optional[int] = None,
        num_repetitions: Optional[int] = None,
        weights: Optional[Tensor] = None,
        sum_dim: Optional[int] = 1,
    ) -> None:
        """
        Create a Sum module.

        Args:
            inputs: Single input module. The sum is over the channel dimension of the input.
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

        # Single input, sum over in_channel dimension
        self.inputs = inputs
        self.sum_dim = sum_dim
        self._out_features = self.inputs.out_features
        self._in_channels_total = self.inputs.out_channels
        self._out_channels_total = out_channels
        self.num_repetitions = num_repetitions
        # ToDo: not optional / generalization: additional dimension as tuple
        if num_repetitions is not None:
            self.weights_shape = (self._out_features, self._in_channels_total, self._out_channels_total, self.num_repetitions)
        else:
            self.weights_shape = (self._out_features, self._in_channels_total, self._out_channels_total)
        self.scope = self.inputs.scope
        self.num_repetitions = num_repetitions

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
        marg_input = marginalize(module.inputs, marg_rvs, prune=prune, dispatch_ctx=dispatch_ctx)

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
        return Sum(inputs=marg_input, weights=module_weights)


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

    ll = log_likelihood(
        module.inputs,
        data,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
    )

    ll = ll.unsqueeze(3)  # shape: (B, F, IC, 1)

    log_weights = module.log_weights.unsqueeze(0)  # shape: (1, F, IC, OC)

    #if log_weights.ndim == 5 and log_weights.shape[1] == 1:
    #    log_weights = log_weights - torch.log(torch.tensor(10))


    # Weighted log-likelihoods
    weighted_lls = ll + log_weights  # shape: (B, F, IC, OC)

    # Sum over input channels (sum_dim + 1 since here the batch dimension is the first dimension)
    output = torch.logsumexp(weighted_lls, dim=module.sum_dim + 1)  # shape: (B, F, OC)

    #output = output.view(data.shape[0], module.out_features, module.out_channels)

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
