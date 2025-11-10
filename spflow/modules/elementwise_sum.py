from __future__ import annotations

from typing import Optional, Dict, Any

import torch
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterCombinationError, ScopeError
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext, init_default_sampling_context
from spflow.modules.module import Module
from spflow.utils.cache import Cache, init_cache
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
        out_channels: int | None = None,
        weights: Tensor | None = None,
        num_repetitions: int | None = None,
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

        if not inputs:
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
                self.num_repetitions,
            )
        else:
            self.weights_shape = (
                self._out_features,
                self._in_channels_per_input,
                self._num_sums,
                self._num_inputs,
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
            weights /= torch.sum(weights, dim=self.sum_dim, keepdim=True)

        # Register unnormalized log-probabilities for weights as torch parameters
        self.logits = torch.nn.Parameter(torch.zeros(self.weights_shape))

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
        return self.inputs[0].feature_to_scope

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

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Optional["ElementwiseSum"]:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: List of random variables to marginalize over.
            prune: Whether to prune the structure.
            cache: Optional cache dictionary.

        Returns:
            The marginalized module or None if fully marginalized.
        """
        # initialize cache
        cache = init_cache(cache)

        # compute module scope (same for all outputs)
        module_scope = self.scope
        marg_input = None

        mutual_rvs = set(module_scope.query).intersection(set(marg_rvs))
        module_weights = self.weights

        # module scope is being fully marginalized over
        if len(mutual_rvs) == len(module_scope.query):
            # passing this loop means marginalizing over the whole scope of this branch
            pass

        # node scope is being partially marginalized
        elif mutual_rvs:
            # marginalize input modules
            marg_input = [inp.marginalize(marg_rvs, prune=prune, cache=cache) for inp in self.inputs]

            if all(mi is None for mi in marg_input):
                marg_input = None

            # if marginalized input is not None
            if marg_input:
                indices = [self.scope.query.index(el) for el in list(mutual_rvs)]
                mask = torch.ones_like(torch.tensor(module_scope.query), dtype=torch.bool)
                mask[indices] = False
                module_weights = module_weights[mask]

        else:
            marg_input = self.inputs

        if marg_input is None:
            return None

        else:
            return ElementwiseSum(inputs=[inp for inp in marg_input], weights=module_weights)

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        check_support: bool = True,
        cache: Cache | None = None,
        sampling_ctx: Optional[SamplingContext] = None,
    ) -> Tensor:
        """Generate samples from the elementwise sum.

        Args:
            num_samples: Number of samples to generate.
            data: The data tensor to populate with samples.
            is_mpe: Whether to use maximum probability estimation instead of sampling.
            check_support: Whether to check the support of the input module.
            cache: Optional cache dictionary for intermediate results.
            sampling_ctx: Optional sampling context.

        Returns:
            The data tensor populated with samples.
        """
        # Prepare data tensor
        data = self._prepare_sample_data(num_samples, data)

        # initialize contexts
        cache = init_cache(cache)
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

        # Index into the correct weight channels given by parent module
        # (stay in logits space since Categorical distribution accepts logits directly)
        if sampling_ctx.repetition_idx is not None:
            logits = self.logits.unsqueeze(0).expand(sampling_ctx.channel_index.shape[0], -1, -1, -1, -1, -1)

            indices = sampling_ctx.repetition_idx  # Shape (30000, 1, 1)

            # Use gather to select the correct repetition
            # Repeat indices to match the target dimension for gathering

            indices = indices.view(-1, 1, 1, 1, 1, 1).expand(
                -1, logits.shape[1], logits.shape[2], logits.shape[3], logits.shape[4], -1
            )
            logits = torch.gather(logits, dim=-1, index=indices).squeeze(-1)
        else:
            logits = self.logits.unsqueeze(0).expand(sampling_ctx.channel_index.shape[0], -1, -1, -1, -1)
        cids_mapped = self.unraveled_channel_indices[sampling_ctx.channel_index]

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

        if "log_likelihood" in cache and all(cache["log_likelihood"][inp] is not None for inp in self.inputs):
            input_lls = [cache["log_likelihood"][inp] for inp in self.inputs]
            input_lls = torch.stack(input_lls, dim=self.sum_dim)  # torch.stack(input_lls, dim=-1)
            if sampling_ctx.repetition_idx is not None:
                indices = sampling_ctx.repetition_idx.view(-1, 1, 1, 1, 1).expand(
                    -1, input_lls.shape[1], input_lls.shape[2], input_lls.shape[3], -1
                )
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

        for i, inp in enumerate(self.inputs):
            # Update feature_mask
            mask = sampling_ctx.mask & (cids_stack == i)

            sampling_ctx_cpy = sampling_ctx.copy()
            sampling_ctx_cpy.mask = mask

            # Sample from input module
            inp.sample(
                data=data,
                is_mpe=is_mpe,
                check_support=check_support,
                cache=cache,
                sampling_ctx=sampling_ctx_cpy,
            )

        return data

    def log_likelihood(
        self,
        data: Tensor,
        check_support: bool = True,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log P(data | module) for elementwise sum.

        Args:
            data: The data tensor.
            check_support: Whether to check the support of the module.
            cache: Optional cache dictionary.

        Returns:
            Log likelihood tensor.
        """
        # initialize cache
        cache = init_cache(cache)
        log_cache = cache.setdefault("log_likelihood", {})

        # Get input log-likelihoods
        lls = []
        for inp in self.inputs:
            ll = inp.log_likelihood(
                data,
                check_support=check_support,
                cache=cache,
            )

            # Prepare for broadcasting
            if inp.out_channels == 1 and self._in_channels_per_input > 1:
                ll = ll.expand(data.shape[0], self.out_features, self._in_channels_per_input)

            log_cache[inp] = ll
            lls.append(ll)

        # Stack input log-likelihoods
        stacked_lls = torch.stack(lls, dim=self.sum_dim)

        ll = stacked_lls.unsqueeze(3)  # shape: (B, F, IC, 1)

        log_weights = self.log_weights.unsqueeze(0)  # shape: (1, F, IC, OC)

        # Weighted log-likelihoods
        weighted_lls = ll + log_weights  # shape: (B, F, IC, OC)

        # Sum over input channels (sum_dim + 1 since here the batch dimension is the first dimension)
        output = torch.logsumexp(weighted_lls, dim=self.sum_dim + 1)
        if self.num_repetitions is not None:
            output = output.view(data.shape[0], self.out_features, self.out_channels, self.num_repetitions)
        else:
            output = output.view(data.shape[0], self.out_features, self.out_channels)

        log_cache[self] = output
        return output

    def expectation_maximization(
        self,
        data: Tensor,
        check_support: bool = True,
        cache: Cache | None = None,
    ) -> None:
        """Expectation-maximization step.

        Args:
            data: The data tensor.
            check_support: Whether to check the support of the module.
            cache: Optional cache dictionary.
        """
        # initialize cache
        cache = init_cache(cache)

        with torch.no_grad():
            # ----- expectation step -----

            # Get input LLs
            input_lls = [cache["log_likelihood"][inp] for inp in self.inputs]
            input_lls = torch.stack(input_lls, dim=3)

            # Get module lls
            module_lls = cache["log_likelihood"][self]

            log_weights = self.log_weights.unsqueeze(0)
            input_lls = input_lls.unsqueeze(3)
            # Get input channel indices
            s = (
                module_lls.shape[0],
                self.out_features,
                self._in_channels_per_input,
                self._num_sums,
                1,
            )
            if self.num_repetitions is not None:
                s = s + (self.num_repetitions,)
            log_grads = torch.log(module_lls.grad).view(s)
            module_lls = module_lls.view(s)

            log_expectations = log_weights + log_grads + input_lls - module_lls
            log_expectations = log_expectations.logsumexp(0)  # Sum over batch dimension
            log_expectations = log_expectations.log_softmax(self.sum_dim)  # Normalize

            # ----- maximization step -----
            self.log_weights = log_expectations

            # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients
            # TODO: Check if the above is still true after the whole reimplementation (don't we set param.data = ...?)

        for inp in self.inputs:
            inp.expectation_maximization(data, check_support=check_support, cache=cache)

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Optional[Tensor] = None,
        check_support: bool = True,
        cache: Cache | None = None,
    ) -> None:
        """Update parameters via maximum likelihood estimation.

        For ElementwiseSum modules, this is equivalent to EM.

        Args:
            data: Input data tensor.
            weights: Optional sample weights (currently unused).
            check_support: Whether to check data support.
            cache: Optional cache dictionary.
        """
        self.expectation_maximization(data, check_support=check_support, cache=cache)
