from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterCombinationError, ScopeError
from spflow.meta.data import Scope
from spflow.modules.base import Module
from spflow.utils.cache import Cache, cached
from spflow.utils.projections import (
    proj_convex_to_real,
)
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


class ElementwiseSum(Module):
    """Elementwise sum operation for mixture modeling.

    Computes weighted combinations of input tensors element-wise. Weights
    are automatically normalized to sum to one. Uses log-domain computations.

    Attributes:
        logits (Parameter): Unnormalized log-weights for gradient optimization.
        unraveled_channel_indices (Tensor): Mapping for flattened channel indices.
    """

    def __init__(
        self,
        inputs: list[Module],
        out_channels: int | None = None,
        weights: Tensor | None = None,
        num_repetitions: int | None = None,
        sum_dim: int = 3,
    ) -> None:
        """Initialize elementwise sum module.

        Args:
            inputs: Input modules (same features, compatible channels).
            out_channels: Number of output nodes per sum.
            weights: Initial weights (if None, randomly initialized).
            num_repetitions: Number of repetitions.
            sum_dim: Dimension over which to sum.
        """
        super().__init__()

        # ========== 1. INPUT VALIDATION ==========
        if not inputs:
            raise ValueError("'Sum' requires at least one input to be specified.")

        # ========== 2. WEIGHTS PARAMETER PROCESSING ==========
        if weights is not None:
            # Validate mutual exclusivity
            if out_channels is not None:
                raise InvalidParameterCombinationError(
                    f"Cannot specify both 'out_channels' and 'weights' for 'Sum' module."
                )
            if num_repetitions is not None:
                raise InvalidParameterCombinationError(
                    f"Cannot specify both 'num_repetitions' and 'weights' for 'Sum' module."
                )

            # Derive configuration from weights shape
            out_channels = weights.shape[2]
            num_repetitions = weights.shape[4]
        else:
            # Set defaults when weights not provided
            if out_channels is None:
                raise ValueError(f"Either 'out_channels' or 'weights' must be specified for 'Sum' module.")
            if num_repetitions is None:
                num_repetitions = 1

        # ========== 3. CONFIGURATION VALIDATION ==========
        if out_channels < 1:
            raise ValueError(
                f"Number of nodes for 'Sum' must be greater of equal to 1 but was {out_channels}."
            )

        # Validate all inputs have the same number of features
        if not all([module.out_features == inputs[0].out_features for module in inputs]):
            raise ValueError("All inputs must have the same number of features.")

        # Validate all inputs have compatible channels (same or 1 for broadcasting)
        if not all([module.out_channels in (1, max(m.out_channels for m in inputs)) for module in inputs]):
            raise ValueError(
                "All inputs must have the same number of channels or 1 channel (in which case the "
                "operation is broadcast)."
            )

        # Validate all input modules have the same scope
        if not Scope.all_equal([module.scope for module in inputs]):
            raise ScopeError("All input modules must have the same scope.")

        # ========== 4. INPUT MODULE SETUP ==========
        self.inputs = nn.ModuleList(inputs)
        self.sum_dim = sum_dim
        self.scope = inputs[0].scope

        # ========== 5. ATTRIBUTE INITIALIZATION ==========
        self._out_features = self.inputs[0].out_features
        self._num_sums = out_channels
        self.num_repetitions = num_repetitions
        self._num_inputs = len(inputs)

        # Compute channel dimensions and weights shape
        self._in_channels_per_input = max([module.out_channels for module in self.inputs])
        self._out_channels_total = self._num_sums * self._in_channels_per_input

        self.weights_shape = (
            self._out_features,
            self._in_channels_per_input,
            self._num_sums,
            self._num_inputs,
            self.num_repetitions,
        )

        # Register unraveled channel indices for mapping flattened indices to (channel, sum) pairs
        # E.g. for 3 in_channels_per_input and 2 sums: [0,1,2,3,4,5] -> [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
        unraveled_channel_indices = torch.tensor(
            [(i, j) for i in range(self._in_channels_per_input) for j in range(self._num_sums)]
        )
        self.register_buffer(name="unraveled_channel_indices", tensor=unraveled_channel_indices)

        # ========== 6. WEIGHT INITIALIZATION & PARAMETER REGISTRATION ==========
        if weights is None:
            # Initialize weights randomly with small epsilon to avoid zeros
            weights = torch.rand(self.weights_shape) + 1e-08
            # Normalize to sum to one along sum_dim
            weights /= torch.sum(weights, dim=self.sum_dim, keepdim=True)

        # Register parameter for unnormalized log-probabilities
        self.logits = torch.nn.Parameter(torch.zeros(self.weights_shape))

        # Set weights (converts to logits internally via property setter)
        self.weights = weights

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def out_channels(self) -> int:
        return self._out_channels_total

    @property
    def feature_to_scope(self) -> np.ndarray:
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
        """Set weights of all nodes.

        Args:
            values: Weight values to set.
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
        """Set log weights of all nodes.

        Args:
            values: Log weight values to set.
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
            marg_rvs: Random variables to marginalize out.
            prune: Whether to prune the resulting module.
            cache: Cache for memoization.

        Returns:
            Optional[ElementwiseSum]: Marginalized module or None if fully marginalized.
        """
        # initialize cache
        if cache is None:
            cache = Cache()

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
        cache: Cache | None = None,
        sampling_ctx: Optional[SamplingContext] = None,
    ) -> Tensor:
        """Generate samples by choosing mixture components.

        Args:
            num_samples: Number of samples to generate.
            data: Existing data tensor to fill with samples.
            is_mpe: Whether to perform most probable explanation.
            cache: Cache for memoization.
            sampling_ctx: Sampling context for conditional sampling.

        Returns:
            Tensor: Generated samples.
        """
        # Prepare data tensor
        data = self._prepare_sample_data(num_samples, data)

        # initialize contexts
        if cache is None:
            cache = Cache()
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
            if self.num_repetitions > 1:
                raise ValueError(
                    "sampling_ctx.repetition_idx must be provided when sampling from a module with "
                    "num_repetitions > 1."
                )
            logits = self.logits[..., 0]  # Select the 0th repetition
            logits = logits.unsqueeze(0)  # Make space for the batch

            # Expand to batch size
            logits = logits.expand(sampling_ctx.channel_index.shape[0], -1, -1, -1, -1)

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

        if (
            cache is not None
            and "log_likelihood" in cache
            and all(cache["log_likelihood"][inp] is not None for inp in self.inputs)
        ):
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
                cache=cache,
                sampling_ctx=sampling_ctx_cpy,
            )

        return data

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log likelihood via weighted log-sum-exp.

        Args:
            data: Input data tensor.
            cache: Cache for memoization.

        Returns:
            Tensor: Computed log likelihood values.
        """
        # Get input log-likelihoods
        lls = []
        for inp in self.inputs:
            ll = inp.log_likelihood(
                data,
                cache=cache,
            )

            # Prepare for broadcasting
            if inp.out_channels == 1 and self._in_channels_per_input > 1:
                ll = ll.expand(
                    data.shape[0], self.out_features, self._in_channels_per_input, self.num_repetitions
                )

            lls.append(ll)

        # Stack input log-likelihoods
        stacked_lls = torch.stack(lls, dim=self.sum_dim)

        ll = stacked_lls.unsqueeze(3)  # shape: (B, F, IC, 1)

        log_weights = self.log_weights.unsqueeze(0)  # shape: (1, F, IC, OC)

        # Weighted log-likelihoods
        weighted_lls = ll + log_weights  # shape: (B, F, IC, OC)

        # Sum over input channels (sum_dim + 1 since here the batch dimension is the first dimension)
        output = torch.logsumexp(weighted_lls, dim=self.sum_dim + 1)
        output = output.view(data.shape[0], self.out_features, self.out_channels, self.num_repetitions)

        return output

    def expectation_maximization(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> None:
        """Perform EM step to update mixture weights.

        Args:
            data: Training data tensor.
            cache: Cache for memoization.
        """
        # initialize cache
        if cache is None:
            cache = Cache()

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

        for inp in self.inputs:
            inp.expectation_maximization(data, cache=cache)

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Optional[Tensor] = None,
        cache: Cache | None = None,
    ) -> None:
        """MLE step (equivalent to EM for sum nodes).

        Args:
            data: Training data tensor.
            weights: Optional weights for data points.
            cache: Cache for memoization.
        """
        self.expectation_maximization(data, cache=cache)
