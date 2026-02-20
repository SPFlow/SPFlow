from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from spflow.exceptions import (
    InvalidParameterError,
    InvalidParameterCombinationError,
    InvalidWeightsError,
    MissingCacheError,
    ScopeError,
    ShapeError,
)
from spflow.meta.data import Scope
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.utils.cache import Cache, cached
from spflow.utils.projections import (
    proj_convex_to_real,
)
from spflow.utils.sampling_context import (
    SamplingContext,
    index_tensor,
    repeat_channel_index,
    repeat_repetition_index,
    sample_from_logits,
)


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
    ) -> None:
        """Initialize elementwise sum module.

        Args:
            inputs: Input modules (same features, compatible channels).
            out_channels: Number of output nodes per sum. Note that this results in a total of
                             out_channels * in_channels (input modules) output channels since we sum over the list of
                             modules.
            weights: Initial weights (if None, randomly initialized).
            num_repetitions: Number of repetitions.
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

            if weights.dim() != 5:
                raise ShapeError(
                    f"Weights for 'ElementwiseSum' must be a 5D tensor but was {weights.dim()}D."
                )

            # Derive configuration from weights shape
            out_channels = weights.shape[2]
            inferred_num_repetitions = weights.shape[4]

            num_repetitions = inferred_num_repetitions
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
        if not all([module.out_shape.features == inputs[0].out_shape.features for module in inputs]):
            raise ShapeError("All inputs must have the same number of features.")

        # Validate all inputs have compatible channels (same or 1 for broadcasting)
        if not all(
            [module.out_shape.channels in (1, max(m.out_shape.channels for m in inputs)) for module in inputs]
        ):
            raise ShapeError(
                "All inputs must have compatible channels: same number of channels or 1 channel (in which "
                "case the operation is broadcast)."
            )

        # Validate all input modules have the same scope
        if not Scope.all_equal([module.scope for module in inputs]):
            raise ScopeError("All input modules must have the same scope.")

        # Validate for each repetition that modules have the same features_to_scope mapping
        for rep in range(num_repetitions):
            feature_to_scope = inputs[0].feature_to_scope[..., rep]
            for module in inputs[1:]:
                if not np.array_equal(feature_to_scope, module.feature_to_scope[..., rep]):
                    raise ScopeError(
                        "All input modules must have the same feature to scope mapping for each repetition."
                    )

        # ========== 4. INPUT MODULE SETUP ==========
        self.inputs = nn.ModuleList(inputs)
        self.sum_dim = 3
        self.scope = inputs[0].scope

        # ========== 5. SHAPE COMPUTATION (early, so shapes can be reused below) ==========
        in_channels = max(module.out_shape.channels for module in self.inputs)
        out_channels_total = out_channels * in_channels
        self._num_sums = out_channels  # Store for use in sampling

        self.in_shape = ModuleShape(inputs[0].out_shape.features, in_channels, num_repetitions)
        self.out_shape = ModuleShape(self.in_shape.features, out_channels_total, num_repetitions)

        # ========== 6. WEIGHT INITIALIZATION & PARAMETER REGISTRATION ==========
        self.weights_shape = (
            self.in_shape.features,
            self.in_shape.channels,
            out_channels,
            len(inputs),
            self.out_shape.repetitions,
        )

        # Register unraveled channel indices for mapping flattened indices to (channel, sum) pairs
        # E.g. for 3 in_channels and 2 out_channels: [0,1,2,3,4,5] -> [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
        unraveled_channel_indices = torch.tensor(
            [(i, j) for i in range(self.in_shape.channels) for j in range(self._num_sums)],
            dtype=torch.long,
        )
        self.register_buffer(name="unraveled_channel_indices", tensor=unraveled_channel_indices)
        # Differentiable sampling routes parent channels as one-hot/soft vectors over the
        # flattened channel axis (ci * co). These projection matrices map that flattened
        # distribution to child-local marginals over ci and co without integer indexing.
        self.register_buffer(
            name="flat_to_input_channels",
            tensor=torch.nn.functional.one_hot(
                unraveled_channel_indices[:, 0], num_classes=self.in_shape.channels
            ).to(dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            name="flat_to_sum_channels",
            tensor=torch.nn.functional.one_hot(
                unraveled_channel_indices[:, 1], num_classes=self._num_sums
            ).to(dtype=torch.get_default_dtype()),
        )

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
            raise ShapeError(f"Invalid shape for weights: {values.shape}.")
        if not torch.all(values > 0):
            raise InvalidWeightsError("Weights for 'Sum' must be all positive.")
        if not torch.allclose(torch.sum(values, dim=self.sum_dim), values.new_tensor(1.0)):
            raise InvalidWeightsError("Weights for 'Sum' must sum up to one.")
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
                mask = torch.ones(len(module_scope.query), device=module_weights.device, dtype=torch.bool)
                mask[indices] = False
                module_weights = module_weights[mask]

        else:
            marg_input = self.inputs

        if marg_input is None:
            return None

        else:
            return ElementwiseSum(inputs=[inp for inp in marg_input], weights=module_weights)

    def _sample(
        self,
        data: Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
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

        # initialize contexts
        sampling_ctx.validate_sampling_context(
            num_samples=data.shape[0],
            num_features=self.out_shape.features,
            num_channels=self.out_shape.channels,
            num_repetitions=self.out_shape.repetitions,
            allowed_feature_widths=(1, self.out_shape.features),
        )
        sampling_ctx.broadcast_feature_width(target_features=self.out_shape.features, allow_from_one=True)
        if sampling_ctx.is_differentiable and not sampling_ctx.hard:
            raise InvalidParameterError(
                "ElementwiseSum differentiable sampling requires hard=True because routing masks are boolean."
            )

        # Index into the correct weight channels given by parent module
        # (stay in logits space since Categorical distribution accepts logits directly)
        batch_size = int(sampling_ctx.channel_index.shape[0])
        logits = repeat(self.logits, "f ci co i r -> b f ci co i r", b=batch_size)
        num_features = int(logits.shape[1])
        num_input_channels = int(logits.shape[2])
        num_output_channels = int(logits.shape[3])
        num_inputs = int(logits.shape[4])
        rep_idx = repeat_repetition_index(
            sampling_ctx.repetition_index,
            "n r -> n f ci co i r",
            f=num_features,
            ci=num_input_channels,
            co=num_output_channels,
            i=num_inputs,
        )
        logits = index_tensor(
            logits,
            index=rep_idx,
            dim=-1,
            is_differentiable=sampling_ctx.is_differentiable,
        )

        if sampling_ctx.is_differentiable:
            flat_to_input = self.flat_to_input_channels.to(
                device=sampling_ctx.channel_index.device,
                dtype=sampling_ctx.channel_index.dtype,
            )
            flat_to_sum = self.flat_to_sum_channels.to(
                device=sampling_ctx.channel_index.device,
                dtype=sampling_ctx.channel_index.dtype,
            )
            # [B,F,CI*CO] @ [CI*CO,CI/CO] -> [B,F,CI] / [B,F,CO]
            cids_in_channels_per_input = sampling_ctx.channel_index @ flat_to_input
            cids_num_sums = sampling_ctx.channel_index @ flat_to_sum
        else:
            cids_mapped = self.unraveled_channel_indices[sampling_ctx.channel_index]
            cids_in_channels_per_input = cids_mapped[..., 0]
            cids_num_sums = cids_mapped[..., 1]

        # Select sum-channel and then input-channel for each parent channel route.
        num_input_channels = int(logits.shape[2])
        num_inputs = int(logits.shape[4])
        cids_num_sums_idx = repeat_channel_index(
            cids_num_sums,
            "b f co -> b f ci co i",
            ci=num_input_channels,
            i=num_inputs,
        )
        logits = index_tensor(
            logits,
            index=cids_num_sums_idx,
            dim=3,
            is_differentiable=sampling_ctx.is_differentiable,
        )
        cids_in_channels_idx = repeat_channel_index(
            cids_in_channels_per_input,
            "b f ci -> b f ci i",
            i=num_inputs,
        )
        logits = index_tensor(
            logits,
            index=cids_in_channels_idx,
            dim=2,
            is_differentiable=sampling_ctx.is_differentiable,
        )

        if "log_likelihood" in cache and all(cache["log_likelihood"][inp] is not None for inp in self.inputs):
            input_lls = []
            for inp in self.inputs:
                inp_ll = cache["log_likelihood"][inp]
                if inp.out_shape.channels == 1 and self.in_shape.channels > 1:
                    inp_ll = repeat(inp_ll, "b f 1 r -> b f ci r", ci=self.in_shape.channels)
                input_lls.append(inp_ll)
            input_lls = torch.stack(input_lls, dim=self.sum_dim)
            if sampling_ctx.repetition_index is not None:
                num_features = int(input_lls.shape[1])
                num_input_channels = int(input_lls.shape[2])
                num_inputs = int(input_lls.shape[3])
                rep_idx = repeat_repetition_index(
                    sampling_ctx.repetition_index,
                    "n r -> n f ci i r",
                    f=num_features,
                    ci=num_input_channels,
                    i=num_inputs,
                )
                input_lls = index_tensor(
                    input_lls,
                    index=rep_idx,
                    dim=-1,
                    is_differentiable=sampling_ctx.is_differentiable,
                )
            is_conditional = True
        else:
            is_conditional = False

        if is_conditional:
            num_inputs = int(input_lls.shape[3])
            cids_in_channels_input_lls = repeat_channel_index(
                cids_in_channels_per_input,
                "b f ci -> b f ci i",
                i=num_inputs,
            )
            input_lls = index_tensor(
                input_lls,
                index=cids_in_channels_input_lls,
                dim=2,
                is_differentiable=sampling_ctx.is_differentiable,
            )

            # Compute log posterior by reweighing logits with input lls
            log_prior = logits
            log_posterior = log_prior + input_lls
            log_posterior = log_posterior.log_softmax(dim=2)
            logits = log_posterior

        # Sample/MPE over the input-module axis.
        cids_stack = sample_from_logits(
            logits=logits,
            dim=-1,
            is_mpe=sampling_ctx.is_mpe,
            is_differentiable=sampling_ctx.is_differentiable,
            hard=sampling_ctx.hard,
            tau=sampling_ctx.tau,
        )
        if sampling_ctx.is_differentiable:
            selected_input = cids_stack.argmax(dim=-1)
        else:
            selected_input = cids_stack

        for i, inp in enumerate(self.inputs):
            child_mask = sampling_ctx.mask & (selected_input == i)
            sampling_ctx_cpy = sampling_ctx.with_routing(
                channel_index=cids_in_channels_per_input,
                mask=child_mask,
            )

            # Sample from input module
            inp._sample(
                data=data,
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
            if inp.out_shape.channels == 1 and self.in_shape.channels > 1:
                num_input_channels = self.in_shape.channels
                ll = repeat(
                    ll,
                    "b f 1 r -> b f ci r",
                    ci=num_input_channels,
                )

            lls.append(ll)

        # Stack input log-likelihoods
        stacked_lls = torch.stack(lls, dim=self.sum_dim)

        ll = rearrange(stacked_lls, "b f ci i r -> b f ci 1 i r")

        log_weights = rearrange(self.log_weights, "f ci co i r -> 1 f ci co i r")

        # Weighted log-likelihoods
        weighted_lls = ll + log_weights  # shape: (B, F, IC, OC)

        # Sum over input channels (sum_dim + 1 since here the batch dimension is the first dimension)
        output = torch.logsumexp(weighted_lls, dim=self.sum_dim + 1)
        output = rearrange(output, "b f ci co r -> b f (ci co) r")

        return output

    def _expectation_maximization_step(
        self,
        data: Tensor,
        bias_correction: bool = True,
        *,
        cache: Cache,
    ) -> None:
        """Perform EM step to update mixture weights.

        Args:
            data: Training data tensor.
            cache: Cache for memoization.
        """
        with torch.no_grad():
            # ----- expectation step -----

            # Get input LLs
            input_lls = []
            for inp in self.inputs:
                inp_ll = cache.get("log_likelihood", inp)
                if inp_ll is None:
                    raise MissingCacheError("Input log-likelihoods not found in cache.")
                input_lls.append(inp_ll)
            input_lls = torch.stack(input_lls, dim=3)

            # Get module lls
            module_lls = cache.get("log_likelihood", self)
            if module_lls is None:
                raise MissingCacheError("Module log-likelihood not found in cache.")

            log_weights = rearrange(self.log_weights, "f ci co i r -> 1 f ci co i r")
            input_lls = rearrange(input_lls, "b f ci i r -> b f ci 1 i r")
            log_grads = rearrange(
                torch.log(module_lls.grad),
                "b f (ci co) r -> b f ci co 1 r",
                ci=self.in_shape.channels,
                co=self._num_sums,
            )
            module_lls = rearrange(
                module_lls,
                "b f (ci co) r -> b f ci co 1 r",
                ci=self.in_shape.channels,
                co=self._num_sums,
            )

            log_expectations = log_weights + log_grads + input_lls - module_lls
            log_expectations = log_expectations.logsumexp(0)  # Sum over batch dimension
            log_expectations = log_expectations.log_softmax(self.sum_dim)  # Normalize

            # ----- maximization step -----
            self.log_weights = log_expectations

        for inp in self.inputs:
            inp._expectation_maximization_step(data, bias_correction=bias_correction, cache=cache)
