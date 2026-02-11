from __future__ import annotations

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor

from spflow.exceptions import (
    InvalidParameterCombinationError,
    InvalidWeightsError,
    MissingCacheError,
    ShapeError,
)
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.ops.cat import Cat
from spflow.utils.cache import Cache, cached
from spflow.utils.projections import (
    proj_convex_to_real,
)
from spflow.utils.sampling_context import SamplingContext, init_default_sampling_context


class Sum(Module):
    """Sum module representing mixture operations in probabilistic circuits.

    Implements mixture modeling by computing weighted combinations of child distributions.
    Weights are normalized to sum to one, maintaining valid probability distributions.
    Supports both single input (mixture over channels) and multiple inputs (mixture
    over concatenated inputs).

    Attributes:
        inputs (Module): Input module(s) to the sum node.
        sum_dim (int): Dimension over which to sum the inputs.
        weights (Tensor): Normalized weights for mixture components.
        logits (Parameter): Unnormalized log-weights for gradient optimization.
    """

    def __init__(
        self,
        inputs: Module | list[Module],
        out_channels: int = 1,
        num_repetitions: int = 1,
        weights: Tensor | list[float] | None = None,
    ) -> None:
        """Create a Sum module for mixture modeling.

        Weights are automatically normalized to sum to one using softmax.
        Multiple inputs are concatenated along dimension 2 internally.

        Args:
            inputs (Module | list[Module]): Single module or list of modules to mix.
            out_channels (int, optional): Number of output mixture components. Defaults to 1.
            num_repetitions (int | None, optional): Number of repetitions for structured
                representations. Inferred from weights if not provided.
            weights (Tensor | list[float] | None, optional): Initial mixture weights.
                Must have compatible shape with inputs and out_channels.

        Raises:
            ValueError: If inputs empty, out_channels < 1, or weights have invalid shape/values.
            InvalidParameterCombinationError: If both out_channels and weights are specified.
        """
        super().__init__()

        # ========== 1. INPUT VALIDATION ==========
        if not inputs:
            raise ValueError("'Sum' requires at least one input to be specified.")

        if weights is not None and isinstance(weights, list):
            weights = torch.as_tensor(weights, dtype=torch.get_default_dtype())

        weights, out_channels_inferred, num_repetitions_inferred = self._process_weights_parameter(
            inputs=inputs,
            weights=weights,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
        )

        # Use inferred values
        out_channels = out_channels_inferred
        if num_repetitions_inferred is not None:
            num_repetitions = num_repetitions_inferred
        if num_repetitions is None:
            num_repetitions = 1

        # ========== 3. CONFIGURATION VALIDATION ==========
        if out_channels < 1:
            raise ValueError(
                f"Number of nodes for 'Sum' must be greater or equal to 1 but was {out_channels}."
            )

        # ========== 4. INPUT MODULE SETUP ==========
        if isinstance(inputs, list):
            if len(inputs) == 1:
                self.inputs = inputs[0]
            else:
                self.inputs = Cat(inputs=inputs, dim=2)
        else:
            self.inputs = inputs

        self.sum_dim = 1
        self.scope = self.inputs.scope

        # ========== 5. SHAPE COMPUTATION (early, so shapes can be reused below) ==========
        self.in_shape = self.inputs.out_shape
        self.out_shape = ModuleShape(
            features=self.in_shape.features, channels=out_channels, repetitions=num_repetitions
        )

        # ========== 6. WEIGHT INITIALIZATION & PARAMETER REGISTRATION ==========
        self.weights_shape = self._get_weights_shape()

        weights = self._initialize_weights(weights)

        # Register parameter for unnormalized log-probabilities
        self.logits = torch.nn.Parameter()

        # Set weights (converts to logits internally via property setter)
        self.weights = weights

    def _process_weights_parameter(
        self,
        inputs: Module | list[Module],
        weights: Tensor | None,
        out_channels: int,
        num_repetitions: int | None,
    ) -> tuple[Tensor | None, int, int | None]:
        if weights is None:
            return weights, out_channels, num_repetitions

        # If out_channels is not the default (1), user explicitly specified both
        if out_channels != 1:
            raise InvalidParameterCombinationError(
                f"Cannot specify both 'out_channels' and 'weights' for 'Sum' module. "
                f"Use only 'weights' to set the number of output channels."
            )

        weight_dim = weights.dim()
        if weight_dim == 1:
            weights = rearrange(weights, "ci -> 1 ci 1 1")
        elif weight_dim == 2:
            weights = rearrange(weights, "ci co -> 1 ci co 1")
        elif weight_dim == 3:
            weights = rearrange(weights, "f ci co -> f ci co 1")
        elif weight_dim == 4:
            pass
        else:
            raise ShapeError(f"Weights for 'Sum' must be a 1D, 2D, 3D, or 4D tensor but was {weight_dim}D.")

        inferred_num_repetitions = weights.shape[-1]
        if num_repetitions is not None and (
            num_repetitions != 1 and num_repetitions != inferred_num_repetitions
        ):
            raise InvalidParameterCombinationError(
                f"Cannot specify 'num_repetitions' that does not match weights shape for 'Sum' module. "
                f"Was {num_repetitions} but weights shape indicates {inferred_num_repetitions}."
            )
        num_repetitions = inferred_num_repetitions

        out_channels = weights.shape[2]

        return weights, out_channels, num_repetitions

    def _get_weights_shape(self) -> tuple[int, int, int, int]:
        return (
            self.in_shape.features,
            self.in_shape.channels,
            self.out_shape.channels,
            self.out_shape.repetitions,
        )

    def _initialize_weights(self, weights: Tensor | None) -> Tensor:
        if weights is None:
            weights = torch.rand(self.weights_shape) + 1e-08
            weights /= torch.sum(weights, dim=self.sum_dim, keepdims=True)
        return weights

    @property
    def feature_to_scope(self) -> np.ndarray:
        return self.inputs.feature_to_scope

    @property
    def log_weights(self) -> Tensor:
        """Returns the log weights of all nodes as a tensor.

        Returns:
            Tensor: Log weights normalized to sum to one.
        """
        # project auxiliary weights onto weights that sum up to one
        return torch.nn.functional.log_softmax(self.logits, dim=self.sum_dim)

    @property
    def weights(self) -> Tensor:
        """Returns the weights of all nodes as a tensor.

        Returns:
            Tensor: Weights normalized to sum to one.
        """
        # project auxiliary weights onto weights that sum up to one
        return torch.nn.functional.softmax(self.logits, dim=self.sum_dim)

    @weights.setter
    def weights(
        self,
        values: Tensor,
    ) -> None:
        if values.shape != self.weights_shape:
            raise ShapeError(
                f"Invalid shape for weights: Was {values.shape} but expected {self.weights_shape}."
            )
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
            values: Tensor containing log weights for each input and node.

        Raises:
            ShapeError: If log weights have invalid shape.
        """
        if values.shape != self.log_weights.shape:
            raise ShapeError(f"Invalid shape for weights: {values.shape}.")
        self.logits.data = values

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, weights={self.weights_shape}"

    @cached
    def log_likelihood(
        self,
        data: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log likelihood P(data | module).

        Computes log likelihood using logsumexp for numerical stability.
        Results are cached for parameter learning algorithms.

        Args:
            data: Input data of shape (batch_size, num_features).
                NaN values indicate evidence for conditional computation.
            cache: Cache for intermediate computations. Defaults to None.

        Returns:
            Tensor: Log-likelihood of shape (batch_size, num_features, out_channels)
                or (batch_size, num_features, out_channels, num_repetitions).
        """
        if cache is None:
            cache = Cache()

        # Get input log-likelihoods
        ll = self.inputs.log_likelihood(
            data,
            cache=cache,
        )

        ll = rearrange(ll, "b f ci r -> b f ci 1 r")

        log_weights = rearrange(self.log_weights, "f ci co r -> 1 f ci co r")

        # Weighted log-likelihoods
        weighted_lls = ll + log_weights  # shape: (B, F, IC, OC, R)

        # Sum over input channels (sum_dim + 1 since here the batch dimension is the first dimension)
        output = torch.logsumexp(weighted_lls, dim=self.sum_dim + 1)

        return output

    def sample(
        self,
        num_samples: int | None = None,
        data: Tensor | None = None,
        is_mpe: bool = False,
        cache: Cache | None = None,
        sampling_ctx: SamplingContext | None = None,
    ) -> Tensor:
        """Generate samples from sum module.

        Args:
            num_samples: Number of samples to generate.
            data: Data tensor with NaN values to fill with samples.
            is_mpe: Whether to perform maximum a posteriori estimation.
            cache: Optional cache dictionary.
            sampling_ctx: Optional sampling context.

        Returns:
            Tensor: Sampled values.
        """
        if cache is None:
            cache = Cache()

        # Handle num_samples case (create empty data tensor)
        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.full((num_samples, len(self.scope.query)), float("nan")).to(self.device)

        # Initialize sampling context if not provided
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)

        # Index into the correct weight channels given by parent module
        if sampling_ctx.repetition_idx is not None:
            batch_size = int(sampling_ctx.channel_index.shape[0])
            logits = repeat(self.logits, "f ci co r -> b f ci co r", b=batch_size)
            # shape [b, n_features, in_c, out_c, r]

            indices = sampling_ctx.repetition_idx

            # Use gather to select the correct repetition
            # Repeat indices to match the target dimension for gathering
            num_features = int(logits.shape[1])
            in_channels_total = logits.shape[2]
            out_channels = int(logits.shape[3])
            indices = repeat(
                rearrange(indices, "... -> (...)"),
                "n -> n f ci co 1",
                f=num_features,
                ci=in_channels_total,
                co=out_channels,
            )
            # Gather the logits based on the repetition indices
            logits = torch.gather(logits, dim=-1, index=indices)
            logits = rearrange(logits, "b f ci co 1 -> b f ci co")

        else:
            if self.out_shape.repetitions > 1:
                raise ValueError(
                    "sampling_ctx.repetition_idx must be provided when sampling from a module with "
                    "num_repetitions > 1."
                )
            logits = self.logits[..., 0]  # Select the 0th repetition
            logits = rearrange(logits, "f ci co -> 1 f ci co")

            # Expand to batch size
            batch_size = int(sampling_ctx.channel_index.shape[0])
            logits = repeat(logits, "1 f ci co -> b f ci co", b=batch_size)

        in_channels_total = logits.shape[2]
        idxs = repeat(sampling_ctx.channel_index, "b f -> b f ci 1", ci=in_channels_total)
        # Gather the logits based on the channel indices
        logits = logits.gather(dim=3, index=idxs)
        logits = rearrange(logits, "b f ci 1 -> b f ci")

        # Check if evidence is given (cached log-likelihoods)
        if (
            cache is not None
            and "log_likelihood" in cache
            and cache["log_likelihood"].get(self.inputs) is not None
        ):
            # Get the log likelihoods from the cache
            input_lls = cache["log_likelihood"][self.inputs]

            if sampling_ctx.repetition_idx is not None:
                num_features = int(input_lls.shape[1])
                in_channels_total = int(input_lls.shape[2])
                indices = repeat(
                    rearrange(sampling_ctx.repetition_idx, "... -> (...)"),
                    "n -> n f ci 1",
                    f=num_features,
                    ci=in_channels_total,
                )

                # Use gather to select the correct repetition
                input_lls = torch.gather(input_lls, dim=-1, index=indices)
                input_lls = rearrange(input_lls, "b f ci 1 -> b f ci")
            else:
                # When no repetition_idx, squeeze the repetitions dimension of input_lls
                if input_lls.dim() == 4 and input_lls.shape[-1] == 1:
                    input_lls = rearrange(input_lls, "b f ci 1 -> b f ci")

            log_prior = logits
            log_posterior = log_prior + input_lls
            log_posterior = log_posterior.log_softmax(dim=2)
            logits = log_posterior

        # Sample from categorical distribution defined by weights to obtain indices into input channels
        if is_mpe:
            # Take the argmax of the logits to obtain the most probable index
            new_channel_index = torch.argmax(logits, dim=-1)
        else:
            # Sample from categorical distribution defined by weights to obtain indices into input channels
            new_channel_index = torch.distributions.Categorical(logits=logits).sample()

        # Update sampling context with new channel indices
        # If shape changes, expand the mask to match new channel_index shape
        if new_channel_index.shape != sampling_ctx.mask.shape:
            # Expand mask from (batch, 1) or (batch, old_features) to (batch, new_features)
            num_features = int(new_channel_index.shape[1])
            current_mask_features = int(sampling_ctx.mask.shape[1])
            if current_mask_features == num_features:
                new_mask = sampling_ctx.mask.contiguous()
            elif current_mask_features == 1:
                new_mask = repeat(sampling_ctx.mask, "b 1 -> b f", f=num_features).contiguous()
            else:
                raise ShapeError(
                    "sampling_ctx.mask has incompatible feature width for sampling update: "
                    f"got {current_mask_features}, expected 1 or {num_features}."
                )
            sampling_ctx.update(new_channel_index, new_mask)
        else:
            sampling_ctx.channel_index = new_channel_index

        # Sample from input module
        self.inputs.sample(
            data=data,
            is_mpe=is_mpe,
            cache=cache,
            sampling_ctx=sampling_ctx,
        )

        return data


    def expectation_maximization(
        self,
        data: Tensor,
        bias_correction: bool = True,
        cache: Cache | None = None,
    ) -> None:
        """Perform expectation-maximization step.

        Args:
            data: Input data tensor.
            cache: Optional cache dictionary with log-likelihoods.
            bias_correction: Whether to apply bias correction.

        Raises:
            MissingCacheError: If required log-likelihoods are not found in cache.
        """
        if cache is None:
            cache = Cache()

        with torch.no_grad():
            # ----- expectation step -----

            # Get input LLs from cache
            input_lls = cache["log_likelihood"].get(self.inputs)
            if input_lls is None:
                raise MissingCacheError(
                    "Input log-likelihoods not found in cache. Call log_likelihood first."
                )

            # Get module lls from cache
            module_lls = cache["log_likelihood"].get(self)
            if module_lls is None:
                raise MissingCacheError(
                    "Module log-likelihoods not found in cache. Call log_likelihood first."
                )

            log_weights = rearrange(self.log_weights, "f ci co r -> 1 f ci co r")
            log_grads = rearrange(torch.log(module_lls.grad), "b f co r -> b f 1 co r")
            input_lls = rearrange(input_lls, "b f ci r -> b f ci 1 r")
            module_lls = rearrange(module_lls, "b f co r -> b f 1 co r")

            log_expectations = log_weights + log_grads + input_lls - module_lls
            log_expectations = log_expectations.logsumexp(0)  # Sum over batch dimension
            log_expectations = log_expectations.log_softmax(self.sum_dim)  # Normalize

            # ----- maximization step -----
            self.log_weights = log_expectations

        # Recursively call EM on inputs
        self.inputs.expectation_maximization(data, cache=cache, bias_correction=bias_correction)

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Tensor | None = None,
        cache: Cache | None = None,
    ) -> None:
        """Update parameters via maximum likelihood estimation.

        For Sum modules, this is equivalent to EM.

        Args:
            data: Input data tensor.
            weights: Optional sample weights (currently unused).
            cache: Optional cache dictionary.
        """
        self.expectation_maximization(data, cache=cache)

    def marginalize(
        self,
        marg_rvs: list[int],
        prune: bool = True,
        cache: Cache | None = None,
    ) -> Sum | None:
        """Marginalize out specified random variables.

        Args:
            marg_rvs: List of random variables to marginalize.
            prune: Whether to prune the module.
            cache: Optional cache dictionary.

        Returns:
            Marginalized Sum module or None.
        """
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
            return None

        # node scope is being partially marginalized
        elif mutual_rvs:
            # marginalize input modules
            marg_input = self.inputs.marginalize(marg_rvs, prune=prune, cache=cache)

            # if marginalized input is not None
            if marg_input:
                # Apply mask to weights per-repetition
                masked_weights_list = []
                for r in range(self.out_shape.repetitions):
                    feature_to_scope_r = self.inputs.feature_to_scope[:, r].copy()
                    # remove mutual_rvs from feature_to_scope list
                    for rv in mutual_rvs:
                        for idx, scope in enumerate(feature_to_scope_r):
                            if scope is not None:
                                if rv in scope.query:
                                    feature_to_scope_r[idx] = scope.remove_from_query(rv)

                    # construct mask with empty scopes
                    mask = torch.tensor(
                        [not scope.empty() for scope in feature_to_scope_r], device=self.device
                    ).bool()

                    # Apply mask to weights for this repetition: (out_features, in_channels, out_channels)
                    masked_weights_r = module_weights[:, :, :, r][mask]
                    masked_weights_list.append(masked_weights_r)

                # Stack weights back along the repetition dimension
                # Handle different repetition counts if needed
                if all(w.shape[0] == masked_weights_list[0].shape[0] for w in masked_weights_list):
                    # All repetitions have same number of features, can stack directly
                    module_weights = torch.stack(masked_weights_list, dim=-1)
                else:
                    # Features differ across repetitions - this shouldn't happen in practice
                    # but handle gracefully by keeping the largest
                    max_features = max(w.shape[0] for w in masked_weights_list)
                    padded_list = []
                    for w in masked_weights_list:
                        if w.shape[0] < max_features:
                            padding = torch.zeros(
                                max_features - w.shape[0],
                                w.shape[1],
                                w.shape[2],
                                device=w.device,
                                dtype=w.dtype,
                            )
                            w = torch.cat([w, padding], dim=0)
                        padded_list.append(w)
                    module_weights = torch.stack(padded_list, dim=-1)

        else:
            marg_input = self.inputs

        if marg_input is None:
            return None

        else:
            return Sum(inputs=marg_input, weights=module_weights)
