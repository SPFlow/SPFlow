from __future__ import annotations

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta.data import Scope
from spflow.modules.base import Module
from spflow.modules.ops.cat import Cat
from spflow.utils.cache import Cache
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
        out_channels: int | None = None,
        num_repetitions: int | None = None,
        weights: Tensor | list[float] | None = None,
        sum_dim: int | None = 1,
    ) -> None:
        """Create a Sum module for mixture modeling.

        Weights are automatically normalized to sum to one using softmax.
        Multiple inputs are concatenated along dimension 2 internally.

        Args:
            inputs (Module | list[Module]): Single module or list of modules to mix.
            out_channels (int | None, optional): Number of output mixture components.
                Required if weights not provided.
            num_repetitions (int | None, optional): Number of repetitions for structured
                representations. Inferred from weights if not provided.
            weights (Tensor | list[float] | None, optional): Initial mixture weights.
                Must have compatible shape with inputs and out_channels.
            sum_dim (int | None, optional): Dimension over which to sum inputs. Default is 1.

        Raises:
            ValueError: If inputs empty, out_channels < 1, or weights have invalid shape/values.
            InvalidParameterCombinationError: If both out_channels and weights are specified.
        """
        super().__init__()

        if not inputs:
            raise ValueError("'Sum' requires at least one input to be specified.")

        if weights is not None:
            if isinstance(weights, list):
                weights = torch.tensor(weights)

            if out_channels is not None:
                raise InvalidParameterCombinationError(
                    f"Cannot specify both 'out_channels' and 'weights' for 'Sum' module."
                )

            match weights.dim():
                case 1:
                    weights = weights.view(1, -1, 1)  # shape: (1, in_channels, 1)
                case 2:
                    if sum_dim > 1:
                        raise ValueError(
                            f"When providing 2D weights, 'sum_dim' must be 0 or 1 but was {sum_dim}."
                        )
                    weights = weights.view(1, weights.shape[0], weights.shape[1])
                case 3:
                    if sum_dim > 3:
                        raise ValueError(
                            f"When providing 3D weights, 'sum_dim' must be 0, 1, or 2 but was {sum_dim}."
                        )
                case 4:
                    if sum_dim > 4:
                        raise ValueError(
                            f"When providing 4D weights, 'sum_dim' must be 0, 1, 2, or 3 but was {sum_dim}."
                        )
                case _:
                    raise ValueError(
                        f"Weights for 'Sum' must be a 1D, 2D, 3D, or 4D tensor but was {weights.dim()}D."
                    )

            out_channels = weights.shape[2]

        if out_channels < 1:
            raise ValueError(
                f"Number of nodes for 'Sum' must be greater of equal to 1 but was {out_channels}."
            )

        # if a list of input modules is provided, concatenate them to single module
        if isinstance(inputs, list):
            if len(inputs) == 1:
                self.inputs = inputs[0]
            else:
                self.inputs = Cat(inputs=inputs, dim=2)
        else:
            self.inputs = inputs

        self.sum_dim = sum_dim
        self._out_features = self.inputs.out_features

        self._in_channels_total = self.inputs.out_channels
        self._out_channels_total = out_channels

        self.num_repetitions = num_repetitions

        # If num_repetitions is not provided, infer it from the weights
        if num_repetitions is None:
            if weights is not None:
                if weights.ndim == 4:
                    self.num_repetitions = weights.shape[3]

        if self.num_repetitions is not None:
            self.weights_shape = (
                self._out_features,
                self._in_channels_total,
                self._out_channels_total,
                self.num_repetitions,
            )
        else:
            self.weights_shape = (self._out_features, self._in_channels_total, self._out_channels_total)

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
    def feature_to_scope(self) -> list[Scope]:
        return self.inputs.feature_to_scope

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def out_channels(self) -> int:
        return self._out_channels_total

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
        """Set weights of all nodes.

        Args:
            values: Tensor containing weights for each input and node.

        Raises:
            ValueError: If weights have invalid shape, contain non-positive values,
                or do not sum to one.
        """
        if values.shape != self.weights_shape:
            raise ValueError(
                f"Invalid shape for weights: Was {values.shape} but expected {self.weights_shape}."
            )
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
            values: Tensor containing log weights for each input and node.

        Raises:
            ValueError: If log weights have invalid shape.
        """
        if values.shape != self.log_weights.shape:
            raise ValueError(f"Invalid shape for weights: {values.shape}.")
        self.logits.data = values

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, weights={self.weights_shape}"

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
        if cache is None: cache = Cache()

        # Get input log-likelihoods
        ll = self.inputs.log_likelihood(
            data,
            cache=cache,
        )

        ll = ll.unsqueeze(3)  # shape: (B, F, input_OC, 1)

        log_weights = self.log_weights.unsqueeze(0)  # shape: (1, F, IC, OC)

        # Weighted log-likelihoods
        weighted_lls = ll + log_weights  # shape: (B, F, IC, OC)

        # Sum over input channels (sum_dim + 1 since here the batch dimension is the first dimension)
        output = torch.logsumexp(weighted_lls, dim=self.sum_dim + 1).squeeze(-1)  # shape: (B, F, OC)

        if self.num_repetitions is None:
            result = output.view(-1, self.out_features, self.out_channels)
        else:
            result = output.view(-1, self.out_features, self.out_channels, self.num_repetitions)

        # Cache the result for EM step
        if "log_likelihood" not in cache:
            cache["log_likelihood"] = {}
        cache["log_likelihood"][self] = result

        return result

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
        if cache is None: cache = Cache()

        # Handle num_samples case (create empty data tensor)
        if data is None:
            if num_samples is None:
                num_samples = 1
            data = torch.full((num_samples, len(self.scope.query)), float("nan")).to(self.device)

        # Initialize sampling context if not provided
        sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0], data.device)

        # Index into the correct weight channels given by parent module
        if sampling_ctx.repetition_idx is not None:
            logits = self.logits.unsqueeze(0).expand(
                sampling_ctx.channel_index.shape[0], -1, -1, -1, -1
            )  # shape [b , n_features , in_c, out_c, r]

            indices = sampling_ctx.repetition_idx  # Shape (30000, 1)

            # Use gather to select the correct repetition
            # Repeat indices to match the target dimension for gathering
            in_channels_total = logits.shape[2]
            indices = indices.view(-1, 1, 1, 1, 1).expand(
                -1, logits.shape[1], in_channels_total, logits.shape[3], -1
            )
            # Gather the logits based on the repetition indices
            logits = torch.gather(logits, dim=-1, index=indices).squeeze(-1)

        else:
            logits = self.logits.unsqueeze(0).expand(sampling_ctx.channel_index.shape[0], -1, -1, -1)

        idxs = sampling_ctx.channel_index[..., None, None]
        in_channels_total = logits.shape[2]
        idxs = idxs.expand(-1, -1, in_channels_total, -1)
        # Gather the logits based on the channel indices
        logits = logits.gather(dim=3, index=idxs).squeeze(3)

        # Check if evidence is given (cached log-likelihoods)
        if cache is not None and "log_likelihood" in cache and cache["log_likelihood"].get(self.inputs) is not None:
            # Get the log likelihoods from the cache
            input_lls = cache["log_likelihood"][self.inputs]

            if sampling_ctx.repetition_idx is not None:
                indices = sampling_ctx.repetition_idx.view(-1, 1, 1, 1).expand(
                    -1, input_lls.shape[1], input_lls.shape[2], -1
                )

                # Use gather to select the correct repetition
                input_lls = torch.gather(input_lls, dim=-1, index=indices).squeeze(-1)

                log_prior = logits
                log_posterior = log_prior + input_lls
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
        cache: Cache | None = None,
    ) -> None:
        """Perform expectation-maximization step.

        Args:
            data: Input data tensor.
            cache: Optional cache dictionary with log-likelihoods.

        Raises:
            ValueError: If required log-likelihoods are not found in cache.
        """
        if cache is None: cache = Cache()

        with torch.no_grad():
            # ----- expectation step -----

            # Get input LLs from cache
            input_lls = cache["log_likelihood"].get(self.inputs)
            if input_lls is None:
                raise ValueError("Input log-likelihoods not found in cache. Call log_likelihood first.")

            # Get module lls from cache
            module_lls = cache["log_likelihood"].get(self)
            if module_lls is None:
                raise ValueError("Module log-likelihoods not found in cache. Call log_likelihood first.")

            log_weights = self.log_weights.unsqueeze(0)
            log_grads = torch.log(module_lls.grad).unsqueeze(2)
            input_lls = input_lls.unsqueeze(3)
            module_lls = module_lls.unsqueeze(2)

            log_expectations = log_weights + log_grads + input_lls - module_lls
            log_expectations = log_expectations.logsumexp(0)  # Sum over batch dimension
            log_expectations = log_expectations.log_softmax(self.sum_dim)  # Normalize

            # ----- maximization step -----
            self.log_weights = log_expectations

        # Recursively call EM on inputs
        self.inputs.expectation_maximization(data, cache=cache)

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
        if cache is None: cache = Cache()

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
                feature_to_scope = self.inputs.feature_to_scope
                # remove mutual_rvs from feature_to_scope list
                for rv in mutual_rvs:
                    for idx, scope in enumerate(feature_to_scope):
                        if scope is not None:
                            if rv in scope.query:
                                feature_to_scope[idx] = scope.remove_from_query(rv)

                # construct mask with empty scopes
                mask = [scope is not None for scope in feature_to_scope]

                module_weights = module_weights[mask]

        else:
            marg_input = self.inputs

        if marg_input is None:
            return None

        else:
            return Sum(inputs=marg_input, weights=module_weights)
