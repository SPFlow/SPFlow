import torch
from einops import rearrange
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterCombinationError
from spflow.modules.leaves.leaf import LeafModule
from spflow.utils.cache import Cache
from spflow.utils.leaves import init_parameter, _handle_mle_edge_cases
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded
from spflow.utils.sampling_context import SIMPLE


class Bernoulli(LeafModule):
    """Bernoulli distribution leaf module.

    Binary random variable with success probability p ∈ [0, 1].
    Parameterized by success probability p ∈ [0, 1] (stored in logit-space for numerical stability).

    Attributes:
        p: Success probability (BoundedParameter).
        distribution: Underlying torch.distributions.Bernoulli.
    """

    def __init__(
        self,
        scope,
        out_channels: int = 1,
        num_repetitions: int = 1,
        parameter_fn: nn.Module = None,
        validate_args: bool | None = True,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
    ):
        """Initialize Bernoulli distribution.

        Args:
            scope: Variable scope (Scope, int, or list[int]).
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions (for 3D event shapes).
            parameter_fn: Optional neural network for parameter generation.
            validate_args: Whether to enable torch.distributions argument validation.
            probs: Success probability tensor in [0, 1].
            logits: Logits corresponding to success probability.
        """
        if probs is not None and logits is not None:
            raise InvalidParameterCombinationError("Bernoulli accepts either probs or logits, not both.")

        param_source = logits if logits is not None else probs
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[param_source],
            parameter_fn=parameter_fn,
            validate_args=validate_args,
        )

        init_fn = torch.randn if logits is not None else torch.rand
        init_value = init_parameter(param=param_source, event_shape=self._event_shape, init=init_fn)
        logits_tensor = init_value if logits is not None else proj_bounded_to_real(init_value, lb=0.0, ub=1.0)

        self._logits = nn.Parameter(logits_tensor)

    @property
    def probs(self) -> Tensor:
        """Success probability in natural space (read via inverse projection of logits)."""
        return proj_real_to_bounded(self._logits, lb=0.0, ub=1.0)

    @probs.setter
    def probs(self, value: Tensor) -> None:
        """Set success probability (stores as logits)."""
        value_tensor = torch.as_tensor(value, dtype=self._logits.dtype, device=self._logits.device)
        self._logits.data = proj_bounded_to_real(value_tensor, lb=0.0, ub=1.0)

    @property
    def logits(self) -> Tensor:
        """Logits of the Bernoulli distribution."""
        return self._logits

    @logits.setter
    def logits(self, value: Tensor) -> None:
        """Set logits directly."""
        self._logits.data = torch.as_tensor(value, dtype=self._logits.dtype, device=self._logits.device)

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return 0.0

    @property
    def _torch_distribution_class(self) -> type[torch.distributions.Bernoulli]:
        return torch.distributions.Bernoulli

    @property
    def _torch_distribution_class_with_differentiable_sampling(
        self,
    ) -> type[torch.distributions.Distribution]:
        """Return a distribution class that supports differentiable sampling."""
        return BernoulliWithDifferentiableSampling

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"logits": self.logits}

    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> dict[str, Tensor]:
        """Compute raw MLE estimates for Bernoulli distribution (without broadcasting).

        For Bernoulli distribution, the MLE is the weighted proportion of successes.

        Args:
            data: Scope-filtered data.
            weights: Normalized sample weights.
            bias_correction: Not used for Bernoulli.

        Returns:
            Dictionary with 'probs' estimate (shape: out_features).
        """
        n_total = weights.sum(dim=0)
        n_success = (weights * data).sum(dim=0)
        p_est = n_success / n_total

        # Handle edge cases (NaN, zero, or near-zero p) before broadcasting
        p_est = _handle_mle_edge_cases(p_est, lb=0.0)

        return {"probs": p_est}

    def _set_mle_parameters(self, params_dict: dict[str, Tensor]) -> None:
        """Set MLE-estimated parameters for Bernoulli distribution.

        Explicitly handles the parameter:
        - probs: Property with setter, calls property setter which updates _logits

        Args:
            params_dict: Dictionary with 'probs' parameter value.
        """
        self.probs = params_dict["probs"]  # Uses property setter

    def _log_likelihood_interval(
        self,
        low: Tensor,
        high: Tensor,
        cache: Cache | None = None,
    ) -> Tensor:
        """Compute log P(low <= X <= high) for interval evidence.

        Args:
            low: Lower bounds of shape (batch, features).
            high: Upper bounds of shape (batch, features).
            cache: Optional cache dictionary.

        Returns:
            Log-likelihood tensor.
        """
        # probs shape: (features, channels, repetitions)
        probs = self.probs

        # Get scope-filtered bounds
        low_scoped = low[:, self.scope.query]
        high_scoped = high[:, self.scope.query]

        # Expand to match (batch, features, channels, repetitions)
        low_expanded = rearrange(low_scoped, "b f -> b f 1 1")
        high_expanded = rearrange(high_scoped, "b f -> b f 1 1")

        # Handle NaN bounds
        low_processed = torch.where(
            torch.isnan(low_expanded),
            torch.zeros_like(low_expanded),
            torch.ceil(low_expanded),
        )
        high_processed = torch.where(
            torch.isnan(high_expanded),
            torch.ones_like(high_expanded),
            torch.floor(high_expanded),
        )

        # Expand probs for broadcasting: (1, features, channels, repetitions)
        p = rearrange(probs, "f ci r -> 1 f ci r")

        # P(X=1) = p
        # P(X=0) = 1-p

        # Check coverage for 0 and 1
        include_0 = (low_processed <= 0) & (high_processed >= 0)
        include_1 = (low_processed <= 1) & (high_processed >= 1)

        prob0 = torch.where(include_0, 1 - p, torch.zeros_like(p))
        prob1 = torch.where(include_1, p, torch.zeros_like(p))

        total_prob = prob0 + prob1

        return torch.log(torch.clamp(total_prob, min=1e-40))


class BernoulliWithDifferentiableSampling(torch.distributions.Bernoulli):
    """Bernoulli distribution class that supports differentiable sampling via RelaxedBernoulli."""

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        """Generate differentiable samples using SIMPLE."""
        if isinstance(sample_shape, torch.Size):
            expanded_shape = tuple(int(s) for s in sample_shape)
        elif isinstance(sample_shape, tuple):
            expanded_shape = tuple(int(s) for s in sample_shape)
        else:
            expanded_shape = (int(sample_shape),)

        logits = self.logits
        log_p0 = torch.nn.functional.logsigmoid(-logits)
        log_p1 = torch.nn.functional.logsigmoid(logits)
        two_class_logits = torch.stack((log_p0, log_p1), dim=-1)

        if expanded_shape:
            two_class_logits = two_class_logits.expand(*expanded_shape, *two_class_logits.shape)

        samples_oh = SIMPLE(logits=two_class_logits, dim=-1, is_mpe=False)
        samples = samples_oh[..., 1]  # Extract class "1".
        return samples
