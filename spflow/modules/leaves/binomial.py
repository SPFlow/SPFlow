import torch
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import init_parameter, _handle_mle_edge_cases
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded


class Binomial(LeafModule):
    """Binomial distribution leaf module for probabilistic circuits.

    Implements univariate Binomial distributions as leaf nodes in probabilistic
    circuits. Supports parameter learning through maximum likelihood estimation
    and efficient inference through PyTorch's built-in distributions.

    The Binomial distribution models the number of successes in a fixed number
    of independent Bernoulli trials, with probability mass function:
        P(X = k | n, p) = C(n, k) * p^k * (1-p)^(n-k)

    where n is the number of trials (fixed), p is the success probability (learnable,
    stored in logit-space for numerical stability), and k is the number of successes (0 ≤ k ≤ n).

    Attributes:
        p: Success probability parameter(s) in [0, 1] (BoundedParameter).
        n: Number of trials parameter(s), non-negative integers (fixed buffer).
        distribution: Underlying torch.distributions.Binomial.
    """

    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        total_count: Tensor | None = None,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        parameter_network: nn.Module = None,
        validate_args: bool | None = True,
    ):
        """Initialize Binomial distribution leaf module.

        Args:
            scope: Scope object specifying the scope of the distribution.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions for the distribution.
            total_count: Number of trials tensor (required).
            probs: Success probability tensor (optional, randomly initialized if None).
            logits: Log-odds tensor for success probability.
            parameter_network: Optional neural network for parameter generation.
            validate_args: Whether to enable torch.distributions argument validation.
        """
        if total_count is None:
            raise InvalidParameterCombinationError("'n' parameter is required for Binomial distribution")
        if probs is not None and logits is not None:
            raise InvalidParameterCombinationError("Binomial accepts either probs or logits, not both.")

        param_source = logits if logits is not None else probs
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[param_source],
            parameter_network=parameter_network,
            validate_args=validate_args,
        )

        init_fn = torch.randn if logits is not None else torch.rand
        init_value = init_parameter(param=param_source, event_shape=self.event_shape, init=init_fn)

        # Register total_count as a fixed buffer
        total_count = torch.broadcast_to(total_count, self.event_shape).clone()
        self.register_buffer("_total_count", total_count)

        logits_tensor = init_value if logits is not None else proj_bounded_to_real(init_value, lb=0.0, ub=1.0)

        self._logits = nn.Parameter(logits_tensor)

    @property
    def total_count(self) -> Tensor:
        """Returns the number of trials."""
        return self._total_count

    @total_count.setter
    def total_count(self, total_count: Tensor):
        """Sets the number of trials.

        Args:
            total_count: Floating point representing the number of trials.
        """
        self._total_count = total_count

    @property
    def probs(self) -> Tensor:
        """Success probability in natural space (read via inverse projection of logit_p)."""
        return proj_real_to_bounded(self._logits, lb=0.0, ub=1.0)

    @probs.setter
    def probs(self, value: Tensor) -> None:
        """Set success probability (stores as logit_p, no validation after init)."""
        value_tensor = torch.as_tensor(value, dtype=self._logits.dtype, device=self._logits.device)
        self._logits.data = proj_bounded_to_real(value_tensor, lb=0.0, ub=1.0)

    @property
    def logits(self) -> Tensor:
        """Logits for the success probability."""
        return self._logits

    @logits.setter
    def logits(self, value: Tensor) -> None:
        value_tensor = torch.as_tensor(value, dtype=self._logits.dtype, device=self._logits.device)
        self._logits.data = value_tensor

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return 0.0

    @property
    def _torch_distribution_class(self) -> type[torch.distributions.Binomial]:
        return torch.distributions.Binomial

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"total_count": self.total_count, "logits": self.logits}

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for success probability p.

        Estimates the success probability parameter p using weighted maximum
        likelihood estimation. The parameter n is fixed and not learned.

        Args:
            data: Scope-filtered data.
            weights: Normalized weights.
            bias_correction: Not used for Binomial (included for interface consistency).
        """
        normalized_weights = weights / weights.sum()
        n_total = normalized_weights.sum() * self.total_count
        n_success = (normalized_weights * data).sum(0)
        success_est = self._broadcast_to_event_shape(n_success)
        probs_est = success_est / n_total

        # Handle edge cases before assigning
        probs_est = _handle_mle_edge_cases(probs_est, lb=0.0, ub=1.0)

        # Convert to logits and assign
        self.logits = proj_bounded_to_real(probs_est, lb=0.0, ub=1.0)
