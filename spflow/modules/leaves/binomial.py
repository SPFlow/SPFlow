import torch
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta.data import Scope
from spflow.modules.leaves.leaf import LeafModule
from spflow.utils.leaves import init_parameter, _handle_mle_edge_cases
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded


class Binomial(LeafModule):
    """Binomial distribution leaf module for probabilistic circuits.

    Implements univariate Binomial distributions as leaf nodes in probabilistic
    circuits. Supports parameter learning through maximum likelihood estimation
    and efficient inference through PyTorch's built-in distributions.

    The Binomial distribution models the number of successes in a fixed number
    of independent Bernoulli trials, with probability mass function::

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
        num_repetitions: int = 1,
        total_count: Tensor | None = None,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        parameter_fn: nn.Module = None,
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
            parameter_fn: Optional neural network for parameter generation.
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
            parameter_fn=parameter_fn,
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

    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> dict[str, Tensor]:
        """Compute raw MLE estimates for binomial distribution (without broadcasting).

        Args:
            data: Input data tensor.
            weights: Weight tensor for each data point.
            bias_correction: Not used for Binomial (included for interface consistency).

        Returns:
            Dictionary with 'probs' estimate (shape: out_features).
        """
        normalized_weights = weights / weights.sum(dim=0)

        n_total = normalized_weights.sum(dim=0) * self.total_count
        n_success = (normalized_weights * data).sum(0)
        probs_est = n_success / n_total

        # Handle edge cases (NaN, out of bounds) before broadcasting
        probs_est = _handle_mle_edge_cases(probs_est, lb=0.0, ub=1.0)

        return {"probs": probs_est}

    def _set_mle_parameters(self, params_dict: dict[str, Tensor]) -> None:
        """Set MLE-estimated parameters for Binomial distribution.

        Explicitly handles the parameter type:
        - probs: Property with setter, calls property setter which updates _logits

        Args:
            params_dict: Dictionary with 'probs' parameter value.
        """
        self.probs = params_dict["probs"]  # Uses property setter
