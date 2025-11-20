import torch
from torch import Tensor, nn

from spflow.exceptions import InvalidParameterCombinationError
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import init_parameter, _handle_mle_edge_cases
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded


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
        out_channels: int = None,
        num_repetitions: int = 1,
        parameter_network: nn.Module = None,
        validate_args: bool | None = True,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
    ):
        """Initialize Bernoulli distribution.

        Args:
            scope: Variable scope (Scope, int, or list[int]).
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions (for 3D event shapes).
            parameter_network: Optional neural network for parameter generation.
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
            parameter_network=parameter_network,
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

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"logits": self.logits}

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for success probability p.

        For Bernoulli distribution, the MLE is the weighted proportion of successes.

        Args:
            data: Scope-filtered data.
            weights: Normalized sample weights.
            bias_correction: Not used for Bernoulli.
        """
        n_total = weights.sum()
        n_success = (weights * data).sum(dim=0)
        p_est = n_success / n_total

        # Handle edge cases (NaN, zero, or near-zero p) before broadcasting
        p_est = _handle_mle_edge_cases(p_est, lb=0.0)
        self.probs = self._broadcast_to_event_shape(p_est)
