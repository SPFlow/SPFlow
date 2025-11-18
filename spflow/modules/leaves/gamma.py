import torch
from torch import Tensor, nn

from .distribution import Distribution
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import validate_all_or_none, init_parameter, _handle_mle_edge_cases, parse_leaf_args


class GammaDistribution(Distribution):
    """Gamma distribution for modeling positive-valued continuous data.

    Parameterized by shape α > 0 and rate β > 0 (both stored in log-space for numerical stability).
    """

    def __init__(self, alpha: Tensor = None, beta: Tensor = None, event_shape: tuple[int, ...] = None):
        """Initialize Gamma distribution.

        Args:
            alpha: Shape parameter α > 0.
            beta: Rate parameter β > 0.
            event_shape: The shape of the event. If None, it is inferred from parameter shapes.
        """
        if event_shape is None:
            event_shape = alpha.shape
        super().__init__(event_shape=event_shape)

        validate_all_or_none(alpha=alpha, beta=beta)

        alpha = init_parameter(param=alpha, event_shape=event_shape, init=torch.rand)
        beta = init_parameter(param=beta, event_shape=event_shape, init=torch.rand)

        # Validate alpha and beta at initialization
        if not (alpha > 0).all():
            raise ValueError("Alpha must be strictly positive")
        if not torch.isfinite(alpha).all():
            raise ValueError("Alpha must be finite")
        if not (beta > 0).all():
            raise ValueError("Beta must be strictly positive")
        if not torch.isfinite(beta).all():
            raise ValueError("Beta must be finite")

        self.log_alpha = nn.Parameter(torch.log(alpha))
        self.log_beta = nn.Parameter(torch.log(beta))

    @property
    def alpha(self) -> Tensor:
        """Shape parameter in natural space (read via exp of log_alpha)."""
        return torch.exp(self.log_alpha)

    @alpha.setter
    def alpha(self, value: Tensor) -> None:
        """Set shape parameter (stores as log_alpha, no validation after init)."""
        self.log_alpha.data = torch.log(
            torch.as_tensor(value, dtype=self.log_alpha.dtype, device=self.log_alpha.device)
        )

    @property
    def beta(self) -> Tensor:
        """Rate parameter in natural space (read via exp of log_beta)."""
        return torch.exp(self.log_beta)

    @beta.setter
    def beta(self, value: Tensor) -> None:
        """Set rate parameter (stores as log_beta, no validation after init)."""
        self.log_beta.data = torch.log(
            torch.as_tensor(value, dtype=self.log_beta.dtype, device=self.log_beta.device)
        )

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return 1.0

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying Gamma distribution."""
        return torch.distributions.Gamma(self.alpha, self.beta)

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"alpha": self.alpha, "beta": self.beta}

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute MLE for shape α and rate β parameters.

        Uses moment-matching equations to estimate parameters with optional bias correction.

        Args:
            data: Scope-filtered data.
            weights: Normalized sample weights.
            bias_correction: Whether to apply bias correction.
        """
        n_total = weights.sum()

        data_log = data.log()
        mean_xlnx = (weights * data_log * data).sum(dim=0) / n_total
        mean_x = (weights * data).sum(dim=0) / n_total
        mean_ln_x = (weights * data_log).sum(dim=0) / n_total

        theta_est = mean_xlnx - mean_x * mean_ln_x
        alpha_est = mean_x / theta_est
        beta_est = 1 / theta_est

        if bias_correction:
            alpha_est = alpha_est - 1 / n_total * (
                3 * alpha_est
                - 2 / 3 * (alpha_est / (1 + alpha_est))
                - 4 / 5 * (alpha_est / (1 + alpha_est) ** 2)
            )
            beta_est = beta_est * ((n_total - 1) / n_total)

        # Handle edge cases before broadcasting
        alpha_est = _handle_mle_edge_cases(alpha_est, lb=0.0)
        beta_est = _handle_mle_edge_cases(beta_est, lb=0.0)

        # Broadcast to event_shape and assign - LogSpaceParameter ensures positivity
        self.alpha = self._broadcast_to_event_shape(alpha_est)
        self.beta = self._broadcast_to_event_shape(beta_est)


class Gamma(LeafModule):
    """Gamma distribution leaf for modeling positive-valued continuous data.

    Parameterized by shape α > 0 and rate β > 0 (both stored in log-space).

    Attributes:
        alpha: Shape parameter α (LogSpaceParameter).
        beta: Rate parameter β (LogSpaceParameter).
        distribution: Underlying torch.distributions.Gamma.
    """

    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        alpha: Tensor = None,
        beta: Tensor = None,
    ):
        """Initialize Gamma distribution leaf.

        Args:
            scope: Variable scope.
            out_channels: Number of output channels (inferred from params if None).
            num_repetitions: Number of repetitions.
            alpha: Shape parameter α > 0.
            beta: Rate parameter β > 0.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[alpha, beta], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape
        self._distribution = GammaDistribution(alpha=alpha, beta=beta, event_shape=event_shape)
