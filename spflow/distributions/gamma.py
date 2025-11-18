import torch
from torch import Tensor, nn

from spflow.distributions.base import Distribution
from spflow.utils.leaves import LogSpaceParameter, validate_all_or_none, init_parameter, _handle_mle_edge_cases


class Gamma(Distribution):
    """Gamma distribution for modeling positive-valued continuous data.

    Parameterized by shape α > 0 and rate β > 0 (both stored in log-space).

    Attributes:
        alpha: Shape parameter α (LogSpaceParameter).
        beta: Rate parameter β (LogSpaceParameter).
    """

    alpha = LogSpaceParameter("alpha")
    beta = LogSpaceParameter("beta")

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

        self.log_alpha = nn.Parameter(torch.empty_like(alpha))
        self.log_beta = nn.Parameter(torch.empty_like(beta))

        self.alpha = alpha.clone().detach()
        self.beta = beta.clone().detach()

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
