import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.leaves.leaf_module import (
    LeafModule,
    LogSpaceParameter,
    validate_all_or_none, init_parameter, parse_leaf_args,
)


class Gamma(LeafModule):
    alpha = LogSpaceParameter("alpha")
    beta = LogSpaceParameter("beta")

    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        alpha: Tensor = None,
        beta: Tensor = None,
    ):
        r"""
        Initialize a Gamma distribution leaves module.

        Args:
            scope: Scope object specifying the scope of the distribution.
            out_channels: The number of output channels. If None, it is determined by the parameter tensors.
            num_repetitions: The number of repetitions for the leaves module.
            alpha: Tensor representing the shape parameters (:math:`\alpha`) of the Gamma distributions, greater than 0.
            beta: Tensor representing the rate parameters (:math:`\beta`) of the Gamma distributions, greater than 0.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[alpha, beta], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        validate_all_or_none(alpha=alpha, beta=beta)

        alpha = init_parameter(param=alpha, event_shape=event_shape, init=torch.rand)
        beta = init_parameter(param=beta, event_shape=event_shape, init=torch.rand)

        self.log_alpha = nn.Parameter(torch.empty_like(alpha))
        self.log_beta = nn.Parameter(torch.empty_like(beta))

        self.alpha = alpha.clone().detach()
        self.beta = beta.clone().detach()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Gamma(self.alpha, self.beta)

    @property
    def _supported_value(self):
        return 1.0

    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Estimate Gamma alpha/beta parameters and assign.

        Args:
            data: Scope-filtered data of shape (batch_size, num_scope_features).
            weights: Normalized weights of shape (batch_size, 1, ...).
            bias_correction: Whether to apply bias correction for alpha and beta.
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

        # Broadcast to event_shape and assign - LogSpaceParameter ensures positivity
        self.alpha = self._broadcast_to_event_shape(alpha_est)
        self.beta = self._broadcast_to_event_shape(beta_est)

    def params(self) -> dict[str, Tensor]:
        return {"alpha": self.alpha, "beta": self.beta}
