import torch
from torch import Tensor, nn
from typing import Optional, Callable

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args, init_parameter
from spflow.exceptions import InvalidParameterCombinationError
from spflow.utils.cache import Cache


class Gamma(LeafModule):
    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        alpha: Tensor = None,
        beta: Tensor = None,
    ):
        r"""
        Initialize a Gamma distribution leaf module.

        Args:
            scope: Scope object specifying the scope of the distribution.
            out_channels: The number of output channels. If None, it is determined by the parameter tensors.
            num_repetitions: The number of repetitions for the leaf module.
            alpha: Tensor representing the shape parameters (:math:`\alpha`) of the Gamma distributions, greater than 0.
            beta: Tensor representing the rate parameters (:math:`\beta`) of the Gamma distributions, greater than 0.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[alpha, beta], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        if not ((alpha is None and beta is None) ^ (alpha is not None and beta is not None)):
            raise InvalidParameterCombinationError(
                "Either alpha and beta must be specified or neither."
            )

        alpha = init_parameter(param=alpha, event_shape=event_shape, init=torch.rand)
        beta = init_parameter(param=beta, event_shape=event_shape, init=torch.rand)

        self.log_alpha = nn.Parameter(torch.empty_like(alpha))
        self.log_beta = nn.Parameter(torch.empty_like(beta))

        self.alpha = alpha.clone().detach()
        self.beta = beta.clone().detach()

    @property
    def alpha(self) -> Tensor:
        """Returns alpha."""
        return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, alpha):
        """Set alpha."""
        # project auxiliary parameter onto actual parameter range
        if not torch.isfinite(alpha).all():
            raise ValueError(f"Values for 'alpha' must be finite, but was: {alpha}")

        if torch.any(alpha <= 0.0):
            raise ValueError(f"Value for 'alpha' must be greater than 0.0, but was: {alpha}")

        self.log_alpha.data = alpha.log()

    @property
    def beta(self) -> Tensor:
        """Returns beta."""
        return self.log_beta.exp()

    @beta.setter
    def beta(self, beta):
        """Set beta."""
        # project auxiliary parameter onto actual parameter range
        if not torch.isfinite(beta).all():
            raise ValueError(f"Values for 'beta' must be finite, but was: {beta}")

        if torch.any(beta <= 0.0):
            raise ValueError(f"Value for 'beta' must be greater than 0.0, but was: {beta}")

        self.log_beta.data = beta.log()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Gamma(self.alpha, self.beta)

    @property
    def _supported_value(self):
        return 1.0

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Optional[Tensor] = None,
        bias_correction: bool = True,
        nan_strategy: Optional[str | Callable] = None,
        check_support: bool = True,
        cache: Cache | None = None,
        preprocess_data: bool = True,
    ) -> None:
        """Maximum likelihood estimation for Gamma distribution parameters.

        Args:
            data: The input data tensor.
            weights: Optional weights tensor. If None, uniform weights are created.
            bias_correction: If True, apply bias correction to the estimate.
            nan_strategy: Optional string or callable specifying how to handle missing data.
            check_support: Boolean value indicating whether to check data support.
            cache: Optional cache dictionary.
            preprocess_data: Boolean indicating whether to select relevant data for scope.
        """
        # Always select data relevant to this scope (same as log_likelihood does)
        data = data[:, self.scope.query]
        # Prepare weights using helper method
        weights = self._prepare_mle_weights(data, weights)

        # total (weighted) number of instances
        n_total = weights.sum()

        # Computation of alpha and beta according to Wikipedia:
        # https://en.wikipedia.org/wiki/Gamma_distribution#Maximum_likelihood_estimation

        mean_xlnx = (weights * data.log() * data).sum(dim=0) / n_total
        mean_x = (weights * data).sum(dim=0) / n_total
        mean_ln_x = (weights * data.log()).sum(dim=0) / n_total

        theta_est = mean_xlnx - mean_x * mean_ln_x
        alpha_est = mean_x / theta_est
        beta_est = 1 / theta_est

        # Handle edge cases for beta
        if torch.any(zero_mask := torch.isclose(beta_est, torch.tensor(0.0))):
            beta_est[zero_mask] = torch.tensor(1e-8, device=data.device)
        if torch.any(nan_mask := torch.isnan(beta_est)):
            beta_est[nan_mask] = torch.tensor(1e-8)

        # Broadcast to event_shape using helper method
        alpha_est = self._broadcast_to_event_shape(alpha_est)
        beta_est = self._broadcast_to_event_shape(beta_est)

        if bias_correction:
            alpha_est = alpha_est - 1 / n_total * (
                3 * alpha_est
                - 2 / 3 * (alpha_est / (1 + alpha_est))
                - 4 / 5 * (alpha_est / (1 + alpha_est) ** 2)
            )
            beta_est = beta_est * ((n_total - 1) / n_total)  # Note that beta = 1 / theta on Wikipedia

        # set parameters of leaf node
        self.alpha = alpha_est
        self.beta = beta_est

    def params(self) -> dict[str, Tensor]:
        return {"alpha": self.alpha, "beta": self.beta}
