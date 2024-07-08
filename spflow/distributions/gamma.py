import torch
from torch import Tensor, nn

from spflow.distributions.distribution import Distribution
from spflow.meta.data import FeatureContext, FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.utils.leaf import init_parameter


class Gamma(Distribution):
    def __init__(self, alpha: Tensor = None, beta: Tensor = None, event_shape: tuple[int, ...] = None):
        r"""Initializes ``Gamma`` leaf node.

        Args:
            alpha: Tensor representing the shape parameters (:math:`\alpha`) of the Gamma distributions, greater than 0.
            beta: Tensor representing the rate parameters (:math:`\beta`) of the Gamma distributions, greater than 0.
            event_shape: Shape of the event space.
        """
        if event_shape is None:
            event_shape = alpha.shape
        super().__init__(event_shape=event_shape)
        assert (alpha is None and beta is None) ^ (
            alpha is not None and beta is not None
        ), "Either alpha and beta must be specified or neither."

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
            raise ValueError(f"Values for 'beta' must be finite, but was: {alpha}")

        if torch.all(alpha <= 0.0):
            raise ValueError(f"Value for 'beta' must be greater than 0.0, but was: {alpha}")

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

        if torch.all(beta <= 0.0):
            raise ValueError(f"Value for 'beta' must be greater than 0.0, but was: {beta}")

        self.log_beta.data = beta.log()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Gamma(self.alpha, self.beta)

    def maximum_likelihood_estimation(self, data: Tensor, weights: Tensor = None, bias_correction=True):
        if weights is None:
            _shape = (data.shape[0], *([1] * (data.dim() - 1)))  # (batch, 1, 1, ...) for broadcasting
            weights = torch.ones(_shape, device=data.device)

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

        # edge case (if all values are the same, not enough samples or very close to each other)
        if torch.any(zero_mask := torch.isclose(beta_est, torch.tensor(0.0))):
            beta_est[zero_mask] = torch.tensor(1e-8)
        if torch.any(nan_mask := torch.isnan(beta_est)):
            beta_est[nan_mask] = torch.tensor(1e-8)

        if len(self.event_shape) == 2:
            # Repeat alpha and beta
            alpha_est = alpha_est.unsqueeze(1).repeat(1, self.out_channels)
            beta_est = beta_est.unsqueeze(1).repeat(1, self.out_channels)

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
