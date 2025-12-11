from typing import Callable, Optional

import torch
from torch import nn

from spflow.meta import Scope
from spflow.modules.leaves.leaf import LeafModule


class DummyLeaf(LeafModule):
    """Minimal leaves to exercise the MLE template plumbing."""

    @property
    def _torch_distribution_class(self) -> type[torch.distributions.Distribution]:
        return torch.distributions.Normal

    def __init__(self, scope: Scope, out_channels: int = 1):
        event_shape = torch.Size([len(scope.query), out_channels])
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        mean = torch.zeros(event_shape)
        std = torch.ones(event_shape)

        self.mean = nn.Parameter(mean)
        self.log_std = nn.Parameter(torch.log(std))
        self.last_data: Optional[torch.Tensor] = None
        self.last_weights: Optional[torch.Tensor] = None

    @property
    def std(self) -> torch.Tensor:
        """Standard deviation in natural space (read via exp of log_std)."""
        return torch.exp(self.log_std)

    @std.setter
    def std(self, value: torch.Tensor) -> None:
        """Set standard deviation (stores as log_std, no validation after init)."""
        self.log_std.data = torch.log(
            torch.as_tensor(value, dtype=self.log_std.dtype, device=self.log_std.device)
        )

    @property
    def _supported_value(self):
        return 0.0

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Normal(self.mean, self.std)

    def params(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}

    def _mle_update_statistics(
        self, data: torch.Tensor, weights: torch.Tensor, bias_correction: bool
    ) -> None:
        """Compute mean normally but set std to fixed test value."""
        # Compute mean
        n_total = weights.sum()
        mean_est = (weights * data).sum(0) / n_total
        self.mean.data = self._broadcast_to_event_shape(mean_est)

        # Set std to fixed test value
        std_est = torch.full_like(mean_est, 0.5)
        self.std = self._broadcast_to_event_shape(std_est)

    def _compute_parameter_estimates(
        self, data: torch.Tensor, weights: torch.Tensor, bias_correction: bool
    ) -> dict[str, torch.Tensor]:
        """Compute raw MLE parameter estimates without broadcasting."""
        n_total = weights.sum()
        mean_est = (weights * data).sum(0) / n_total
        std_est = torch.full_like(mean_est, 0.5)
        return {"mean": mean_est, "std": std_est}

    def maximum_likelihood_estimation(
        self,
        data: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        bias_correction: bool = False,
        nan_strategy: Optional[Callable] = "ignore",
        cache: Optional[dict] = None,
    ) -> None:
        """Override MLE to track data for testing."""
        # Call parent to do the normal MLE workflow
        super().maximum_likelihood_estimation(
            data=data,
            weights=weights,
            bias_correction=bias_correction,
            nan_strategy=nan_strategy,
            cache=cache,
        )

        # After MLE is done, track the data that was used
        # We need to re-prepare the data to get what was actually used
        data_prepared, weights_prepared = self._prepare_mle_data(
            data=data,
            weights=weights,
            nan_strategy=nan_strategy,
        )
        self.last_data = data_prepared
        self.last_weights = weights_prepared
