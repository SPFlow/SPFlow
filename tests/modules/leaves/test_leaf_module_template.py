from typing import Callable, Optional

import torch

from spflow.meta import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.distributions.normal import Normal as NormalDistribution


class _DummyNormalDistribution(NormalDistribution):
    """Helper distribution for testing that returns fixed std value."""

    def _mle_update_statistics(self, data: torch.Tensor, weights: torch.Tensor, bias_correction: bool) -> None:
        """Compute mean normally but set std to fixed test value."""
        # Compute mean
        n_total = weights.sum()
        mean_est = (weights * data).sum(0) / n_total
        self.mean.data = self._broadcast_to_event_shape(mean_est)

        # Set std to fixed test value
        std_est = torch.full_like(mean_est, 0.5)
        self.std = self._broadcast_to_event_shape(std_est)


class DummyLeaf(LeafModule):
    """Minimal leaves to exercise the MLE template plumbing."""

    def __init__(self, scope: Scope, out_channels: int = 1):
        event_shape = torch.Size([len(scope.query), out_channels])
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        mean = torch.zeros(event_shape)
        std = torch.ones(event_shape)

        self._distribution = _DummyNormalDistribution(mean=mean, std=std, event_shape=event_shape)
        self.last_data: Optional[torch.Tensor] = None
        self.last_weights: Optional[torch.Tensor] = None

    @property
    def mean(self):
        """Delegate to distribution's mean."""
        return self._distribution.mean

    @property
    def std(self):
        """Delegate to distribution's std."""
        return self._distribution.std

    @property
    def distribution(self) -> _DummyNormalDistribution:
        return self._distribution

    @property
    def _supported_value(self):
        return 0.0

    def params(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}

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


def test_leaf_module_template_handles_nan_strategy_and_descriptors():
    scope = Scope([0])
    leaf = DummyLeaf(scope)

    data = torch.tensor([[0.0], [1.0], [float("nan")], [2.0]])
    weights = torch.tensor([1.0, 1.0, 1.0, 2.0])

    leaf.maximum_likelihood_estimation(
        data,
        weights=weights,
        nan_strategy="ignore",
    )

    assert leaf.last_data is not None
    assert leaf.last_weights is not None
    # NaN row should be removed once by the template helper.
    assert leaf.last_data.shape[0] == 3
    assert torch.all(torch.isfinite(leaf.last_data))

    expected_mean = torch.ones_like(leaf.mean) * 1.25  # Weighted mean after ignoring NaNs.
    torch.testing.assert_close(leaf.mean.detach(), expected_mean)
    torch.testing.assert_close(leaf.std, torch.ones_like(leaf.std) * 0.5)
