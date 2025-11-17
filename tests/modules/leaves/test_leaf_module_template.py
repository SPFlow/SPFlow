import torch
from torch import nn

from spflow.meta import Scope
from spflow.modules.leaves.base import LeafModule
from utils.leaves import LogSpaceParameter


class DummyLeaf(LeafModule):
    """Minimal leaves to exercise the MLE template plumbing."""

    scale = LogSpaceParameter("scale")

    def __init__(self, scope: Scope, out_channels: int = 1):
        event_shape = torch.Size([len(scope.query), out_channels])
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        loc = torch.zeros(event_shape)
        scale = torch.ones(event_shape)

        self.loc = nn.Parameter(loc)
        self.log_scale = nn.Parameter(torch.empty_like(scale))
        self.scale = scale.clone().detach()
        self.last_data: torch.Tensor | None = None
        self.last_weights: torch.Tensor | None = None

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Normal(self.loc, self.scale)

    @property
    def _supported_value(self):
        return 0.0

    def params(self) -> dict[str, torch.Tensor]:
        return {"loc": self.loc, "scale": self.scale}

    def _mle_compute_statistics(
        self, data: torch.Tensor, weights: torch.Tensor, bias_correction: bool
    ) -> None:
        """Compute statistics and assign parameters directly."""
        self.last_data = data
        self.last_weights = weights
        mean_est = (weights * data).sum(dim=0) / weights.sum()
        scale_est = torch.full_like(mean_est, 0.5)
        # Assign directly - LogSpaceParameter handles log-space storage
        self.loc.data = self._broadcast_to_event_shape(mean_est)
        self.scale = self._broadcast_to_event_shape(scale_est)


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

    expected_loc = torch.ones_like(leaf.loc) * 1.25  # Weighted mean after ignoring NaNs.
    torch.testing.assert_close(leaf.loc.detach(), expected_loc)
    torch.testing.assert_close(leaf.scale, torch.ones_like(leaf.scale) * 0.5)
