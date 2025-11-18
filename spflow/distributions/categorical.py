import torch
from torch import Tensor, nn

from spflow.distributions.base import Distribution
from spflow.utils.leaves import init_parameter, SimplexParameter


class Categorical(Distribution):
    """Categorical distribution for discrete choice over K categories.

    Parameterized by probabilities p over K categories (normalized to form a simplex).

    Attributes:
        p: Categorical probabilities (normalized, shape includes extra dimension for K).
        K: Number of categories.
    """

    p = SimplexParameter("p")

    def __init__(self, p: Tensor = None, K: int = None, event_shape: tuple[int, ...] = None):
        """Initialize Categorical distribution.

        Args:
            p: Probability tensor of shape (*event_shape, K).
            K: Number of categories.
            event_shape: The shape of the event (without K dimension). If None, inferred from p.
        """
        if event_shape is None:
            # p has shape (*event_shape, K), so we need to extract event_shape
            if p is not None:
                event_shape = p.shape[:-1]
            else:
                raise ValueError("Either event_shape or p must be provided")
        super().__init__(event_shape=event_shape)

        self.K = K

        # Normalize probabilities if provided to ensure they sum to 1 along last dimension
        # Only normalize if probabilities are valid (non-negative, <= 1.0, and finite) but don't sum to 1
        if p is not None:
            if torch.all(p >= 0.0) and torch.all(p <= 1.0) and torch.isfinite(p).all():
                # Check if probabilities already sum to 1 (within tolerance)
                sums = p.sum(dim=-1, keepdim=True)
                if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6):
                    p = p / sums

        # Initialize parameter with K categories
        p = init_parameter(
            param=p, event_shape=(*event_shape, K), init=lambda shape: torch.rand(shape).softmax(dim=-1)
        )

        self.logits_p = nn.Parameter(torch.empty_like(p))  # initialize empty, set with setter in next line
        self.p = p.clone().detach()

    @property
    def _supported_value(self):
        """Fallback value for unsupported data."""
        return 1

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying Categorical distribution."""
        return torch.distributions.Categorical(self.p)

    def params(self) -> dict[str, Tensor]:
        """Returns distribution parameters."""
        return {"p": self.p}

    def _mle_update_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Estimate categorical probabilities for each category.

        Args:
            data: Scope-filtered data.
            weights: Normalized weights.
            bias_correction: Not used for Categorical (included for interface consistency).
        """
        weights_flat = weights.reshape(weights.shape[0], -1)[:, 0]
        n_total = weights_flat.sum()

        if self.K is not None:
            num_categories = self.K
        else:
            finite_values = data[~torch.isnan(data)]
            num_categories = int(finite_values.max().item()) + 1 if finite_values.numel() else 1

        p_entries: list[Tensor] = []
        for column in range(data.shape[1]):
            cat_probs: list[Tensor] = []
            for cat in range(num_categories):
                cat_mask = (data[:, column] == cat).float()
                cat_est = torch.sum(weights_flat * cat_mask) / n_total
                cat_probs.append(cat_est)
            p_entries.append(torch.stack(cat_probs))

        p_est = torch.stack(p_entries, dim=0).to(data.device)

        # Broadcast to event_shape and assign
        # p setter handles normalization and clamping
        self.p = self._broadcast_to_event_shape(p_est)
