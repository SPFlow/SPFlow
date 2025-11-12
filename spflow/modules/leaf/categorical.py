import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule, init_parameter, parse_leaf_args


class Categorical(LeafModule):
    def __init__(
        self,
        scope: Scope,
        out_channels: int = None,
        num_repetitions: int = None,
        K: int = None,
        p: Tensor = None,
    ):
        """
        Initialize a Categorical distribution leaf module.

        Args:
            scope (Scope): The scope of the distribution.
            out_channels (int, optional): The number of output channels. If None, it is determined by the parameter tensor.
            num_repetitions (int, optional): The number of repetitions for the leaf module.
            K (int, optional): The number of categories.
            p (Tensor, optional): The probability tensor.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[p], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape
        self.K = K

        # Initialize parameter
        p = init_parameter(param=p, event_shape=(*event_shape, K), init=torch.rand)

        self.log_p = nn.Parameter(torch.empty_like(p))  # initialize empty, set with setter in next line
        self.p = p.clone().detach()

    @property
    def p(self) -> Tensor:
        """Returns the probabilities."""
        return self.log_p.exp()

    @p.setter
    def p(self, p):
        """Set the probabilities."""
        # Keep manual setter since descriptor helpers cannot enforce simplex normalization.
        # project auxiliary parameter onto actual parameter range
        if not torch.isfinite(p).all():
            raise ValueError(f"Values for 'p' must be finite, but was: {p}")

        if torch.any(p < 0.0) or torch.any(p > 1.0):
            raise ValueError(f"Value for 'p' must be in [0.0, 1.0], but was: {p}")

        # make sure that p adds up to 1
        p = p / p.sum(-1, keepdim=True)

        self.log_p.data = p.log()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying torch distribution object."""
        return torch.distributions.Categorical(self.p)

    @property
    def _supported_value(self):
        """Returns the supported values of the distribution."""
        return 1

    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Estimate categorical probabilities for each feature and assign parameter.

        Args:
            data: Scope-filtered data of shape (batch_size, num_scope_features).
            weights: Normalized weights of shape (batch_size, 1, ...).
            bias_correction: Not used for Categorical (included for template consistency).
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
        # Broadcast to event_shape (num_features, out_channels, K) and assign
        # p setter handles normalization and clamping
        self.p = self._broadcast_to_event_shape(p_est)

    def params(self) -> dict[str, Tensor]:
        """Returns the parameters of the distribution."""
        return {"p": self.p}
