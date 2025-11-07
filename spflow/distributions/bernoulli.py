import torch
from torch import Tensor, nn

from spflow.distributions.distribution import Distribution

from spflow.meta.data.meta_type import MetaType
from spflow.utils.leaf import init_parameter


class Bernoulli(Distribution):
    def __init__(self, p: Tensor = None, event_shape: tuple[int, ...] = None):
        r"""
        Args:
            p: PyTorch tensor representing the success probabilities of the Bernoulli distributions
            event_shape: The shape of the event. If None, it is inferred from the shape of the parameter tensor.
        """
        if event_shape is None:
            event_shape = p.shape
        super().__init__(event_shape=event_shape)

        p = init_parameter(param=p, event_shape=event_shape, init=torch.rand)

        self.log_p = nn.Parameter(torch.empty_like(p))  # initialize empty, set with setter in next line
        self.p = p.clone().detach()

    @property
    def p(self) -> Tensor:
        """Returns the p parameters."""
        return torch.exp(self.log_p)  # torch.sigmoid(torch.exp(self.log_p))

    @p.setter
    def p(self, p):
        """Set the p parameters."""
        # project auxiliary parameter onto actual parameter range
        if not torch.isfinite(p).all():
            raise ValueError(f"Values for 'p' must be finite, but was: {p}")

        if torch.any(p < 0.0):  # all
            raise ValueError(f"Value for 'p' must not be smaller than 0.0, but was: {p}")

        if torch.any(p > 1.0):
            raise ValueError(f"Value for 'p' must not be smaller than 1.0, but was: {p}")

        self.log_p.data = p.log()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Bernoulli(self.p)

    @property
    def _supported_value(self):
        return 0.0

    def maximum_likelihood_estimation(self, data: Tensor, weights: Tensor = None, bias_correction=True):
        if weights is None:
            _shape = (data.shape[0], *([1] * (data.dim() - 1)))  # (batch, 1, 1, ...) for broadcasting
            weights = torch.ones(_shape, device=data.device)

        # total (weighted) number of instances
        n_total = weights.sum()

        # count (weighted) number of total successes
        n_success = (weights * torch.nan_to_num(data, nan=0.0)).sum(dim=0)

        # estimate (weighted) success probability
        p_est = n_success / n_total

        # edge case (if all values are the same, not enough samples or very close to each other)
        if torch.any(zero_mask := torch.isclose(p_est, torch.tensor(0.0))):
            p_est[zero_mask] = torch.tensor(1e-8)

        if torch.any(nan_mask := torch.isnan(p_est)):
            p_est[nan_mask] = torch.tensor(1e-8)

        if len(self.event_shape) == 2:
            # Repeat p
            p_est = p_est.unsqueeze(1).repeat(1, self.out_channels)
        if len(self.event_shape) == 3:
            # Repeat p
            p_est = p_est.unsqueeze(1).unsqueeze(1).repeat(1, self.out_channels, self.num_repetitions)

        # set parameters of leaf node
        self.p = p_est

    def params(self) -> dict[str, Tensor]:
        return {"p": self.p}
