import torch
from torch import Tensor, nn

from spflow.distributions.distribution import Distribution

from spflow.meta.data.meta_type import MetaType
from spflow.utils.leaf import init_parameter


class Categorical(Distribution):
    def __init__(self, p: Tensor, K: int = None, event_shape: tuple[int, ...] = None):
        r"""

        Args:
            p: Tensor containing the event probabilities of the distribution. Has shape (event_shape, k) where k is the number of categories.
            K: The number of categories. If None, it is inferred from the shape of the parameter tensor.
            event_shape: The shape of the event. If None, it is inferred from the shape of the parameter tensor.
        """
        if event_shape is None:
            event_shape = p.shape[:2]
        super().__init__(event_shape=event_shape)

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
        return torch.distributions.Categorical(self.p)

    @property
    def _supported_value(self):
        return 1

    def maximum_likelihood_estimation(self, data: Tensor, weights: Tensor = None, bias_correction=True):
        if weights is None:
            _shape = (data.shape[0], *([1] * (data.dim() - 1)))  # (batch, 1, 1, ...) for broadcasting
            weights = torch.ones(_shape, device=data.device)

        # total (weighted) number of instances
        n_total = weights.sum()

        # count (weighted) number of total successes
        n_success = (weights * data).sum(dim=0)

        p_est = []
        for column in range(data.shape[1]):
            p_k_est = []
            for cat in range(len(torch.unique(data))):
                cat_indices = data[:, column] == cat
                cat_data = cat_indices.float()
                cat_est = torch.sum(weights[0] * cat_data)
                cat_est /= n_total
                p_k_est.append(cat_est)
            p_est.append(p_k_est)

        p_est = torch.tensor(p_est)

        # edge case (if all values are the same, not enough samples or very close to each other)
        if torch.any(zero_mask := torch.isclose(p_est, torch.tensor(0.0))):
            p_est[zero_mask] = torch.tensor(1e-8)
        if torch.any(nan_mask := torch.isnan(p_est)):
            p_est[nan_mask] = torch.tensor(1e-8)

        if len(self.event_shape) == 2:
            # Repeat mean and std
            p_est = p_est.unsqueeze(1).repeat(1, self.out_channels, 1)

        if len(self.event_shape) == 3:
            p_est = p_est.unsqueeze(1).unsqueeze(2).repeat(1, self.out_channels, self.num_repetitions, 1)

        # set parameters of leaf node and make sure they add up to 1
        self.p = p_est.to(data.device)

    def params(self) -> dict[str, Tensor]:
        return {"p": self.p}
