import torch
from torch import Tensor, nn

from spflow.distributions.distribution import Distribution
from spflow.exceptions import InvalidParameterCombinationError

from spflow.meta.data.meta_type import MetaType
from spflow.utils.leaf import init_parameter


class Normal(Distribution):


    def __init__(self, mean: Tensor = None, std: Tensor = None, event_shape: tuple[int, ...] = None):
        r"""

        Args:
            mean: Tensor containing the mean (:math:`\mu`) of the distribution.
            std: Tensor containing the standard deviation (:math:`\sigma`) of the distribution.
            event_shape: The shape of the event. If None, it is inferred from the shape of the parameter tensor.
        """
        if event_shape is None:
            event_shape = mean.shape
        super().__init__(event_shape=event_shape)
        if not (mean is None and std is None) ^ (mean is not None and std is not None):
            raise InvalidParameterCombinationError("Either mean and std must be specified or neither.")

        mean = init_parameter(param=mean, event_shape=event_shape, init=torch.randn)
        std = init_parameter(param=std, event_shape=event_shape, init=torch.rand)

        self.mean = nn.Parameter(mean)
        self.log_std = nn.Parameter(torch.empty_like(std))  # initialize empty, set with setter in next line
        self.std = std.clone().detach()

    @property
    def std(self) -> Tensor:
        """Returns the standard deviation."""
        return self.log_std.exp()
        #return self.log_std.clamp(min=-20.0, max=2.0).exp()

    @std.setter
    def std(self, std):
        """Set the standard deviation."""
        # project auxiliary parameter onto actual parameter range
        if not torch.isfinite(std).all():
            raise ValueError(f"Values for 'std' must be finite, but was: {std}")

        if torch.all(std <= 0.0):
            raise ValueError(f"Value for 'std' must be greater than 0.0, but was: {std}")

        self.log_std.data = std.log()
        #self.log_std.data = (std + 1e-6).log()

    def mode(self) -> Tensor:
        return self.mean

    @property
    def _supported_value(self):
        return 0.0

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Normal(self.mean, self.std)

    def maximum_likelihood_estimation(self, data: Tensor, weights: Tensor = None, bias_correction=True):
        # TODO: make some assertions about the event_shape and the data
        if weights is None:
            _shape = (data.shape[0], *([1] * (data.dim() - 1)))  # (batch, 1, 1, ...) for broadcasting
            weights = torch.ones(_shape, device=data.device)

        # total (weighted) number of instances
        n_total = weights.sum()

        # calculate mean and standard deviation from data
        mean_est = (weights * data).sum(0) / n_total
        std_est = (weights * (data - mean_est) ** 2).sum(0)

        if bias_correction:
            std_est = torch.sqrt((weights * torch.pow(data - mean_est, 2)).sum(0) / (n_total - 1))
        else:
            std_est = torch.sqrt((weights * torch.pow(data - mean_est, 2)).sum(0) / n_total)


        # edge case (if all values are the same, not enough samples or very close to each other)
        if torch.any(zero_mask := torch.isclose(std_est, torch.tensor(0.0))):
            std_est[zero_mask] = torch.tensor(1e-8)
        if torch.any(nan_mask := torch.isnan(std_est)):
            std_est[nan_mask] = torch.tensor(1e-8)
            



        if len(self.event_shape) == 2:
            # Repeat mean and std
            mean_est = mean_est.unsqueeze(1).repeat(1, self.out_channels)
            std_est = std_est.unsqueeze(1).repeat(1, self.out_channels)
        if len(self.event_shape) == 3:
            # Repeat mean and std
            mean_est = mean_est.unsqueeze(1).unsqueeze(1).repeat(1, self.out_channels, self.num_repetitions)
            std_est = std_est.unsqueeze(1).unsqueeze(1).repeat(1, self.out_channels, self.num_repetitions)

        # set parameters of leaf node
        self.mean.data = mean_est
        self.std = std_est

    def params(self):
        return {"mean": self.mean, "std": self.std}
