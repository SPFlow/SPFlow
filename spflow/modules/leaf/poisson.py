import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule, LogSpaceParameter
from spflow.utils.leaf import parse_leaf_args, init_parameter


class Poisson(LeafModule):
    rate = LogSpaceParameter("rate")

    def __init__(
        self, scope: Scope, out_channels: int = None, num_repetitions: int = None, rate: Tensor = None
    ):
        r"""
        Initialize a Poisson distribution leaf module.

        Args:
            scope: Scope object specifying the scope of the distribution.
            out_channels: The number of output channels. If None, it is determined by the parameter tensor.
            num_repetitions: The number of repetitions for the leaf module.
            rate: Tensor representing the rate parameters (:math:`\lambda`) of the Poisson distributions.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[rate], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        rate = init_parameter(param=rate, event_shape=event_shape, init=torch.ones)

        self.log_rate = nn.Parameter(
            torch.empty_like(rate)
        )  # initialize empty, set with descriptor in next line
        self.rate = rate.clone().detach()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.Poisson(self.rate)

    @property
    def _supported_value(self):
        return 0

    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Estimate Poisson rate parameter and assign.

        Args:
            data: Scope-filtered data of shape (batch_size, num_scope_features).
            weights: Normalized weights of shape (batch_size, 1, ...).
            bias_correction: Not used for Poisson (included for template consistency).
        """
        n_total = weights.sum()

        rate_est = (weights * data).sum(dim=0) / n_total
        # Broadcast to event_shape and assign - LogSpaceParameter ensures positivity
        self.rate = self._broadcast_to_event_shape(rate_est)

    def params(self) -> dict[str, Tensor]:
        return {"rate": self.rate}
