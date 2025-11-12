import torch
from torch import Tensor, nn

from spflow.meta.data import Scope
from spflow.modules.leaves.leaf_module import LeafModule, BoundedParameter, init_parameter, parse_leaf_args


class Bernoulli(LeafModule):
    r"""
    Create a Bernoulli leaves module.
    """
    p = BoundedParameter("p", lb=0.0, ub=1.0)

    def __init__(self, scope: Scope, out_channels: int = None, num_repetitions: int = None, p: Tensor = None):
        r"""
        Args:
            scope (Scope): The scope of the leaves module.
            out_channels (int, optional): The number of output channels. If None, it is inferred from the shape of the parameter tensor.
            num_repetitions (int, optional): The number of repetitions for the leaves module.
            p (Tensor): PyTorch tensor representing the success probabilities of the Bernoulli distributions.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[p], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        p = init_parameter(param=p, event_shape=event_shape, init=torch.rand)

        self.log_p = nn.Parameter(torch.empty_like(p))  # initialize empty, set with setter in next line
        self.p = p.clone().detach()

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Returns the underlying torch distribution object."""
        return torch.distributions.Bernoulli(self.p)

    @property
    def _supported_value(self):
        """Returns the supported values of the distribution."""
        return 0.0

    def _mle_compute_statistics(self, data: Tensor, weights: Tensor, bias_correction: bool) -> None:
        """Compute Bernoulli probability estimate and assign parameter.

        Args:
            data: Scope-filtered data of shape (batch_size, num_scope_features).
            weights: Normalized weights of shape (batch_size, 1, ...).
            bias_correction: Not used for Bernoulli (included for template consistency).
        """
        n_total = weights.sum()
        n_success = (weights * data).sum(dim=0)
        p_est = n_success / n_total

        # Broadcast to event_shape and assign directly
        # BoundedParameter descriptor handles clamping to [0, 1]
        self.p = self._broadcast_to_event_shape(p_est)

    def params(self) -> dict[str, Tensor]:
        """Returns the parameters of the distribution."""
        return {"p": self.p}
