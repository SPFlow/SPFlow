import torch
from torch import Tensor, nn

from spflow.distributions.distribution import Distribution
from spflow.meta.data import FeatureContext, FeatureTypes
from spflow.utils.leaf import init_parameter


class NegativeBinomial(Distribution):
    def __init__(self, n: Tensor, p: Tensor = None, event_shape: tuple[int, ...] = None):
        r"""Initializes ``NegativeBinomial`` leaf node.

        Args:
            scope: Scope object specifying the scope of the distribution.
            n: Tensor representing the numbers of  successes (greater or equal to 0).
            p: Tensor containing the success probability (:math:`p`) of each trial in :math:`[0,1]`.
        """
        if event_shape is None:
            event_shape = p.shape
        super().__init__(event_shape=event_shape)
        p = init_parameter(param=p, event_shape=event_shape, init=torch.rand)

        self._n = torch.broadcast_to(n, event_shape).clone()
        self.log_p = nn.Parameter(torch.empty_like(p))  # initialize empty, set with setter in next line
        self.p = p.clone().detach()

    @property
    def p(self) -> Tensor:
        """Returns the success proability."""
        # project auxiliary parameter onto actual parameter range
        return torch.exp(self.log_p)  # type: ignore

    @p.setter
    def p(self, p: Tensor):
        r"""Sets the success probability.

        Args:
            p:
                Floating point representing the success probability in :math:`[0,1]`.

        Raises:
            ValueError: Invalid arguments.
        """
        if isinstance(p, float):
            p = Tensor(p)

        if p.lt(0.0).any() or p.ge(1.0).any() or not torch.isfinite(p).all():
            raise ValueError(
                f"Value of 'p' for 'NegativeBinomial' distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        self.log_p.data = torch.log(p)  # type: ignore

    @property
    def n(self) -> Tensor:
        """Returns the number of trials."""
        return self._n

    @n.setter
    def n(self, n: Tensor):
        r"""Sets the number of trials.

        Args:
            n:
                Floating point representing the number of trials.

        Raises:
            ValueError: Invalid arguments.
        """

        if n.lt(0.0).any() or not torch.isfinite(n):  # ToDo is 0 a valid value for n?
            raise ValueError(
                f"Value of 'n' for 'NegativeBinomial' distribution must to be greater than 0, but was: {n}"
            )

        self._n = n

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.NegativeBinomial(total_count=self.n, probs=self.p)

    def maximum_likelihood_estimation(self, data: Tensor, weights: Tensor = None, bias_correction=True):
        """
        Maximum likelihood estimation for the Negative Binomial distribution.

        Note: The PyTorch implementation models this as the number of successful independent and identical Bernoulli
        trials before n failures are achieved. Scipy models this as the number of failures before n successes are achieved,
        which aligns with common statistical definitions. Therefore, the PyTorch MLE of p_est is 1 - p_est of Scipy.
        """
        if weights is None:
            _shape = (data.shape[0], *([1] * (data.dim() - 1)))  # (batch, 1, 1, ...) for broadcasting
            weights = torch.ones(_shape, device=data.device)

        # total (weighted) number of instances times number of trials per instance
        n_total = weights.sum() * self.n

        if bias_correction:
            n_total -= 1

        # count (weighted) number of total successes
        n_success = (weights * data).sum(0)

        # estimate (weighted) success probability
        p_est = 1 - n_total / (n_success.unsqueeze(1) + n_total)

        # edge case (if all values are the same, not enough samples or very close to each other)
        if torch.any(zero_mask := torch.isclose(p_est, torch.tensor(0.0))):
            p_est[zero_mask] = torch.tensor(1e-8)
        if torch.any(nan_mask := torch.isnan(p_est)):
            p_est[nan_mask] = torch.tensor(1e-8)

        # set parameters of leaf node
        self.p = p_est

    def params(self):
        return {"n": self.n, "p": self.p}
