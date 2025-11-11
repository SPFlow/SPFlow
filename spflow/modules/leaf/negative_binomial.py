import torch
from torch import Tensor, nn
from typing import Optional, Callable

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args, init_parameter
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded
from spflow.utils.cache import Cache


class NegativeBinomial(LeafModule):
    def __init__(
        self, scope: Scope, n: Tensor, out_channels: int = None, num_repetitions: int = None, p: Tensor = None
    ):
        r"""
        Initialize a NegativeBinomial distribution leaf module.

        Args:
            scope: Scope object specifying the scope of the distribution.
            out_channels: The number of output channels. If None, it is determined by the parameter tensors.
            num_repetitions: The number of repetitions for the leaf module.
            n: Tensor representing the numbers of  successes (greater or equal to 0).
            p: Tensor containing the success probability (:math:`p`) of each trial in :math:`[0,1]`.
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[p], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape

        p = init_parameter(param=p, event_shape=event_shape, init=torch.rand)

        if not torch.isfinite(n).all() or n.lt(0.0).any():
            raise ValueError(f"Values for 'n' must be finite and greater than 0, but was: {n}")

        n = torch.broadcast_to(n, event_shape).clone()
        self.register_buffer("_n", n)
        self.n = n
        self.log_p = nn.Parameter(torch.empty_like(p))  # initialize empty, set with setter in next line
        self.p = p.clone().detach()

    @property
    def p(self) -> Tensor:
        """Returns the success proability."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.log_p, lb=0.0, ub=1.0)  # type: ignore

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

        if p.lt(0.0).any() or p.gt(1.0).any() or not torch.isfinite(p).all():
            raise ValueError(
                f"Value of 'p' for 'Binomial' distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        self.log_p.data = proj_bounded_to_real(p, lb=0.0, ub=1.0)  # type: ignore

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

        if torch.any(n < 1.0) or not torch.isfinite(n).any():  # ToDo is 0 a valid value for n?
            raise ValueError(
                f"Value of 'n' for 'Binomial' distribution must to be greater than 0, but was: {n}"
            )

        self._n = n

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return torch.distributions.NegativeBinomial(total_count=self.n, probs=self.p)

    @property
    def _supported_value(self):
        return 0

    def maximum_likelihood_estimation(
        self,
        data: Tensor,
        weights: Optional[Tensor] = None,
        bias_correction: bool = True,
        nan_strategy: Optional[str | Callable] = None,
        check_support: bool = True,
        cache: Cache | None = None,
        preprocess_data: bool = True,
    ) -> None:
        """
        Maximum likelihood estimation for the Negative Binomial distribution.

        Note: The PyTorch implementation models this as the number of successful independent and identical Bernoulli
        trials before n failures are achieved. Scipy models this as the number of failures before n successes are achieved,
        which aligns with common statistical definitions. Therefore, the PyTorch MLE of p_est is 1 - p_est of Scipy.

        Args:
            data: The input data tensor.
            weights: Optional weights tensor. If None, uniform weights are created.
            bias_correction: If True, apply bias correction to the estimate.
            nan_strategy: Optional string or callable specifying how to handle missing data.
            check_support: Boolean value indicating whether to check data support.
            cache: Optional cache dictionary.
            preprocess_data: Boolean indicating whether to select relevant data for scope.
        """
        # Always select data relevant to this scope (same as log_likelihood does)
        data = data[:, self.scope.query]
        # Prepare weights using helper method
        weights = self._prepare_mle_weights(data, weights)

        # total (weighted) number of instances times number of trials per instance
        n_total = weights.sum() * self.n

        if bias_correction:
            n_total -= 1

        # count (weighted) number of total successes
        n_success = (weights * data).sum(0)

        # estimate (weighted) success probability
        # For NegativeBinomial, the MLE doesn't use the standard broadcast helper
        # because of the specific shape handling needed
        if len(self.event_shape) == 3:
            p_est = 1 - n_total / (n_success.unsqueeze(1).unsqueeze(2) + n_total)
        else:
            p_est = 1 - n_total / (n_success.unsqueeze(1) + n_total)

        # Handle edge cases using helper method
        p_est = self._handle_mle_edge_cases(p_est)

        # set parameters of leaf node
        self.p = p_est

    def params(self):
        return {"n": self.n, "p": self.p}
