import torch
from torch import Tensor, nn

from spflow.distributions.distribution import Distribution
from spflow.meta.data import FeatureContext, FeatureTypes
from spflow.utils.leaf import init_parameter
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded


class Binomial(Distribution):
    def __init__(self, n: Tensor, p: Tensor = None, event_shape: tuple[int, ...] = None):
        r"""Initializes ``Binomial`` leaf node.

        Args:
            scope: Scope object specifying the scope of the distribution.
            n: Tensor containing the number (:math:`n`) of total trials.
            p: Tensor containing the success probability (:math:`p`) of each trial in :math:`[0,1]`.
            n_out: Number of nodes per scope. Only relevant if mean and std is None.

        """
        if event_shape is None:
            event_shape = p.shape
        super().__init__(event_shape=event_shape)
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
        return torch.distributions.Binomial(total_count=self.n, probs=self.p)

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        # leaf only has one output
        if len(signatures) != 1:
            return False

        # get single output signature
        feature_ctx = signatures[0]
        domains = feature_ctx.get_domains()

        # leaf is a single non-conditional univariate node
        if (
            len(domains) != 1
            or len(feature_ctx.scope.query) != len(domains)
            or len(feature_ctx.scope.evidence) != 0
        ):
            return False

        # leaf is a discrete Binomial distribution
        # NOTE: only accept instances of 'FeatureTypes.Binomial', otherwise required parameter 'n' is not specified. Reject 'FeatureTypes.Discrete' for the same reason.
        if not isinstance(domains[0], FeatureTypes.Binomial):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Binomial":
        if not cls.accepts(signatures):
            raise ValueError(
                f"'Binomial' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if isinstance(domain, FeatureTypes.Binomial):
            n = domain.n
            p = domain.p
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Binomial' that was not caught during acception checking."
            )

        return Binomial(n=n, p=p)

    def maximum_likelihood_estimation(self, data: Tensor, weights: Tensor = None, bias_correction=True):
        """
        Maximum likelihood estimation for the Binomial distribution. bias_correction is ignored since p = x / n is unbiased.
        """
        if weights is None:
            _shape = (data.shape[0], *([1] * (data.dim() - 1)))  # (batch, 1, 1, ...) for broadcasting
            weights = torch.ones(_shape, device=data.device)

        # normalize weights to sum to n_samples
        weights /= weights.sum()

        # total (weighted) number of instances times number of trials per instance
        n_total = weights.sum() * self.n

        # count (weighted) number of total successes
        n_success = (weights * data).sum(0)

        # estimate (weighted) success probability
        p_est = n_success.unsqueeze(1) / n_total

        # edge case (if all values are the same, not enough samples or very close to each other)
        if torch.any(zero_mask := torch.isclose(p_est, torch.tensor(0.0))):
            p_est[zero_mask] = torch.tensor(1e-8)
        elif torch.any(one_mask := torch.isclose(p_est, torch.tensor(1.0))):
            p_est[one_mask] = torch.tensor(1 - 1e-8)

        # set parameters of leaf node
        self.p = p_est

    def params(self) -> dict[str, Tensor]:
        return {"n": self.n, "p": self.p}

    def marginalized_params(self, indices: list[int]) -> dict[str, Tensor]:
        return {"n": self.n[indices], "p": self.p[indices]}
