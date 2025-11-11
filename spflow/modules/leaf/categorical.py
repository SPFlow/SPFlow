import torch
from torch import Tensor, nn
from typing import Optional, Callable

from spflow.meta.data import Scope
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.utils.leaf import parse_leaf_args, init_parameter
from spflow.utils.cache import Cache


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
        """Maximum likelihood estimation for Categorical distribution parameters.

        Args:
            data: The input data tensor.
            weights: Optional weights tensor. If None, uniform weights are created.
            bias_correction: Unused for Categorical, kept for API consistency.
            nan_strategy: Optional string or callable specifying how to handle missing data.
            check_support: Boolean value indicating whether to check data support.
            cache: Optional cache dictionary.
            preprocess_data: Boolean indicating whether to select relevant data for scope.
        """
        # Always select data relevant to this scope (same as log_likelihood does)
        data = data[:, self.scope.query]
        # Prepare weights using helper method
        weights = self._prepare_mle_weights(data, weights)

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

        # Handle edge cases using helper method
        p_est = self._handle_mle_edge_cases(p_est)

        # Broadcasting for channels and repetitions (special handling for Categorical)
        if len(self.event_shape) == 2:
            p_est = p_est.unsqueeze(1).repeat(1, self.out_channels, 1)

        if len(self.event_shape) == 3:
            p_est = p_est.unsqueeze(1).unsqueeze(2).repeat(1, self.out_channels, self.num_repetitions, 1)

        # set parameters of leaf node and make sure they add up to 1
        self.p = p_est.to(data.device)

    def params(self) -> dict[str, Tensor]:
        """Returns the parameters of the distribution."""
        return {"p": self.p}
