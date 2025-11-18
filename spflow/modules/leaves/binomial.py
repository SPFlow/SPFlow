from torch import Tensor

from spflow.distributions.binomial import Binomial as BinomialDistribution
from spflow.meta.data import Scope
from spflow.modules.leaves.base import LeafModule
from spflow.utils.leaves import parse_leaf_args


class Binomial(LeafModule):
    """Binomial distribution leaf module for probabilistic circuits.

    Implements univariate Binomial distributions as leaf nodes in probabilistic
    circuits. Supports parameter learning through maximum likelihood estimation
    and efficient inference through PyTorch's built-in distributions.

    The Binomial distribution models the number of successes in a fixed number
    of independent Bernoulli trials, with probability mass function:
        P(X = k | n, p) = C(n, k) * p^k * (1-p)^(n-k)

    where n is the number of trials (fixed), p is the success probability (learnable),
    and k is the number of successes (0 ≤ k ≤ n).

    Attributes:
        p: Success probability parameter(s) in [0, 1] (BoundedParameter).
        n: Number of trials parameter(s), non-negative integers (fixed buffer).
        distribution: Underlying torch.distributions.Binomial.
    """

    def __init__(
        self, scope: Scope, n: Tensor, out_channels: int = None, num_repetitions: int = None, p: Tensor = None
    ):
        """Initialize Binomial distribution leaf module.

        Args:
            scope: Scope object specifying the scope of the distribution.
            n: Tensor containing the number (n) of total trials (fixed, non-negative).
            out_channels: Number of output channels (inferred from p if None).
            num_repetitions: Number of repetitions for the distribution.
            p: Tensor containing the success probability (p) of each trial in [0, 1].
        """
        event_shape = parse_leaf_args(
            scope=scope, out_channels=out_channels, params=[p], num_repetitions=num_repetitions
        )
        super().__init__(scope, out_channels=event_shape[1])
        self._event_shape = event_shape
        self._distribution = BinomialDistribution(n=n, p=p, event_shape=event_shape)
