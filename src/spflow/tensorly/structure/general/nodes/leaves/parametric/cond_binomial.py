"""Contains conditional Binomial leaf node for SPFlow in the ``base`` backend.
"""
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import tensorly as tl
from scipy.stats import binom  # type: ignore
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.base.structure.general.nodes.leaf_node import LeafNode
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext


class CondBinomial(LeafNode):
    r"""Conditional (univariate) Binomial distribution leaf node in the ``base`` backend.

    Represents a conditional univariate Binomial distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \binom{n}{k}p^k(1-p)^{n-k}

    where
        - :math:`p` is the success probability of each trial
        - :math:`n` is the number of total trials
        - :math:`k` is the number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Attributes:
        n:
            Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
        cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing ``p`` as a key, and the value should be
            a floating point value representing the success probability in :math:`[0,1]`.
    """

    def __init__(self, scope: Scope, n: int, cond_f: Optional[Callable] = None) -> None:
        r"""Initializes ``ConditionalBernoulli`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            n:
                Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point value representing the success probability in :math:`[0,1]`.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'CondBinomial' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence) == 0:
            raise ValueError(f"Evidence scope for 'CondBinomial' should not be empty.")

        super().__init__(scope=scope)

        # set non-conditional parameters
        self.set_params(n)

        # set optional conditional function
        self.set_cond_f(cond_f)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``CondBinomial`` can represent a single univariate node with ``BinomialType`` domain.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf only has one output
        if len(signatures) != 1:
            return False

        # get single output signature
        feature_ctx = signatures[0]
        domains = feature_ctx.get_domains()

        # leaf is a single non-conditional univariate node
        if len(domains) != 1 or len(feature_ctx.scope.query) != len(domains) or len(feature_ctx.scope.evidence) == 0:
            return False

        # leaf is a discrete Binomial distribution
        # NOTE: only accept instances of 'FeatureTypes.Binomial', otherwise required parameter 'n' is not specified. Reject 'FeatureTypes.Discrete' for the same reason.
        if not isinstance(domains[0], FeatureTypes.Binomial):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "CondBinomial":
        """Creates an instance from a specified signature.

        Returns:
            ``CondBinomial`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'CondBinomial' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if isinstance(domain, FeatureTypes.Binomial):
            n = domain.n
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'CondBinomial' that was not caught during acception checking."
            )

        return CondBinomial(feature_ctx.scope, n=n)

    def set_cond_f(self, cond_f: Optional[Callable] = None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``p`` as a key, and the value should be
                a floating point value representing the success probability in :math:`[0,1]`.
        """
        self.cond_f = cond_f

    def dist(self, p: float) -> rv_frozen:
        r"""Returns the SciPy distribution represented by the leaf node.

        Args:
            p:
                Floating point value representing the success probability of each trial between zero and one.

        Returns:
            ``scipy.stats.distributions.rv_frozen`` distribution.
        """
        return binom(n=self.n, p=p)

    def set_params(self, n: int) -> None:
        """Sets the parameters for the represented distribution.

        Args:
            n:
                Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
        """
        if n < 0 or not np.isfinite(n):
            raise ValueError(f"Value of 'n' for 'Binomial' must to greater of equal to 0, but was: {n}")

        if not (np.remainder(n, 1.0) == 0.0):
            raise ValueError(f"Value of 'n' for 'Binomial' must be (equal to) an integer value, but was: {n}")

        self.n = n

    def retrieve_params(self, data: tl.tensor, dispatch_ctx: DispatchContext) -> Tuple[Union[tl.tensor, float]]:
        r"""Retrieves the conditional parameter of the leaf node.

        First, checks if conditional parameter (``p``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameter.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameter.

        Args:
            data:
                Two-dimensional NumPy array containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            Floating point or scalar NumPy array representing the success probability.

        Raises:
            ValueError: No way to retrieve conditional parameters or invalid conditional parameters.
        """
        p, cond_f = None, None

        # check dispatch cache for required conditional parameter 'p'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if a value for 'p' is specified (highest priority)
            if "p" in args:
                p = args["p"]
            # check if alternative function to provide 'p' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'p' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'p' nor 'cond_f' is specified (via node or arguments)
        if p is None and cond_f is None:
            raise ValueError("'CondBinomial' requires either 'p' or 'cond_f' to retrieve 'p' to be specified.")

        # if 'p' was not already specified, retrieve it
        if p is None:
            p = cond_f(data)["p"]

        # check if value for 'p' is valid
        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of 'p' for 'CondBinomial' distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        return p

    def get_params(self) -> Tuple[int]:
        """Returns the parameters of the represented distribution.

        Returns:
            Integer number representing the number of i.i.d. Bernoulli trials and the floating point value representing the success probability.
        """
        return (self.n,)

    def check_support(self, data: tl.tensor, is_scope_data: bool = False) -> tl.tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Binomial distribution, which is:

        .. math::

            \text{supp}(\text{Binomial})=\{0,\hdots,n\}

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional NumPy array containing sample instances.
                Each row is regarded as a sample.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leaf's scope in the correct order (True) or if it needs to be extracted from the full data set.
                Defaults to False.

        Returns:
            Two-dimensional NumPy array indicating for each instance, whether they are part of the support (True) or not (False).
        """
        if is_scope_data:
            scope_data = data
        else:
            # select relevant data for scope
            scope_data = data[:, self.scope.query]

        if tl.ndim(scope_data) != 2 or tl.shape(scope_data)[1] != len(self.scope.query):
            raise ValueError(
                f"Expected 'scope_data' to be of shape (n,{len(self.scope.query)}), but was: {scope_data.shape}"
            )

        valid = tl.ones(scope_data.shape, dtype=bool)

        # nan entries (regarded as valid)
        nan_mask = np.isnan(scope_data)

        # check for infinite values
        valid[~nan_mask] &= ~np.isinf(scope_data[~nan_mask])

        # check if all values are valid integers
        valid[valid & ~nan_mask] &= np.remainder(scope_data[valid & ~nan_mask], 1) == 0

        # check if values are in valid range
        valid[valid & ~nan_mask] &= (scope_data[valid & ~nan_mask] >= 0) & (scope_data[valid & ~nan_mask] <= self.n)

        return valid
