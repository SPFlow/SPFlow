"""Contains Binomial leaf node for SPFlow in the ``torch`` backend.
"""
from typing import Callable, List, Optional, Tuple, Union


import spflow.tensor.dtype
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.dispatch.sampling_context import (
    SamplingContext,
    init_default_sampling_context,
)
from spflow.modules.node.leaf_node import LeafNode
from spflow.modules.node.leaf.utils import apply_nan_strategy
from spflow.utils import Tensor
from spflow import tensor as T
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded


class Binomial(LeafNode):
    r"""(Univariate) Binomial distribution leaf node in the ``torch`` backend.

    Represents an univariate Binomial distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k) = \binom{n}{k}p^k(1-p)^{n-k}

    where
        - :math:`p` is the success probability of each trial in :math:`[0,1]`
        - :math:`n` is the number of total trials
        - :math:`k` is the number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Internally :math:`p` is represented as an unbounded parameter that is projected onto the bounded range :math:`[0,1]` for representing the actual success probability.

    Attributes:
        n:
            Scalar PyTorch tensor representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
        p_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual success probability.
        p:
            Scalar PyTorch tensor representing the success probability of the Bernoulli distribution (projected from ``p_aux``).
    """

    def __init__(self, scope: Scope, n: int, p: float = 0.5) -> None:
        r"""Initializes ``Binomial`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            n:
                Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
            p:
                Floating point value representing the success probability of each trial between zero and one.
                Defaults to 0.5.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Binomial' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Binomial' should be empty, but was {scope.evidence}.")
        if n < 1:
            raise ValueError(f"Value of 'n' for 'Binomial' must be greater than 0, but was: {n}")

        super().__init__(scope=scope)

        # register number of trials n as torch buffer (not trainable)
        self._n = T.tensor(n, dtype=T.int32(), device=self.device)
        self.n = T.tensor(n, dtype=T.int32(), device=self.device)

        # register auxiliary torch parameter for the success probability p
        self.log_p = T.requires_grad_(T.zeros(1, device=self.device))  # type: ignore
        self.p = p

    @property
    def p(self) -> Tensor:
        """Returns the success proability."""
        # project auxiliary parameter onto actual parameter range
        # return self.p_aux
        return proj_real_to_bounded(self.log_p, lb=0.0, ub=1.0)  # type: ignore

    @property
    def n(self) -> Tensor:
        return self._n

    @n.setter
    def n(self, n: int):
        if isinstance(n, float):
            if not n.is_integer():
                raise ValueError(
                    f"Value of 'n' for 'Binomial' must be (equal to) an integer value, but was: {n}"
                )
            n = T.tensor(int(n))
        elif isinstance(n, int):
            n = T.tensor(n)
        if n < 0 or not T.isfinite(n):
            raise ValueError(f"Value of 'n' for 'Binomial' must to greater of equal to 0, but was: {n}")

        self._n = T.tensor(n, dtype=T.int32(), device=self.device)

    @p.setter
    def p(self, p: float) -> None:
        r"""Sets the success probability.

        Args:
            p:
                Floating point representing the success probability in :math:`[0,1]`.

        Raises:
            ValueError: Invalid arguments.
        """
        if p < 0.0 or p > 1.0 or not T.isfinite(p):
            raise ValueError(
                f"Value of 'p' for 'Binomial' distribution must to be between 0.0 and 1.0, but was: {p}"
            )

        log_p = proj_bounded_to_real(T.tensor(p), lb=0.0, ub=1.0)  # type: ignore
        self.log_p = T.set_tensor_data(self.log_p, log_p)  # type: ignore

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Binomial`` can represent a single univariate node with ``BinomialType`` domain.

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
        """Creates an instance from a specified signature.

        Returns:
            ``Binomial`` instance.

        Raises:
            Signatures not accepted by the module.
        """
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

        return Binomial(feature_ctx.scope, n=n, p=p)

    def set_parameters(self, n: int, p: float) -> None:
        """Sets the parameters for the represented distribution.

        Bounded parameter ``p`` is projected onto the unbounded parameter ``p_aux``.

        TODO: projection function

        Args:
            n:
                Integer representing the number of i.i.d. Bernoulli trials (greater or equal to 0).
            p:
                Floating point value representing the success probability of the Binomial distribution between zero and one.
        """
        if isinstance(n, float):
            if not n.is_integer():
                raise ValueError(
                    f"Value of 'n' for 'Binomial' must be (equal to) an integer value, but was: {n}"
                )
            n = T.tensor(int(n))
        elif isinstance(n, int):
            n = T.tensor(n)
        if n < 0 or not T.isfinite(n):
            raise ValueError(f"Value of 'n' for 'Binomial' must to greater of equal to 0, but was: {n}")

        self.p = T.tensor(p)
        self.n = T.tensor(self.n, dtype=T.int32())

    def get_trainable_parameters(self) -> tuple[int, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Integer number representing the number of i.i.d. Bernoulli trials and the floating point value representing the success probability.
        """
        # return self.n.data.cpu().numpy(), self.p.data.cpu().numpy()  # type: ignore
        return [self.log_p]  # type: ignore

    def get_parameters(self) -> tuple[int, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Integer number representing the number of i.i.d. Bernoulli trials and the floating point value representing the success probability.
        """
        return self.n, self.p  # type: ignore

    def check_support(self, data: Tensor, is_scope_data: bool = False) -> Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Binomial distribution, which is:

        .. math::

            \text{supp}(\text{Binomial})=\{0,\hdots,n\}

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leaf's scope in the correct order (True) or if it needs to be extracted from the full data set.
                Defaults to False.

        Returns:
            Two dimensional PyTorch tensor indicating for each instance, whether they are part of the support (True) or not (False).
        """
        if is_scope_data:
            scope_data = data
        else:
            # select relevant data for scope
            scope_data = data[:, self.scope.query]

        if scope_data.ndim != 2 or scope_data.shape[1] != len(self.scope.query):
            raise ValueError(
                f"Expected 'scope_data' to be of shape (n,{len(self.scope.query)}), but was: {scope_data.shape}"
            )

        # nan entries (regarded as valid)
        nan_mask = T.isnan(scope_data)

        # nan entries (regarded as valid)
        nan_mask = T.isnan(scope_data)
        inf_mask = T.isinf(scope_data)
        # TODO: Add support checks here! Previously, we used the pytorch distribution support checks as follows
        #       valid[~nan_mask] = self.dist.support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore
        #       Now we need our own support checks.
        invalid_support_mask = T.zeros(scope_data.shape[0], 1, dtype=bool, device=self.device)

        # Valid is everything that is not nan, inf or outside of the support
        return (~nan_mask) & (~inf_mask) & (~invalid_support_mask)

    def to(self, dtype=None, device=None):
        super().to(dtype=dtype, device=device)
        self.n = T.to(self.n, dtype=dtype, device=device)
        self.log_p = T.to(self.log_p, dtype=dtype, device=device)

    def describe_node(self) -> str:
        return f"N={self.n.item()}, p={self.p.item():.3f}"


def _log_factorial(n):
    max_n = T.max(n)
    arange = (
        T.arange(1, max_n + 1, device=T.device(n), dtype=spflow.tensor.dtype.get_default_float_dtype()) + 1
    )
    log_fact = T.lgamma(arange)
    index = n - 1
    index = T.index_update(index, index < 0, 0)
    result = log_fact[index]  # Adjust index since Python is 0-based
    zero_mask = n == 0
    result = T.index_update(result, zero_mask, 0)
    return result


def _log_binomial_coefficient(n, k):
    return _log_factorial(n) - (_log_factorial(k) + _log_factorial(n - k))


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    leaf: Binomial,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    r"""Computes log-likelihoods for ``Binomial`` node in the ``torch`` backend given input data.

    Log-likelihood for ``Binomial`` is given by the logarithm of its probability mass function (PMF):

    .. math::

        \log(\text{PMF}(k)) = \log(\binom{n}{k}p^k(1-p)^{n-k})

    where
        - :math:`p` is the success probability of each trial in :math:`[0,1]`
        - :math:`n` is the number of total trials
        - :math:`k` is the number of successes
        - :math:`\binom{n}{k}` is the binomial coefficient (n choose k)

    Missing values (i.e., NaN) are marginalized over.

    Args:
        node:
            Leaf node to perform inference for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the distribution.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional PyTorch tensor containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.

    Raises:
        ValueError: Data outside of support.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # get information relevant for the scope
    # scope_data = data[:, leaf.scope.query]
    scope_data = data[:, leaf.scope.query]

    # initialize zeros tensor (number of output values matches batch_size)
    log_prob = T.zeros(data.shape[0], 1, device=leaf.device)

    # ----- marginalization -----
    # Get marginalization ids, indicated by nans in the tensor
    # NOTE: we don't need to set these to zero (1 in non-logspace) since log_prob is initialized with zeros
    marg_ids = T.sum(T.isnan(scope_data), axis=1) == len(leaf.scope.query)

    # if all instances should be marginalized, return early
    if T.all(marg_ids):
        return log_prob

    # ----- log probabilities -----

    if check_support:
        # create masked based on distribution's support
        valid_ids = leaf.check_support(scope_data[~marg_ids], is_scope_data=True)

        if not all(valid_ids):
            raise ValueError(
                f"Encountered data instances that are not in the support of the Binomial distribution."
            )

    # Actual binomial log_prob computation:
    # compute probabilities for values inside distribution support
    scope_data = T.to(T.squeeze(scope_data[~marg_ids], 1), dtype=T.int64())
    log_coef = _log_binomial_coefficient(
        T.full((scope_data.shape[0],), leaf.n, dtype=T.int64(), device=leaf.device), scope_data
    )
    log_pmf = log_coef + scope_data * T.log(leaf.p) + (leaf.n - scope_data) * T.log(1 - leaf.p)
    lls = T.unsqueeze(log_pmf, -1)
    log_prob = T.index_update(log_prob, ~marg_ids, lls)

    return log_prob


@dispatch  # type: ignore
def sample(
    leaf: Binomial,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    r"""Samples from ``Binomial`` nodes in the ``torch`` backend given potential evidence.

    Samples missing values proportionally to its probability mass function (PMF).

    Args:
        leaf:
            Leaf node to sample from.
        data:
            Two-dimensional PyTorch tensor containing potential evidence.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
        sampling_ctx:
            Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

    Returns:
        Two-dimensional PyTorch tensor containing the sampled values together with the specified evidence.
        Each row corresponds to a sample.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    if any([i >= data.shape[0] for i in sampling_ctx.instance_ids]):
        raise ValueError("Some instance ids are out of bounds for data tensor.")

    marg_ids = (T.isnan(data[:, leaf.scope.query]) == len(leaf.scope.query)).squeeze(1)

    instance_ids_mask = T.zeros(data.shape[0], device=leaf.device, dtype=T.int32())
    instance_ids_mask = T.index_update(instance_ids_mask, sampling_ctx.instance_ids, 1)

    sampling_ids = T.logical_and(marg_ids, instance_ids_mask)

    n_samples = sampling_ids.sum().item()

    # Generate random numbers between 0 and 1
    random_numbers = T.rand(n_samples, leaf.n, device=leaf.device)
    # Compare with probability p to determine successes
    successes = random_numbers < leaf.p
    # Sum the successes for each sample
    samples = T.to(T.sum(successes, axis=1), dtype=T.float32())

    data = T.assign_at_index_2(
        destination=data, index_1=sampling_ids, index_2=leaf.scope.query, values=samples
    )

    return data


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: Binomial,
    data: Tensor,
    weights: Optional[Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``Binomial`` node parameters in the ``torch`` backend.

    Estimates the success probability :math:`p` of a Binomial distribution from data, as follows:

    .. math::

        p^{\*}=\frac{1}{n\sum_{i=1}^N w_i}\sum_{i=1}^{N}w_ix_i

    where
        - :math:`N` is the number of samples in the data set
        - :math:`x_i` is the data of the relevant scope for the `i`-th sample of the data set
        - :math:`w_i` is the weight for the `i`-th sample of the data set

    The number of i.i.d. Bernoulli trials is fixed and will not be estimated.
    Weights are normalized to sum up to :math:`N`.

    Args:
        leaf:
            Leaf node to estimate parameters of.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        weights:
            Optional one-dimensional PyTorch tensor containing non-negative weights for all data samples.
            Must match number of samples in ``data``.
            Defaults to None in which case all weights are initialized to ones.
        bias_corrections:
            Boolen indicating whether or not to correct possible biases.
            Has no effect for ``Binomial`` nodes.
            Defaults to True.
        nan_strategy:
            Optional string or callable specifying how to handle missing data.
            If 'ignore', missing values (i.e., NaN entries) are ignored.
            If a callable, it is called using ``data`` and should return another PyTorch tensor of same size.
            Defaults to None.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Raises:
        ValueError: Invalid arguments.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # select relevant data for scope
    scope_data = data[:, leaf.scope.query]

    # handle nans
    scope_data, weights = apply_nan_strategy(nan_strategy, scope_data, leaf, weights, check_support)

    # normalize weights to sum to n_samples
    weights /= weights.sum() / scope_data.shape[0]

    # total (weighted) number of instances times number of trials per instance
    n_total = weights.sum() * leaf.n

    # count (weighted) number of total successes
    n_success = (weights * scope_data).sum()

    # estimate (weighted) success probability
    p_est = n_success / n_total

    # edge case: if prob. 1 (or 0), set to smaller (or larger) value
    if T.isclose(p_est, 0.0):
        p_est = 1e-8
    elif T.isclose(p_est, 1.0):
        p_est = 1 - 1e-8

    # set parameters of leaf node
    leaf.p = p_est


@dispatch(memoize=True)  # type: ignore
def em(
    leaf: Binomial,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``Binomial`` in the ``torch`` backend.

    Args:
        leaf:
            Leaf node to perform EM step for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
    """
    assert (
        leaf.backend == "pytorch"
    ), f"EM is only supported in PyTorch but was called for backend '{leaf.backend}'."

    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # with torch.no_grad():  # TODO: this was present in the torch impl. do we still need this?
    # ----- expectation step -----

    # get cached log-likelihood gradients w.r.t. module log-likelihoods
    expectations = dispatch_ctx.cache["log_likelihood"][leaf].grad
    # normalize expectations for better numerical stability
    expectations /= expectations.sum()

    # ----- maximization step -----

    # update parameters through maximum weighted likelihood estimation
    maximum_likelihood_estimation(
        leaf,
        data,
        weights=expectations.squeeze(1),
        bias_correction=False,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
    )

    # NOTE: since we explicitely override parameters in 'maximum_likelihood_estimation', we do not need to zero/None parameter gradients
