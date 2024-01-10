from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from spflow import tensor as T
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
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
from spflow.tensor import Tensor
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded


class Bernoulli(LeafNode):
    r"""(Univariate) Bernoulli distribution leaf node in the ``torch`` backend.

    Represents an univariate Bernoulli distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k)=\begin{cases} p   & \text{if } k=1\\
                                    1-p & \text{if } k=0\end{cases}

    where
        - :math:`p` is the success probability in :math:`[0,1]`
        - :math:`k` is the outcome of the trial (0 or 1)

    Internally :math:`p` is represented as an unbounded parameter that is projected onto the bounded range :math:`[0,1]` for representing the actual success probability.

    Attributes:
        p_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual success probability.
        p:
            Scalar PyTorch tensor representing the success probability of the Bernoulli distribution (projected from ``p_aux``).
    """

    def __init__(self, scope: Scope, p: float = 0.5) -> None:
        r"""Initializes ``Bernoulli`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            p:
                Floating point value representing the success probability of the Bernoulli distribution between zero and one.
                Defaults to 0.5.

        Raises:
            ValueError: Invalid arguments.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Bernoulli' should be 1, but was: {len(scope.query)}")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Bernoulli' should be empty, but was {scope.evidence}.")

        super().__init__(scope=scope)

        # register auxiliary torch parameter for the success probability p
        self.log_p = T.requires_grad_(T.zeros(1, device=self.device))  # type: ignore
        self.p = p

    @property
    def p(self) -> Tensor:
        """Returns the success proability."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.log_p, lb=0.0, ub=1.0)  # type: ignore

    @p.setter
    def p(self, p: float) -> None:
        r"""Sets the success probability.

        Args:
            p:
                Floating point representing the success probability in :math:`[0,1]`.

        Raises:
            ValueError: Invalid arguments.
        """
        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(f"Value of 'p' for 'Bernoulli' must to be between 0.0 and 1.0, but was: {p}")

        log_p = proj_bounded_to_real(T.tensor(p), lb=0.0, ub=1.0)  # type: ignore
        self.log_p = T.set_tensor_data(self.log_p, log_p)  # type: ignore

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Bernoulli`` can represent a single univariate node with ``MetaType.Discrete`` or ``BernoulliType`` domain.

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

        # leaf is a discrete Bernoulli distribution
        if not (
            domains[0] == FeatureTypes.Discrete
            or domains[0] == FeatureTypes.Bernoulli
            or isinstance(domains[0], FeatureTypes.Bernoulli)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Bernoulli":
        """Creates an instance from a specified signature.

        Returns:
            ``Bernoulli`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'Bernoulli' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Discrete:
            p = 0.5
        elif domain == FeatureTypes.Bernoulli:
            # instantiate object
            p = domain().p
        elif isinstance(domain, FeatureTypes.Bernoulli):
            p = domain.p
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Bernoulli' that was not caught during acception checking."
            )

        return Bernoulli(feature_ctx.scope, p=p)

    def parameters(self) -> list[Tensor]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the success probability.
        """
        return [self.log_p]

    def check_support(self, data: Tensor, is_scope_data: bool = False) -> Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or not instances are part of the support of the Bernoulli distribution, which is:

        .. math::

            \text{supp}(\text{Bernoulli})=\{0,1\}

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
        inf_mask = T.isinf(scope_data)

        invalid_support_mask = T.zeros(scope_data.shape[0], 1, dtype=bool, device=self.device)

        # TODO: Add support checks here! Previously, we used the pytorch distribution support checks as follows
        #       valid[~nan_mask] = self.dist.support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore
        #       Now we need our own support checks.

        # check for infinite values
        invalid_support_mask = T.zeros(scope_data.shape[0], 1, dtype=bool, device=self.device)

        # Valid is everything that is not nan, inf or outside of the support
        return (~nan_mask) & (~inf_mask) & (~invalid_support_mask)

    def to(self, dtype=None, device=None):
        super().to(dtype=dtype, device=device)
        self.log_p = T.to(self.log_p, dtype=dtype, device=device)

    def describe_node(self) -> str:
        return f"p={self.p.item():.3f}"


@dispatch  # type: ignore
def sample(
    leaf: Bernoulli,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    r"""Samples from ``Bernoulli`` nodes in the ``torch`` backend given potential evidence.

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

    # Randomly sample from bernoulli distribution with success probability p
    samples = T.to(T.rand(n_samples, device=leaf.device) < leaf.p, dtype=T.float32())

    data = T.assign_at_index_2(
        destination=data, index_1=sampling_ids, index_2=leaf.scope.query, values=samples
    )

    return data


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: Bernoulli,
    data: Tensor,
    weights: Optional[Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``Bernoulli`` node parameters in the ``torch`` backend.

    Estimates the success probability :math:`p` of a Bernoulli distribution from data, as follows:

    .. math::

        p^{\*}=\frac{1}{\sum_{i=1}^N w_i}\sum_{i=1}^{N}w_ix_i

    where
        - :math:`n` is the number of i.i.d. Bernoulli trials
        - :math:`N` is the number of samples in the data set
        - :math:`x_i` is the data of the relevant scope for the `i`-th sample of the data set
        - :math:`w_i` is the weight for the `i`-th sample of the data set

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
            Not relevant for ``Bernoulli`` nodes.
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

    if weights is None:
        weights = T.ones(data.shape[0], device=leaf.device)

    if weights.ndim != 1 or weights.shape[0] != data.shape[0]:
        raise ValueError(
            "Number of specified weights for maximum-likelihood estimation does not match number of data points."
        )

    # reshape weights
    weights = weights.reshape(-1, 1)

    if check_support:
        if T.any(~leaf.check_support(scope_data, is_scope_data=True)):
            raise ValueError("Encountered values outside of the support for 'Bernoulli'.")

    # NaN entries (no information)
    nan_mask = T.isnan(scope_data)

    if T.all(nan_mask):
        raise ValueError("Cannot compute maximum-likelihood estimation on nan-only data.")

    if nan_strategy is None and T.any(nan_mask):
        raise ValueError(
            "Maximum-likelihood estimation cannot be performed on missing data by default. Set a strategy for handling missing values if this is intended."
        )

    if isinstance(nan_strategy, str):
        if nan_strategy == "ignore":
            # simply ignore missing data
            scope_data = scope_data[~nan_mask.squeeze(1)]
            weights = weights[~nan_mask.squeeze(1)]
        else:
            raise ValueError("Unknown strategy for handling missing (NaN) values for 'Bernoulli'.")
    elif isinstance(nan_strategy, Callable):
        scope_data = nan_strategy(scope_data)
        # TODO: how to handle weights?
    elif nan_strategy is not None:
        raise ValueError(
            f"Expected 'nan_strategy' to be of type '{type(str)}, or '{Callable}' or '{None}', but was of type {type(nan_strategy)}."
        )

    # normalize weights to sum to n_samples
    weights /= weights.sum() / scope_data.shape[0]

    # total (weighted) number of instances
    n_total = weights.sum()

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
    leaf: Bernoulli,
    data: T.Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``Bernoulli`` in the ``torch`` backend.

    Args:
        leaf:
            Leaf node to perform EM step for.
        data:
            Two-dimensional PyTorch tensor containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # with torch.no_grad():  # TODO: this was present in the torch impl. do we still need this?
    # with T.no_grad():
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


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    leaf: Bernoulli,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    r"""Computes log-likelihoods for ``Bernoulli`` node in the ``torch`` backend given input data.

    Log-likelihood for ``Bernoulli`` is given by the logarithm of its probability mass function (PMF):

    .. math::

        \log(\text{PMF}(k))=\begin{cases} \log(p)   & \text{if } k=1\\
                                          \log(1-p) & \text{if } k=0\end{cases}

    where
        - :math:`p` is the success probability in :math:`[0,1]`
        - :math:`k` is the outcome of the trial (0 or 1)

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
    scope_data = data[:, leaf.scope.query]

    # initialize empty tensor (number of output values matches batch_size)
    log_prob = T.zeros(data.shape[0], 1, device=leaf.device)

    # ----- marginalization -----

    marg_ids = T.sum(T.isnan(scope_data), axis=1) == len(leaf.scope.query)

    # if all instances should be marginalized, return early
    if T.all(marg_ids):
        return log_prob

    # ----- log probabilities -----

    if check_support:
        # create masked based on distribution's support
        valid_ids = leaf.check_support(scope_data[~marg_ids], is_scope_data=True).squeeze(1)

        if not all(valid_ids):
            raise ValueError(
                f"Encountered data instances that are not in the support of the Bernoulli distribution."
            )

    # Actual binomial log_prob computation:
    # compute probabilities for values inside distribution support
    scope_data = T.to(T.squeeze(scope_data[~marg_ids], 1), dtype=T.int32())

    # Bernoulli log_pmf
    log_pmf = T.log(leaf.p) * scope_data + T.log(1 - leaf.p) * (1 - scope_data)

    lls = T.unsqueeze(log_pmf, -1)
    log_prob = T.index_update(log_prob, ~marg_ids, lls)

    return log_prob
