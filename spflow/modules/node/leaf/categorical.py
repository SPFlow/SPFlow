"""Contains Categorical leaf node for SPFlow in the ``torch`` backend.
"""
from spflow.utils.projections import proj_convex_to_real
from typing import Callable, Optional, Union

from spflow import tensor as T
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
from spflow.modules.node.leaf.utils import apply_nan_strategy
from spflow.modules.node.leaf_node import LeafNode
from spflow.utils import Tensor


class Categorical(LeafNode):
    def __init__(self, scope: Scope, probs: list[float]) -> None:
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Categorical' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Categorical' should be empty, but was {scope.evidence}.")

        super().__init__(scope=scope)

        # register logits
        self.logits = T.requires_grad_(T.zeros(len(probs), device=self.device))  # type: ignore
        self.probs = probs

    @property
    def probs(self) -> Tensor:
        return T.softmax(self.logits, axis=0)

    @probs.setter
    def probs(self, probs: list[float]) -> None:
        if isinstance(probs, list):
            probs = T.tensor(probs, device=self.device)
        if T.ndim(probs) != 1:
            raise ValueError(
                f"Numpy array of weight probs for 'Categorical' is expected to be one-dimensional, but is {probs.ndim}-dimensional."
            )
        if not T.all(probs > 0):
            raise ValueError("Probabilities for 'Categorical' must be all positive.")
        if not T.isclose(T.sum(probs), 1.0):
            raise ValueError("Probabilities for 'Categorical' must sum up to one.")

        values = T.to(proj_convex_to_real(probs), device=self.device)
        self.logits = T.set_tensor_data(self.logits, values)

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Categorical`` can represent a single univariate node with ``CategoricalType`` domain.

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

        # leaf is a discrete Categorical distribution
        # NOTE: only accept instances of 'FeatureTypes.Categorical', otherwise required parameter 'n' is not specified. Reject 'FeatureTypes.Discrete' for the same reason.
        if not isinstance(domains[0], FeatureTypes.Categorical):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Categorical":
        """Creates an instance from a specified signature.

        Returns:
            ``Categorical`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'Categorical' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if isinstance(domain, FeatureTypes.Categorical):
            probs = domain.probs
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Categorical' that was not caught during acception checking."
            )

        return Categorical(feature_ctx.scope, probs=probs)

    def set_parameters(self, probs: list[float]) -> None:
        """Sets the parameters of the represented distribution.

        Args:
            probs:
                List of floating point values representing the success probabilities.
        """
        if isinstance(probs, list):
            self.probs = T.Tensor(probs, copy=True)
        self.probs = probs

    def get_trainable_parameters(self) -> tuple[int, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Integer number representing the number of i.i.d. Bernoulli trials and the floating point value representing the success probability.
        """
        return [self.logits]  # type: ignore

    def get_parameters(self) -> tuple[list[float]]:
        """Returns the parameters of the represented distribution.

        Returns:
            Integer number representing the number of i.i.d. Bernoulli trials and the floating point value representing the success probability.
        """
        return (self.probs,)

    def check_support(self, data: Tensor, is_scope_data: bool = False) -> Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Categorical distribution, which is:

        .. math::

            \text{supp}(\text{Categorical})=\{0,\hdots,n\}

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
        self.logits = T.to(self.logits, dtype=dtype, device=device)

    def describe_node(self) -> str:
        formatted_probs = [f"{num:.3f}" for num in T.tolist(self.probs)]
        return f"probs=[{', '.join(formatted_probs)}]"


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    leaf: Categorical,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    r"""Computes log-likelihoods for ``Categorical`` node in the ``torch`` backend given input data.

    Log-likelihood for ``Categorical`` is given by the logarithm of its probability mass function (PMF):

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
                "Encountered data instances that are not in the support of the Categorical distribution."
            )

    # Actual binomial log_prob computation:
    # compute probabilities for values inside distribution support
    scope_data = T.to(T.squeeze(scope_data[~marg_ids], 1), dtype=T.int64())
    # Compute the log_likelihood of scope_data w.r.t. the categorical probabilities given in self.logits
    logits = T.log_softmax(leaf.logits, axis=0)
    log_pmf = logits[scope_data]
    lls = T.unsqueeze(log_pmf, -1)
    log_prob = T.index_update(log_prob, ~marg_ids, lls)

    return log_prob


@dispatch  # type: ignore
def sample(
    leaf: Categorical,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    r"""Samples from ``Categorical`` nodes in the ``torch`` backend given potential evidence.

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

    # Sample from categorical distribution with probabilities self.probs
    probs = leaf.probs

    # Generate N random numbers from a uniform distribution
    random_nums = T.rand(n_samples)

    # Compute the cumulative sum of probabilities
    cumulative_probs = T.cumsum(probs, axis=0)

    # Find indices where each random number fits in the cumulative sum array
    sampled_indices = T.searchsorted(cumulative_probs, random_nums)

    return sampled_indices


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: Categorical,
    data: Tensor,
    weights: Optional[Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``Categorical`` node parameters in the ``torch`` backend.

    Estimates the success probabilities :math:`p` of a Categorical distribution from data, as follows:

    .. math::

        TODO

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
            Has no effect for ``Categorical`` nodes.
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

    # compute weighted counts
    weighted_counts = T.bincount(
        scope_data.reshape(-1), weights=weights.reshape(-1), minlength=len(leaf.probs)
    )

    # update parameters
    leaf.probs = weighted_counts / weighted_counts.sum()


@dispatch(memoize=True)  # type: ignore
def em(
    leaf: Categorical,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``Categorical`` in the ``torch`` backend.

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
