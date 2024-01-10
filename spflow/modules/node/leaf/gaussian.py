from typing import List, Tuple

from spflow.meta.data import FeatureContext, FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.meta.dispatch.sampling_context import init_default_sampling_context
from spflow.modules.node.leaf_node import LeafNode
from spflow.modules.node.leaf.utils import apply_nan_strategy
from spflow.tensor.ops import Tensor
from spflow import tensor as T
from spflow.utils.projections import proj_bounded_to_real, proj_real_to_bounded

from typing import Callable, Optional, Union


from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class Gaussian(LeafNode):
    def __init__(self, scope: Scope, mean: float = 0.0, std: float = 1.0) -> None:
        r"""Initializes ``Gaussian`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            mean:
                Floating point value representing the mean (:math:`\mu`) of the distribution.
                Defaults to 0.0.
            std:
                Floating point values representing the standard deviation (:math:`\sigma`) of the distribution (must be greater than 0).
                Defaults to 1.0.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Gaussian' should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Gaussian' should be empty, but was {scope.evidence}.")

        super().__init__(scope=scope)

        self._mean = T.requires_grad_(T.zeros(1, device=self.device))
        self._log_std = T.requires_grad_(T.zeros(1, device=self.device))

        self.mean = mean
        self.std = std

    @property
    def std(self) -> Tensor:
        """Returns the standard deviation."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self._log_std, lb=0.0)  # type: ignore

    @std.setter
    def std(self, std) -> Tensor:
        """Set the standard deviation."""
        # project auxiliary parameter onto actual parameter range
        if std <= 0.0:
            raise ValueError(f"Value for 'std' must be greater than 0.0, but was: {std}")
        if not T.isfinite(std):
            raise ValueError(f"Values for 'std' must be finite, but was: {std}")

        log_std = proj_bounded_to_real(T.tensor(std), lb=0.0)
        self._log_std = T.set_tensor_data(self._log_std, log_std)  # type: ignore

    @property
    def mean(self) -> Tensor:
        """Returns the mean."""
        return self._mean

    @mean.setter
    def mean(self, mean) -> Tensor:
        """Set the mean."""
        if not T.isfinite(mean):
            raise ValueError(f"Values for 'mean' must be finite, but was: {mean}")
        self._mean = T.set_tensor_data(self._mean, mean)

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Gaussian`` can represent a single univariate node with ``MetaType.Continuous`` or ``GaussianType`` domain.

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

        # leaf is a continuous Gaussian distribution
        if not (
            domains[0] == FeatureTypes.Continuous
            or domains[0] == FeatureTypes.Gaussian
            or isinstance(domains[0], FeatureTypes.Gaussian)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]) -> "Gaussian":
        """Creates an instance from a specified signature.

        Returns:
            ``Gaussian`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'Gaussian' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Continuous:
            mean, std = 0.0, 1.0
        elif domain == FeatureTypes.Gaussian:
            # instantiate object
            domain = domain()
            mean, std = domain.mean, domain.std
        elif isinstance(domain, FeatureTypes.Gaussian):
            mean, std = domain.mean, domain.std
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Gaussian' that was not caught during acception checking."
            )

        return Gaussian(feature_ctx.scope, mean=mean, std=std)

    def parameters(self) -> list[Tensor]:
        return [self.mean, self._log_std]

    def check_support(self, data: Tensor, is_scope_data: bool = False) -> Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Gaussian distribution, which is:

        .. math::

            \text{supp}(\text{Gaussian})=(-\infty,+\infty)

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
            Two-dimensional PyTorch tensor indicating for each instance, whether they are part of the support (True) or not (False).
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
        # TODO: Add support checks here! Previously, we used the pytorch distribution support checks as follows
        #       valid[~nan_mask] = self.dist.support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore
        #       Now we need our own support checks.
        invalid_support_mask = T.zeros(scope_data.shape[0], 1, dtype=bool, device=self.device)

        # Valid is everything that is not nan, inf or outside of the support
        return (~nan_mask) & (~inf_mask) & (~invalid_support_mask)

    def to(self, dtype=None, device=None):
        super().to(dtype=dtype, device=device)
        self.mean = T.to(self.mean, dtype=dtype, device=device)
        self.std = T.to(self.std, dtype=dtype, device=device)

    def describe_node(self) -> str:
        return f"mean={self.mean.item():.3f}, std={self.std.item():.3f}"


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    leaf: Gaussian,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Tensor:
    r"""Computes log-likelihoods for ``Gaussian`` node in the ``torch`` backend given input data.

    Log-likelihood for ``Gaussian`` is given by the logarithm of its probability distribution function (PDF):

    .. math::

        \log(\text{PDF}(x)) = \log(\frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2}))

    where
        - :math:`x` the observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

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

    # initialize zeros tensor (number of output values matches batch_size)
    log_prob: Tensor = T.zeros(data.shape[0], 1, dtype=data.dtype, device=leaf.device)

    # ----- marginalization -----
    # Get marginalization ids, indicated by nans in the tensor
    # NOTE: we don't need to set these to zero (1 in non-logspace) since log_prob is initialized with zeros
    marg_ids = T.sum(T.isnan(scope_data), axis=1) == len(leaf.scope.query)

    # ----- log probabilities -----

    if check_support:
        # create mask based on distribution's support
        valid_ids = leaf.check_support(scope_data[~marg_ids], is_scope_data=True)

        if not all(valid_ids):
            raise ValueError(
                f"Encountered data instances that are not in the support of the Gaussian distribution."
            )

    # compute probabilities for values inside distribution support
    scope_data = scope_data[~marg_ids]
    variance = leaf.std**2
    mean = leaf.mean
    lls = -0.5 * T.log(2 * T.PI * variance) - (scope_data - mean) ** 2 / (2 * variance)
    log_prob = T.index_update(log_prob, ~marg_ids, lls)

    return log_prob


@dispatch  # type: ignore
def sample(
    leaf: Gaussian,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    r"""Samples from ``Gaussian`` nodes in the ``torch`` backend given potential evidence.

    Samples missing values proportionally to its probability distribution function (PDF).

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

    scope_data = data[:, leaf.scope.query]
    nan_mask = T.isnan(scope_data)
    marg_ids = T.squeeze(T.to(nan_mask, T.int32()) == len(leaf.scope.query), axis=1)

    instance_ids_mask = T.zeros(data.shape[0], device=leaf.device, dtype=T.int32())
    instance_ids_mask = T.index_update(instance_ids_mask, sampling_ctx.instance_ids, 1)

    sampling_ids = T.logical_and(marg_ids, T.to(instance_ids_mask, dtype=bool))
    # Translate sampling ids to indices in the data tensor
    n_samples = T.sum(sampling_ids)
    samples = T.randn(n_samples, dtype=data.dtype, device=leaf.device) * leaf.std + leaf.mean

    data = T.assign_at_index_2(
        destination=data, index_1=sampling_ids, index_2=leaf.scope.query, values=samples
    )

    return data


@dispatch(memoize=True)  # type: ignore
def maximum_likelihood_estimation(
    leaf: Gaussian,
    data: Tensor,
    weights: Optional[Tensor] = None,
    bias_correction: bool = True,
    nan_strategy: Optional[Union[str, Callable]] = None,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    r"""Maximum (weighted) likelihood estimation (MLE) of ``Gaussian`` node parameters in the ``torch`` backend.

    Estimates the mean and standard deviation :math:`\mu` and :math:`\sigma` of a Gaussian distribution from data, as follows:

    .. math::

        \mu^{\*}=\frac{1}{n\sum_{i=1}^N w_i}\sum_{i=1}^{N}w_ix_i\\
        \sigma^{\*}=\frac{1}{\sum_{i=1}^N w_i}\sum_{i=1}^{N}w_i(x_i-\mu^{\*})^2

    or

    .. math::

        \sigma^{\*}=\frac{1}{(\sum_{i=1}^N w_i)-1}\sum_{i=1}^{N}w_i(x_i-\mu^{\*})^2

    if bias correction is used, where
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
    weights /= T.sum(weights) / scope_data.shape[0]

    # total (weighted) number of instances
    n_total = T.sum(weights)

    # calculate mean from data
    mean_est = T.sum(weights * scope_data) / n_total

    # calculate std from data
    bias_correction_term = 1 if bias_correction else 0
    std_est = T.sqrt(T.sum(weights * T.pow(scope_data - mean_est, 2)) / (n_total - bias_correction_term))

    # edge case (if all values are the same, not enough samples or very close to each other)
    if T.isclose(std_est, 0.0) or T.isnan(std_est):
        std_est = T.tensor(1e-8)

    # set parameters of leaf node
    leaf.mean = mean_est
    leaf.std = std_est


@dispatch(memoize=True)  # type: ignore
def em(
    leaf: Gaussian,
    data: Tensor,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> None:
    """Performs a single expectation maximizaton (EM) step for ``Gaussian`` in the ``torch`` backend.

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

    import torch  # TODO: solve this more elegantly

    with torch.no_grad():
        # ----- expectation step -----

        # get cached log-likelihood gradients w.r.t. module log-likelihoods
        expectations = dispatch_ctx.cache["log_likelihood"][leaf].grad
        # normalize expectations for better numerical stability
        assert expectations is not None  # TODO remove assert
        expectations /= T.sum(expectations)

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


if __name__ == "__main__":
    g = Gaussian(scope=Scope([0]), mean=0.0, std=1.0)
    # g.to_device("cuda")
    print(g)

    data = T.tensor([[1.0], [2.0], [3.0]])  # .to("cuda")
    print(log_likelihood(g, data))
    print(sample(g, num_samples=3))
    maximum_likelihood_estimation(g, data)
    print(g.mean, g.std)
