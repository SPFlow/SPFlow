# -*- coding: utf-8 -*-
"""Contains conditional Multivariate Normal leaf node for SPFlow in the ``torch`` backend.
"""
import numpy as np
import torch
import torch.distributions as D
from typing import Tuple, Union, Optional, Iterable, Callable, List, Type
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import MetaType, FeatureType, FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.spn.nodes.node import LeafNode
from spflow.torch.structure.nodes.leaves.parametric.cond_gaussian import (
    CondGaussian,
)
from spflow.base.structure.nodes.leaves.parametric.cond_multivariate_gaussian import (
    CondMultivariateGaussian as BaseCondMultivariateGaussian,
)


class CondMultivariateGaussian(LeafNode):
    r"""Conditional Multivariate Gaussian distribution leaf node in the ``torch`` backend.

    Represents a conditional multivariate Gaussian distribution, with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{(2\pi)^d\det\Sigma}}\exp\left(-\frac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)\right)

    where
        - :math:`d` is the dimension of the distribution
        - :math:`x` is the :math:`d`-dim. vector of observations
        - :math:`\mu` is the :math:`d`-dim. mean vector
        - :math:`\Sigma` is the :math:`d\times d` covariance matrix

    Attributes
        cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing ``mean``,``cov`` as keys.
            The value for ``mean`` should be a list of floating point values, one-dimensional NumPy array
            or one-dimensional PyTorch tensor containing the means.
            The value for ``cov`` should be a list of lists of floating points, two-dimensional NumPy array
            or two-dimensional PyTorch tensor containing a symmetric positive semi-definite matrix.
    """

    def __init__(
        self,
        scope: Scope,
        cond_f: Optional[Callable] = None,
    ) -> None:
        r"""Initializes ``CondMultivariateGaussian`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``mean``,``cov`` as keys.
                The value for ``mean`` should be a list of floating point values, one-dimensional NumPy array
                or one-dimensional PyTorch tensor containing the means.
                The value for ``cov`` should be a list of lists of floating points, two-dimensional NumPy array
                or two-dimensional PyTorch tensor containing a symmetric positive semi-definite matrix.
        """
        # check if scope contains duplicates
        if len(set(scope.query)) != len(scope.query):
            raise ValueError(
                "Query scope for 'CondMultivariateGaussian' contains duplicate variables."
            )
        if len(scope.evidence) == 0:
            raise ValueError(
                f"Evidence scope for 'CondMultivariateGaussian' should not be empty."
            )
        if len(scope.query) < 1:
            raise ValueError(
                "Size of query scope for 'CondMultivariateGaussian' must be at least 1."
            )

        super(CondMultivariateGaussian, self).__init__(scope=scope)

        # dimensions
        self.d = len(scope)

        self.set_cond_f(cond_f)

    @classmethod
    def accepts(self, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``CondMultivariateGaussian`` can represent a single multivariate node with ``MetaType.Continuous`` or ``GaussianType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf only has one output
        if len(signatures) != 1:
            return False

        # get single output signature
        feature_ctx = signatures[0]
        domains = feature_ctx.get_domains()

        # leaf is a single non-conditional (possibly multivariate) node
        if (
            len(domains) < 1
            or len(feature_ctx.scope.query) != len(domains)
            or len(feature_ctx.scope.evidence) == 0
        ):
            return False

        # leaf is a continuous (multivariate) Gaussian distribution
        if not all(
            [
                domain == FeatureTypes.Continuous
                or domain == FeatureTypes.Gaussian
                or isinstance(domain, FeatureTypes.Gaussian)
                for domain in domains
            ]
        ):
            return False

        return True

    @classmethod
    def from_signatures(
        self, signatures: List[FeatureContext]
    ) -> "CondMultivariateGaussian":
        """Creates an instance from a specified signature.

        Returns:
            ``CondMultivariateGaussian`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not self.accepts(signatures):
            raise ValueError(
                f"'CondMultivariateGaussian' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]

        for domain in feature_ctx.get_domains():
            # read or initialize parameters
            if (
                domain == MetaType.Continuous
                or domain == FeatureTypes.Gaussian
                or isinstance(domain, FeatureTypes.Gaussian)
            ):
                pass
            else:
                raise ValueError(
                    f"Unknown signature type {type} for 'CondMultivariateGaussian' that was not caught during acception checking."
                )

        return CondMultivariateGaussian(feature_ctx.scope)

    def set_cond_f(self, cond_f: Optional[Callable] = None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``mean``,``cov`` as keys.
                The value for ``mean`` should be a list of floating point values, one-dimensional NumPy array
                or one-dimensional PyTorch tensor containing the means.
                The value for ``cov`` should be a list of lists of floating points, two-dimensional NumPy array
                or two-dimensional PyTorch tensor containing a symmetric positive semi-definite matrix.
        """
        self.cond_f = cond_f

    def dist(
        self,
        mean: torch.Tensor,
        cov: Optional[torch.Tensor] = None,
        cov_tril: Optional[torch.Tensor] = None,
    ) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Args:
            mean:
                A one-dimensional PyTorch tensor containing the means (:math:`\mu`) of each of the one-dimensional Normal distributions.
                Must have exactly as many elements as the scope of this leaf.
                Defaults to all zeros.
            cov:
                A two-dimensional PyTorch tensor (representing a :math:`d\times d` symmetric positive semi-definite matrix, where :math:`d` is the length
                of the scope) describing the covariances of the distribution. The diagonal holds the variances (:math:`\sigma^2`) of each of the one-dimensional distributions.
                Defaults to the identity matrix.

        Returns:
            ``torch.distributions.MultivariateNormal`` instance.
        """
        if cov is None and cov_tril is None:
            raise ValueError(
                "Calling 'dist' of CondMultivariateGaussian requries either 'cov' or 'cov_tril' to be specified."
            )
        elif cov is not None and cov_tril is not None:
            raise ValueError(
                "Calling 'dist' of CondMultivariateGaussian requries either 'cov' or 'cov_tril' to be specified, but not both."
            )

        if cov is not None:
            return D.MultivariateNormal(loc=mean, covariance_matrix=cov)
        else:
            return D.MultivariateNormal(loc=mean, scale_tril=cov_tril)

    def retrieve_params(
        self, data: torch.Tensor, dispatch_ctx: DispatchContext
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""Retrieves the conditional parameter of the leaf node.

        First, checks if conditional parameters (``mean``,``cov``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameters.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameters.

        Args:
            data:
                Two-dimensional PyTorch tensor containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            Tuple of a one- and a two-dimensional PyTorch tensor representing the mean and covariance matrix.

        Raises:
            ValueError: No way to retrieve conditional parameters or invalid conditional parameters.
        """
        mean, cov, cov_tril, cond_f = None, None, None, None
        specified_tril = False

        # check dispatch cache for required conditional parameters 'mean', 'cov'/'cov_tril'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if values for 'mean', 'cov'/'cov_tril' are specified (highest priority)
            if "mean" in args:
                mean = args["mean"]
            if "cov" in args:
                cov = args["cov"]
            if "cov_tril" in args:
                cov_tril = args["cov_tril"]
            # check if alternative function to provide 'mean', 'cov'/'cov_tril' is specified (second to highest priority)
            if "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'mean','cov'/'cov_tril' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'mean' or 'cov'/'cov_tril' nor 'cond_f' is specified (via node or arguments)
        if (
            mean is None or (cov is None and cov_tril is None)
        ) and cond_f is None:
            raise ValueError(
                "'CondMultivariateGaussian' requires either 'mean' and 'cov'/'cov_tril' or 'cond_f' to retrieve 'mean', 'cov'/'cov_tril' to be specified."
            )

        # if 'mean' or 'cov' not already specified, retrieve them
        if mean is None or (cov is None and cov_tril is None):
            params = cond_f(data)
            mean = params["mean"]

            if "cov" in params:
                cov = params["cov"]
            if "cov_tril" in params:
                cov_tril = params["cov_tril"]

            if cov is None and cov_tril is None:
                raise ValueError(
                    "CondMultivariateGaussian requries either 'cov' or 'cov_tril' to be specified."
                )
            elif cov is not None and cov_tril is not None:
                raise ValueError(
                    "CondMultivariateGaussian requries either 'cov' or 'cov_tril' to be specified, but not both."
                )

        # cov_tril specified (and not cov)
        if cov_tril is not None:
            cov = cov_tril
            specified_tril = True

        if cov is not None:
            # cast lists to torch tensors
            if isinstance(mean, list):
                # convert float list to torch tensor
                mean = torch.tensor([float(v) for v in mean])
            elif isinstance(mean, np.ndarray):
                # convert numpy array to torch tensor
                mean = torch.from_numpy(mean).type(torch.get_default_dtype())

            if isinstance(cov, list):
                # convert numpy array to torch tensor
                cov = torch.tensor([[float(v) for v in row] for row in cov])
            elif isinstance(cov, np.ndarray):
                # convert numpy array to torch tensor
                cov = torch.from_numpy(cov).type(torch.get_default_dtype())

            # check mean vector for nan or inf values
            if torch.any(torch.isinf(mean)):
                raise ValueError(
                    "Value of 'mean' for 'CondMultivariateGaussian' may not contain infinite values"
                )
            if torch.any(torch.isnan(mean)):
                raise ValueError(
                    "Value of 'mean' for 'CondMultivariateGaussian' may not contain NaN values"
                )

            # make sure that number of dimensions matches scope length
            if (
                (mean.ndim == 1 and mean.shape[0] != len(self.scope.query))
                or (mean.ndim == 2 and mean.shape[1] != len(self.scope.query))
                or mean.ndim > 2
            ):
                raise ValueError(
                    f"Dimensions of 'mean' for 'CondMultivariateGaussian' should match scope size {len(self.scope.query)}, but was: {mean.shape}"
                )

            if mean.ndim == 2:
                mean = mean.squeeze(0)

            # make sure that dimensions of covariance matrix are correct
            if cov.ndim != 2 or (
                cov.ndim == 2
                and (
                    cov.shape[0] != len(self.scope.query)
                    or cov.shape[1] != len(self.scope.query)
                )
            ):
                raise ValueError(
                    f"Value of 'cov' for 'CondMultivariateGaussian' expected to be of shape ({len(self.scope.query), len(self.scope.query)}), but was: {cov.shape}"
                )

            # check covariance matrix for nan or inf values
            if torch.any(torch.isinf(cov)):
                raise ValueError(
                    "Value of 'cov for 'CondMultivariateGaussian' may not contain infinite values"
                )
            if torch.any(torch.isnan(cov)):
                raise ValueError(
                    "Value of 'cov' for 'CondMultivariateGaussian' may not contain NaN values"
                )

            if specified_tril:
                # compute eigenvalues of cov variance matrix
                eigvals = torch.linalg.eigvalsh(torch.matmul(cov, cov.T))
            else:
                # compute eigenvalues (can use eigvalsh here since we already know matrix is symmetric)
                eigvals = torch.linalg.eigvalsh(cov)

            if torch.any(eigvals < 0.0):
                raise ValueError(
                    "Value of 'cov' for 'CondMultivariateGaussian' is not symmetric positive semi-definite (contains negative real eigenvalues)."
                )

        if specified_tril:
            return mean, None, cov_tril
        else:
            return mean, cov, None

    def check_support(
        self, data: torch.Tensor, is_scope_data: bool = False
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Multivariate Gaussian distribution, which is:

        .. math::

            \text{supp}(\text{MultivariateGaussian})=(-\infty,+\infty)^k

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

        if scope_data.ndim != 2 or scope_data.shape[1] != len(
            self.scopes_out[0].query
        ):
            raise ValueError(
                f"Expected 'scope_data' to be of shape (n,{len(self.scopes_out[0].query)}), but was: {scope_data.shape}"
            )

        # different to univariate distributions, cannot simply check via torch distribution's support due to possible incomplete data in multivariate case; therefore do it ourselves (not difficult here since support is R)
        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)

        # check for infinite values (may return NaNs despite support)
        valid &= ~scope_data.isinf().sum(dim=1, keepdim=True).bool()

        return valid


@dispatch(memoize=True)  # type: ignore
def marginalize(
    node: CondMultivariateGaussian,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[CondMultivariateGaussian, CondGaussian, None]:
    """Structural marginalization for ``CondMultivariateGaussian`` nodes in the ``torch`` backend.

    Structurally marginalizes the leaf node.
    If the node's scope contains non of the random variables to marginalize, then the node is returned unaltered.
    If the node's scope is fully marginalized over, then None is returned.
    If the node's scope is partially marginalized over, a marginal uni- or multivariate Gaussian is returned instead.

    Args:
        node:
            Node module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect here. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
            Unaltered node if module is not marginalized, marginalized uni- or multivariate Gaussian leaf node, or None if it is completely marginalized.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # scope after marginalization (important: must remain order of scope indices since they map to the indices of the mean vector and covariance matrix!)
    marg_scope = []
    marg_scope_ids = []

    scope = node.scope

    for rv in scope.query:
        if rv not in marg_rvs:
            marg_scope.append(rv)
            marg_scope_ids.append(scope.query.index(rv))

    # return univariate Gaussian if one-dimensional
    if len(marg_scope) == 1:
        return CondGaussian(Scope(marg_scope, scope.evidence))
    # entire node is marginalized over
    elif len(marg_scope) == 0:
        return None
    # node is partially marginalized over
    else:
        return CondMultivariateGaussian(Scope(marg_scope, scope.evidence))


@dispatch(memoize=True)  # type: ignore
def toTorch(
    node: BaseCondMultivariateGaussian,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> CondMultivariateGaussian:
    """Conversion for ``CondMultivariateGaussian`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondMultivariateGaussian(node.scope)


@dispatch(memoize=True)  # type: ignore
def toBase(
    node: CondMultivariateGaussian,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> BaseCondMultivariateGaussian:
    """Conversion for ``CondMultivariateGaussian`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondMultivariateGaussian(node.scope)
