"""Contains conditional Gaussian leaf node for SPFlow in the ``torch`` backend.
"""
from typing import Callable, List, Optional, Tuple, Type, Union

import torch
import torch.distributions as D
import tensorly as tl

from spflow.base.structure.general.nodes.leaves.parametric.cond_gaussian import (
    CondGaussian as BaseCondGaussian,
)
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_gaussian import CondGaussian as GeneralCondGaussian
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureType, FeatureTypes, MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaf_node import LeafNode


class CondGaussian(LeafNode):
    r"""Conditional (univariate) Normal distribution leaf node in the ``torch`` backend.

    Represents a conditional univariate Gaussian distribution, with the following probability density function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})

    where
        - :math:`x` the observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Attributes:
        cond_f:
            Optional callable to retrieve the conditional parameter for the leaf node.
            Its output should be a dictionary containing ``mean``,``std`` as keys, and the values should be
            floats, scalar NumPy arrays or scalar PyTorch tensors, where the value for ``std`` should be greater than 0.
    """

    def __init__(self, scope: Scope, cond_f: Optional[Callable] = None) -> None:
        r"""Initializes ``CondGaussian`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``mean``,``std`` as keys, and the values should be
                floats, scalar NumPy arrays or scalar PyTorch tensors, where the value for ``std`` should be greater than 0.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'CondGaussian' should be 1, but was: {len(scope.query)}.")
        if len(scope.evidence) == 0:
            raise ValueError(f"Evidence scope for 'CondGaussian' should not be empty.")

        super().__init__(scope=scope)

        self.set_cond_f(cond_f)
        self.backend = "pytorch"

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``CondGaussian`` can represent a single univariate node with ``MetaType.Continuous`` or ``GamamType`` domain.

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

        # leaf is a continuous Gaussian distribution
        if not (
            domains[0] == FeatureTypes.Continuous
            or domains[0] == FeatureTypes.Gaussian
            or isinstance(domains[0], FeatureTypes.Gaussian)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "CondGaussian":
        """Creates an instance from a specified signature.

        Returns:
            ``CondGaussian`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'CondGaussian' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if (
            domain == MetaType.Continuous
            or domain == FeatureTypes.Gaussian
            or isinstance(domain, FeatureTypes.Gaussian)
        ):
            pass
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'CondGaussian' that was not caught during acception checking."
            )

        return CondGaussian(feature_ctx.scope)

    def set_cond_f(self, cond_f: Optional[Callable] = None) -> None:
        r"""Sets the function to retrieve the node's conditonal parameter.

        Args:
            cond_f:
                Optional callable to retrieve the conditional parameter for the leaf node.
                Its output should be a dictionary containing ``mean``,``std`` as keys, and the values should be
                floats, scalar NumPy arrays or scalar PyTorch tensors, where the value for ``std`` should be greater than 0.
        """
        self.cond_f = cond_f

    def retrieve_params(self, data: torch.Tensor, dispatch_ctx: DispatchContext) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Retrieves the conditional parameter of the leaf node.

        First, checks if conditional parameters (``mean``,``std``) is passed as an additional argument in the dispatch context.
        Secondly, checks if a function (``cond_f``) is passed as an additional argument in the dispatch context to retrieve the conditional parameters.
        Lastly, checks if a ``cond_f`` is set as an attributed to retrieve the conditional parameters.

        Args:
            data:
                Two-dimensional PyTorch tensor containing the data to compute the conditional parameters.
                Each row is regarded as a sample.
            dispatch_ctx:
                Dispatch context.

        Returns:
            Tuple of two scalar PyTorch tensors representing the mean and standard deviation.

        Raises:
            ValueError: No way to retrieve conditional parameters or invalid conditional parameters.
        """
        mean, std, cond_f = None, None, None

        # check dispatch cache for required conditional parameters 'mean', 'std'
        if self in dispatch_ctx.args:
            args = dispatch_ctx.args[self]

            # check if values for 'mean', 'std' are specified (highest priority)
            if "mean" in args:
                mean = args["mean"]
            if "std" in args:
                std = args["std"]
            # check if alternative function to provide 'mean', 'std' is specified (second to highest priority)
            elif "cond_f" in args:
                cond_f = args["cond_f"]
        elif self.cond_f:
            # check if module has a 'cond_f' to provide 'mean','std' specified (lowest priority)
            cond_f = self.cond_f

        # if neither 'mean' or 'std' nor 'cond_f' is specified (via node or arguments)
        if (mean is None or std is None) and cond_f is None:
            raise ValueError(
                "'CondExponential' requires either 'mean' and 'std' or 'cond_f' to retrieve 'mean', 'std' to be specified."
            )

        # if 'mean' or 'std' not already specified, retrieve them
        if mean is None or std is None:
            params = cond_f(data)
            mean = params["mean"]
            std = params["std"]

        if isinstance(mean, float):
            mean = torch.tensor(mean)
        if isinstance(std, float):
            std = torch.tensor(std)

        # check if values for 'mean', 'std' are valid
        if not (torch.isfinite(mean) and torch.isfinite(std)):
            raise ValueError(f"Values for 'mean' and 'std' for 'CondGaussian' must be finite, but were: {mean}, {std}")
        if std <= 0.0:
            raise ValueError(f"Value for 'std' for 'CondGaussian' must be greater than 0.0, but was: {std}")

        return mean, std

    def dist(self, mean: torch.Tensor, std: torch.Tensor) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Args:
            mean:
                Scalar PyTorch tensor representing the mean (:math:`\mu`) of the distribution.
            std:
                Scalar PyTorch tensor representing the standard deviation (:math:`\sigma`) of the distribution (must be greater than 0).

        Returns:
            ``torch.distributions.Normal`` distribution.
        """
        return D.Normal(loc=mean, scale=std)

    def check_support(self, data: torch.Tensor, is_scope_data: bool = False) -> torch.Tensor:
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
        nan_mask = torch.isnan(scope_data)

        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool)
        valid[~nan_mask] = self.dist(torch.tensor(0.0), torch.tensor(1.0)).support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf().squeeze(-1)

        return valid


@dispatch(memoize=True)  # type: ignore
def toTorch(node: BaseCondGaussian, dispatch_ctx: Optional[DispatchContext] = None) -> CondGaussian:
    """Conversion for ``CondGaussian`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CondGaussian(node.scope)


@dispatch(memoize=True)  # type: ignore
def toBase(node: CondGaussian, dispatch_ctx: Optional[DispatchContext] = None) -> BaseCondGaussian:
    """Conversion for ``CondGaussian`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCondGaussian(node.scope)

@dispatch(memoize=True)  # type: ignore
def updateBackend(leaf_node: CondGaussian, dispatch_ctx: Optional[DispatchContext] = None):
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    data = tl.tensor([])
    cond_f = None
    if leaf_node.cond_f != None:
        params = leaf_node.cond_f(data)

        for key in leaf_node.cond_f(params):
            # Update the value for each key
            params[key] = tl.tensor(params[key])
        cond_f = lambda data: params
    return GeneralCondGaussian(scope=leaf_node.scope, cond_f=cond_f)