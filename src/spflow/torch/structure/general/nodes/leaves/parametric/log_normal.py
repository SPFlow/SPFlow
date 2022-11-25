"""Contains Log-Normal leaf node for SPFlow in the ``torch`` backend.
"""
import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import List, Tuple, Optional
from spflow.torch.utils.projections import (
    proj_bounded_to_real,
    proj_real_to_bounded,
)
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import MetaType, FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaf_node import LeafNode
from spflow.base.structure.general.nodes.leaves.parametric.log_normal import (
    LogNormal as BaseLogNormal,
)


class LogNormal(LeafNode):
    r"""(Univariate) Log-Normal distribution leaf node in the ``torch`` backend.

    Represents an univariate Log-Normal distribution, with the following probability distribution function (PDF):

    .. math::

        \text{PDF}(x) = \frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln(x)-\mu)^2}{2\sigma^2}\right)

    where
        - :math:`x` is an observation
        - :math:`\mu` is the mean
        - :math:`\sigma` is the standard deviation

    Internally :math:`\mu,\sigma` are represented as unbounded parameters that are projected onto the bounded range :math:`(0,\infty)` for representing the actual shape and rate parameters, respectively.

    Attributes:
        mean:
            Scalar PyTorch tensor representing the mean (:math:`\mu`) of the Gamma distribution.
        std_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual standard deviation.
        std:
            Scalar PyTorch tensor representing the standard deviation (:math:`\sigma`) of the Gaussian distribution, greater than 0 (projected from ``std_aux``).
    """

    def __init__(
        self,
        scope: Scope,
        mean: Optional[float] = 0.0,
        std: Optional[float] = 1.0,
    ) -> None:
        r"""Initializes ``LogNormal`` leaf node.

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
            raise ValueError(
                f"Query scope size for 'LogNormal' should be 1, but was: {len(scope.query)}."
            )
        if len(scope.evidence) != 0:
            raise ValueError(
                f"Evidence scope for 'LogNormal' should be empty, but was {scope.evidence}."
            )

        super().__init__(scope=scope)

        # register mean as torch parameter
        self.mean = Parameter()
        # register auxiliary torch paramter for standard deviation
        self.std_aux = Parameter()

        # set parameters
        self.set_params(mean, std)

    @property
    def std(self) -> torch.Tensor:
        """TODO"""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.std_aux, lb=0.0)  # type: ignore

    @classmethod
    def accepts(self, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``LogNormal`` can represent a single univariate node with ``MetaType.Continuous`` or ``LogNormalType`` domain.

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

        # leaf is a continuous Log-Normal distribution
        if not (
            domains[0] == FeatureTypes.Continuous
            or domains[0] == FeatureTypes.LogNormal
            or isinstance(domains[0], FeatureTypes.LogNormal)
        ):
            return False

        return True

    @classmethod
    def from_signatures(self, signatures: List[FeatureContext]) -> "LogNormal":
        """Creates an instance from a specified signature.

        Returns:
            ``LogNormal`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not self.accepts(signatures):
            raise ValueError(
                f"'LogNormal' cannot be instantiated from the following signatures: {signatures}."
            )

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Continuous:
            mean, std = 0.0, 1.0
        elif domain == FeatureTypes.LogNormal:
            # instantiate object
            domain = domain()
            mean, std = domain.mean, domain.std
        elif isinstance(domain, FeatureTypes.LogNormal):
            mean, std = domain.mean, domain.std
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'LogNormal' that was not caught during acception checking."
            )

        return LogNormal(feature_ctx.scope, mean=mean, std=std)

    @property
    def dist(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.LogNormal`` instance.
        """
        return D.LogNormal(loc=self.mean, scale=self.std)

    def set_params(self, mean: float, std: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            mean:
                Floating point value representing the mean (:math:`\mu`) of the distribution.
            std:
                Floating point values representing the standard deviation (:math:`\sigma`) of the distribution (must be greater than 0).
        """
        if not (np.isfinite(mean) and np.isfinite(std)):
            raise ValueError(
                f"Values for 'mean' and 'std' for 'Gaussian' must be finite, but were: {mean}, {std}"
            )
        if std <= 0.0:
            raise ValueError(
                f"Value for 'std' for 'Gaussian' must be greater than 0.0, but was: {std}"
            )

        self.mean.data = torch.tensor(float(mean))
        self.std_aux.data = proj_bounded_to_real(
            torch.tensor(float(std)), lb=0.0
        )

    def get_params(self) -> Tuple[float, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of the floating point values representing the mean and standard deviation.
        """
        return self.mean.data.cpu().numpy(), self.std.data.cpu().numpy()  # type: ignore

    def check_support(
        self, data: torch.Tensor, is_scope_data: bool = False
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Log-Normal distribution, which is:

        .. math::

            \text{supp}(\text{LogNormal})=(0,\infty)

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
        valid[~nan_mask] = self.dist.support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check for infinite values
        valid[~nan_mask & valid] &= (
            ~scope_data[~nan_mask & valid].isinf().squeeze(-1)
        )

        return valid


@dispatch(memoize=True)  # type: ignore
def toTorch(
    node: BaseLogNormal, dispatch_ctx: Optional[DispatchContext] = None
) -> LogNormal:
    """Conversion for ``LogNormal`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return LogNormal(node.scope, *node.get_params())


@dispatch(memoize=True)  # type: ignore
def toBase(
    node: LogNormal, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseLogNormal:
    """Conversion for ``LogNormal`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseLogNormal(node.scope, *node.get_params())
