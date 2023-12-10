"""Contains Gamma leaf node for SPFlow in the ``torch`` backend.
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter

from spflow.base.structure.general.nodes.leaves.parametric.gamma import (
    Gamma as BaseGamma,
)
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gamma import Gamma as GeneralGamma
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes, MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaf_node import LeafNode
from spflow.torch.utils.projections import proj_bounded_to_real, proj_real_to_bounded


class Gamma(LeafNode):
    r"""(Univariate) Gamma distribution leaf node in the ``base`` backend.

    Represents an univariate Exponential distribution, with the following probability distribution function (PDF):

    .. math::
    
        \text{PDF}(x) = \begin{cases} \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x} & \text{if } x > 0\\
                                      0 & \text{if } x <= 0\end{cases}

    where
        - :math:`x` is the input observation
        - :math:`\Gamma` is the Gamma function
        - :math:`\alpha` is the shape parameter
        - :math:`\beta` is the rate parameter

    Internally :math:`\alpha,\beta` are represented as unbounded parameters that are projected onto the bounded range :math:`(0,\infty)` for representing the actual shape and rate parameters, respectively.

    Attributes:
        alpha_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual shape parameter.
        alpha:
            Scalar PyTorch tensor representing the shape parameter (:math:`\alpha`) of the Gamma distribution, greater than 0 (projected from ``alpha_aux``).
        beta_aux:
            Unbounded scalar PyTorch parameter that is projected to yield the actual rate parameter.
        beta:
            Scalar PyTorch tensor representing the rate parameter (:math:`\beta`) of the Gamma distribution, greater than 0 (projected from ``beta_aux``).
    """

    def __init__(self, scope: Scope, alpha: float = 1.0, beta: float = 1.0) -> None:
        r"""Initializes ``Exponential`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            alpha:
                Floating point value representing the shape parameter (:math:`\alpha`), greater than 0.
                Defaults to 1.0.
            beta:
                Floating point value representing the rate parameter (:math:`\beta`), greater than 0.
                Defaults to 1.0.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Gamma' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Gamma' should be empty, but was {scope.evidence}.")

        super().__init__(scope=scope)
        self.backend = "pytorch"

        # register auxiliary torch parameters for alpha and beta
        self.alpha_aux = Parameter()
        self.beta_aux = Parameter()

        # set parameters
        self.set_params(alpha, beta)

    @property
    def alpha(self) -> torch.Tensor:
        """Returns the shape parameter."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.alpha_aux, lb=0.0)  # type: ignore

    @property
    def beta(self) -> torch.Tensor:
        """Returns the rate parameter."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.beta_aux, lb=0.0)  # type: ignore

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Gamma`` can represent a single univariate node with ``MetaType.Continuous`` or ``GammaType`` domain.

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
        if len(domains) != 1 or len(feature_ctx.scope.query) != len(domains) or len(feature_ctx.scope.evidence) != 0:
            return False

        # leaf is a continuous Gamma distribution
        if not (
            domains[0] == FeatureTypes.Continuous
            or domains[0] == FeatureTypes.Gamma
            or isinstance(domains[0], FeatureTypes.Gamma)
        ):
            return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "Gamma":
        """Creates an instance from a specified signature.

        Returns:
            ``Gamma`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'Gamma' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Continuous:
            alpha, beta = 1.0, 1.0
        elif domain == FeatureTypes.Gamma:
            # instantiate object
            domain = domain()
            alpha, beta = domain.alpha, domain.beta
        elif isinstance(domain, FeatureTypes.Gamma):
            alpha, beta = domain.alpha, domain.beta
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Gamma' that was not caught during acception checking."
            )

        return Gamma(feature_ctx.scope, alpha=alpha, beta=beta)

    @property
    def dist(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.Gamma`` instance.
        """
        return D.Gamma(concentration=self.alpha, rate=self.beta)

    def set_params(self, alpha: float, beta: float) -> None:
        r"""Sets the parameters for the represented distribution.

        Args:
            alpha:
                Floating point value representing the shape parameter (:math:`\alpha`), greater than 0.
            beta:
                Floating point value representing the rate parameter (:math:`\beta`), greater than 0.
        """
        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError(f"Value of alpha for Gamma distribution must be greater than 0, but was: {alpha}")
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError(f"Value of beta for Gamma distribution must be greater than 0, but was: {beta}")

        self.alpha_aux.data = proj_bounded_to_real(torch.tensor(float(alpha), dtype=self.dtype, device=self.device), lb=0.0)
        self.beta_aux.data = proj_bounded_to_real(torch.tensor(float(beta), dtype=self.dtype, device=self.device), lb=0.0)

    def get_trainable_params(self) -> Tuple[float, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of the floating points representing the shape and rate parameters.
        """
        #return self.alpha.data.cpu().numpy(), self.beta.data.cpu().numpy()  # type: ignore
        return [self.alpha_aux, self.beta_aux]  # type: ignore

    def get_params(self) -> Tuple[float, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of the floating points representing the shape and rate parameters.
        """
        return self.alpha.data.cpu().numpy(), self.beta.data.cpu().numpy()  # type: ignore

    def check_support(self, data: torch.Tensor, is_scope_data: bool = False) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or note instances are part of the support of the Gamma distribution, which is:

        .. math::

            \text{supp}(\text{Gamma})=(0,+\infty)

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

        valid = torch.ones(scope_data.shape[0], 1, dtype=torch.bool, device=self.device)
        valid[~nan_mask] = self.dist.support.check(scope_data[~nan_mask]).squeeze(-1)  # type: ignore

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf().squeeze(-1)

        return valid

    def to_dtype(self, dtype):
        self.dtype = dtype
        self.set_params(self.alpha.data, self.beta.data)

    def to_device(self, device):
        self.device = device
        self.set_params(self.alpha.data, self.beta.data)


@dispatch(memoize=True)  # type: ignore
def toTorch(node: BaseGamma, dispatch_ctx: Optional[DispatchContext] = None) -> Gamma:
    """Conversion for ``Gamma`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return Gamma(node.scope, *node.get_trainable_params())


@dispatch(memoize=True)  # type: ignore
def toBase(node: Gamma, dispatch_ctx: Optional[DispatchContext] = None) -> BaseGamma:
    """Conversion for ``Gamma`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseGamma(node.scope, *node.get_trainable_params())

@dispatch(memoize=True)  # type: ignore
def updateBackend(leaf_node: Gamma, dispatch_ctx: Optional[DispatchContext] = None):
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return GeneralGamma(scope=leaf_node.scope, alpha=leaf_node.alpha.data.item(), beta=leaf_node.beta.data.item())