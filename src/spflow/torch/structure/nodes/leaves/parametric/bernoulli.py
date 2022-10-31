# -*- coding: utf-8 -*-
"""Contains Bernoulli leaf node for SPFlow in the ``torch`` backend.
"""
import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter
from typing import List, Tuple, Optional, Union, Type
from .projections import proj_bounded_to_real, proj_real_to_bounded
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import MetaType, FeatureType, FeatureTypes
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.nodes.node import LeafNode
from spflow.base.structure.nodes.leaves.parametric.bernoulli import (
    Bernoulli as BaseBernoulli,
)


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
            raise ValueError(
                f"Query scope size for 'Bernoulli' should be 1, but was: {len(scope.query)}"
            )
        if len(scope.evidence) != 0:
            raise ValueError(
                f"Evidence scope for 'Bernoulli' should be empty, but was {scope.evidence}."
            )

        super(Bernoulli, self).__init__(scope=scope)

        # register auxiliary torch parameter for the success probability p
        self.p_aux = Parameter()

        # set parameters
        self.set_params(p)

    @property
    def p(self) -> torch.Tensor:
        """TODO"""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)  # type: ignore

    @p.setter
    def p(self, p: float) -> None:
        """TODO"""
        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError(
                f"Value of 'p' for 'Bernoulli' must to be between 0.0 and 1.0, but was: {p}"
            )

        self.p_aux.data = proj_bounded_to_real(
            torch.tensor(float(p)), lb=0.0, ub=1.0
        )

    @classmethod
    def accepts(self, signatures: List[Tuple[List[Union[MetaType, FeatureType, Type[FeatureType]]], Scope]]) -> bool:
        """TODO"""
        # leaf only has one output
        if len(signatures) != 1:
            return False

        # get single output signature
        types, scope = signatures[0]

        # leaf is a single non-conditional univariate node
        if len(types) != 1 or len(scope.query) != len(types) or len(scope.evidence) != 0:
            return False

        # leaf is a discrete Bernoulli distribution
        if not (types[0] == FeatureTypes.Discrete or types[0] == FeatureTypes.Bernoulli or isinstance(types[0], FeatureTypes.Bernoulli)):
            return False

        return True

    @classmethod
    def from_signatures(self, signatures: List[Tuple[List[Union[MetaType, FeatureType, Type[FeatureType]]], Scope]]) -> "Bernoulli":
        """TODO"""
        if not self.accepts(signatures):
            raise ValueError(f"'Bernoulli' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        types, scope = signatures[0]
        type = types[0]

        # read or initialize parameters
        if type == MetaType.Discrete:
            p = 0.5
        elif type == FeatureTypes.Bernoulli:
            # instantiate object
            p = type().p
        elif isinstance(type, FeatureTypes.Bernoulli):
            p = type.p
        else:
            raise ValueError(f"Unknown signature type {type} for 'Bernoulli' that was not caught during acception checking.")

        return Bernoulli(scope, p=p)

    @property
    def dist(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.Bernoulli`` instance.
        """
        return D.Bernoulli(probs=self.p)

    def set_params(self, p: float) -> None:
        """Sets the parameters for the represented distribution.

        Bounded parameter ``p`` is projected onto the unbounded parameter ``p_aux``.

        TODO: projection function

        Args:
            p:
                Floating point value representing the success probability of the Bernoulli distribution between zero and one.
        """
        self.p = torch.tensor(float(p))

    def get_params(self) -> Tuple[float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the success probability.
        """
        return (self.p.data.cpu().numpy(),)  # type: ignore

    def check_support(
        self, data: torch.Tensor, is_scope_data: bool = False
    ) -> torch.Tensor:
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
    node: BaseBernoulli, dispatch_ctx: Optional[DispatchContext] = None
) -> Bernoulli:
    """Conversion for ``Bernoulli`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return Bernoulli(node.scope, *node.get_params())


@dispatch(memoize=True)  # type: ignore
def toBase(
    node: Bernoulli, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseBernoulli:
    """Conversion for ``Bernoulli`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseBernoulli(node.scope, *node.get_params())
