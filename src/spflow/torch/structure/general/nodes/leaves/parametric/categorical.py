"""Contains Categorical leaf node for SPFlow in the ``torch`` backend.
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter

from spflow.base.structure.general.nodes.leaves.parametric.categorical import (
    Categorical as BaseCategorical,
)
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


class Categorical(LeafNode):
    r"""(Univariate) Categorical distribution leaf node in the ``torch`` backend.

    Represents an univariate Categorical distribution, with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k)= p_k  
        
    where
        - :math:`k` is a positive integer associated with a category
        - :math:`p_k` is the success probability of the associated category k in :math:`[0,1]`

    Internally :math:`p_k` are represented as unbounded parameters that are projected onto the bounded range :math:`[0,1]` for representing the actual probabilities.

    Attributes:
        k:
            The number of categories
        p_aux:
            Unbounded (scalar PyTorch tensor?) parameter that is projected to yield the actual success probability.
        p:
            Scalar PyTorch tensor representing the selection probabilities of the Categorical distribution (projected from ``p_aux``).
    """

    def __init__(self, scope: Scope, k: int = 2, p: List[float] = None) -> None:
        r"""Initializes ``Categorical`` leaf node.

        Args:
            scope:
                Scope object specifying the scope of the distribution.
            k: 
                A positive integer representing the number of categories.
                Defaults to 2.
            p:
                A list of floating point values representing the probability that the k-th category is selected, each in the range [0, 1].
                Defaults to uniformly distributed over k categories.

        Raises:
            ValueError: Invalid arguments.
        """
        if len(scope.query) != 1:
            raise ValueError(f"Query scope size for 'Categorical' should be 1, but was {len(scope.query)}.")
        if len(scope.evidence) != 0:
            raise ValueError(f"Evidence scope for 'Categorical' should be empty, but was {scope.evidence}.")
        if k is None or not isinstance(k, int) or k < 1:
            raise ValueError(f"Number of categories needs to a positive integer, but was: {k}")
        super().__init__(scope=scope)

        # register auxiliary torch parameter for the number of categories k and the selection probability p
        self.k_aux = Parameter(requires_grad=False)
        self.p_aux = Parameter()

        if p is None: 
            p = [1.0/k for i in range(k)]
        if len(p) != k:
            raise ValueError(f"p needs to be the length of k, but len(p) and k were: ({len(p)}, {k})")


        # set parameters
        self.set_params(k, p)


    @property
    def k(self) -> torch.Tensor:
        """Returns number of categories"""
        return self.k_aux

    @property
    def p(self) -> torch.Tensor:
        """Returns the selection probabilities associated with each category"""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)  # type: ignore
    

    @k.setter
    def k(self, k: int) -> None:
        r"""Sets the number of categories

        Args:
            k: 
                Positive integer representing the number of categories in the distribution
                
        Raises:
            ValueError: Invalid arguments.
        """
        if k < 0 or not np.isfinite(k):
            raise ValueError(f"Value of 'k' for 'Categorical' must be a positive integer, but was {k}")
        self.k_aux.data = k
        

    @p.setter
    def p(self, p: List[float]) -> None:
        r"""Sets the success probability.

        Args:
            p:
                List of floating points representing the selection probability of the associated category, each in :math:`[0,1]`.

        Raises:
            ValueError: Invalid arguments.
        """
        p = np.array(p)
        if (not all(p >= 0.0)) or (not all(p <= 1.0)) or (not all(np.isfinite(p))):
            raise ValueError(f"All values of 'p' for 'Categorical' must to be between 0.0 and 1.0, but were: {p}")
        if not np.isclose(sum(p), 1.0):
            raise ValueError(f"The sum of all values in p needs to be 1.0, but was: {sum(p)}") 
        
        self.p_aux.data = proj_bounded_to_real(torch.tensor(p), lb=0.0, ub=1.0)


    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``Categorical`` can represent a single univariate node with ``MetaType.Discrete`` or ``CategoricalType`` domain.

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

        # leaf is a discrete Categorical distribution
        if not (
            domains[0] == FeatureTypes.Discrete
            or domains[0] == FeatureTypes.Categorical
            or isinstance(domains[0], FeatureTypes.Categorical)
        ):
            return False

        return True
    

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "Categorical":
        """Creates an instance from a specified signature.

        Returns:
            ``Categorical`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'Categorical' cannot be instantiated from the following signatures: {signatures}.")

        # get single output signature
        feature_ctx = signatures[0]
        domain = feature_ctx.get_domains()[0]

        # read or initialize parameters
        if domain == MetaType.Discrete:
            k = 2
            p = [0.5, 0.5]
        elif domain == FeatureTypes.Categorical:
            # instantiate object
            k = domain().k
            p = domain().p
        elif isinstance(domain, FeatureTypes.Categorical):
            k = domain.k
            p = domain.p
        else:
            raise ValueError(
                f"Unknown signature type {domain} for 'Categorical' that was not caught during acception checking."
            )

        return Categorical(feature_ctx.scope, k=k, p=p)
    

    @property
    def dist(self) -> D.Distribution:
        r"""Returns the PyTorch distribution represented by the leaf node.

        Returns:
            ``torch.distributions.Categorical`` instance.
        """
        return D.Categorical(probs=self.p)
    

    def set_params(self, k: int, p: List[float]) -> None:
        """Sets the parameters for the represented distribution.

        Bounded parameter ``p`` is projected onto the unbounded parameter ``p_aux``.

        TODO: projection function

        Args:
            k:
                Positive integer representing the number of categories in the distribution
            p:
                List of floating point values representing the selection probabilities of the Categorical distribution between zero and one.
        """
        # TODO: do we need to check values here?
        self.k = torch.tensor(int(k))
        self.p = torch.FloatTensor(p)


    def get_params(self) -> Tuple[int, float]:
        """Returns the parameters of the represented distribution.

        Returns:
            Floating point value representing the success probability.
        """
        return int(self.k.data.cpu().numpy()), self.p.data.cpu().numpy()  # type: ignore
    

    def check_support(self, data: torch.Tensor, is_scope_data: bool = False) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distribution.

        Determines whether or not instances are part of the support of the Categorical distribution, which is:

        .. math::

            \text{supp}(\text{Categorical})=\{0,1, ..., k-1\}

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
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf().squeeze(-1)

        return valid


@dispatch(memoize=True)  # type: ignore
def toTorch(node: BaseCategorical, dispatch_ctx: Optional[DispatchContext] = None) -> Categorical:
    """Conversion for ``Categorical`` from ``base`` backend to ``torch`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return Categorical(node.scope, *node.get_params())


@dispatch(memoize=True)  # type: ignore
def toBase(node: Categorical, dispatch_ctx: Optional[DispatchContext] = None) -> BaseCategorical:
    """Conversion for ``Categorical`` from ``torch`` backend to ``base`` backend.

    Args:
        node:
            Leaf node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCategorical(node.scope, *node.get_params())
