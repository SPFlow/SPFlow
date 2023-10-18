"""Contains Categorical leaf layer for SPFlow in the ``torch`` backend.
"""
from functools import reduce
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter

from spflow.base.structure.general.layers.leaves.parametric.categorical import (
    CategoricalLayer as BaseCategoricalLayer,
)
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.torch.structure.general.nodes.leaves.parametric.categorical import Categorical
from spflow.torch.structure.module import Module
from spflow.torch.utils.projections import proj_bounded_to_real, proj_real_to_bounded


class CategoricalLayer(Module):
    r"""Layer of multiple (univariate) Categorical distribution leaf nodes in the ``torch`` backend.

    Represents multiple univariate Categorical distributions with independent scopes, each with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k)= p_k
        
    where
        - :math:`k` is an integer associated with a category
        - :math:`p_k` is the success probability of the associated category k in :math:`[0,1]`

    Internally :math:`p` are represented as unbounded parameters that are projected onto the bounded range :math:`[0,1]` for representing the actual success probabilities.

    Attributes:
        k:
            The number of categories
        p_aux:
            Unbounded (scalar PyTorch tensor?) parameter that is projected to yield the actual success probability.
        p:
            Scalar PyTorch tensor representing the selection probabilities of the Categorical distribution (projected from ``p_aux``).
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        k: Union[int, List[int], np.ndarray, torch.Tensor] = 2,
        p: Optional[Union[List[float], List[List[float]], np.ndarray, torch.Tensor]] = None,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``CategoricalLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            k: 
                Positive integer or list of positive integeres representing the number of categories of the Categorical distributions.
                If a single integer is given it is broadcast to all nodes.
                Defaults to 2.
            p:
                List of floating points, list of list of floats or two-dimensional NumPy array representing the success probabilities of the Categorical distributions between zero and one.
                If a single list of floating point values is given it is broadcast to all nodes.
                Defaults to uniformly distributed k values.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'CategoricalLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'CategoricalLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")
            if len(s.evidence) != 0:
                raise ValueError(f"Evidence scope for 'CategoricalLayer' should be empty, but was {s.evidence}.")

        super().__init__(children=[], **kwargs)

        # register auxiliary torch parameter for the number of categories k and selection probabilities p for each implicit node
        self.k_aux = Parameter(requires_grad=False)
        self.p_aux = Parameter()

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(lambda s1, s2: s1.join(s2), self.scopes_out)

        # parse weights
        self.set_params(k, p)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``CategoricalLayer`` can represent one or more univariate nodes with ``MetaType.discrete`` or ``CategoricalType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not Categorical.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "CategoricalLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``CategoricalLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'CategoricalLayer' cannot be instantiated from the following signatures: {signatures}.")

        k = []
        p = []
        scopes = []

        for feature_ctx in signatures:

            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if domain == MetaType.Discrete:
                k.append(2)
                p.append([0.5, 0.5])
            elif domain == FeatureTypes.Categorical:
                # instantiate object
                k.append(domain().k)
                p.append(domain().p)
            elif isinstance(domain, FeatureTypes.Categorical):
                k.append(domain.k)
                p.append(domain.p)
            else:
                raise ValueError(
                    f"Unknown signature domain {domain} for 'CategoricalLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return CategoricalLayer(scopes, k=k, p=p)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out
    
    @property
    def k(self) -> torch.Tensor:
        """Returns the number of categories of the represented distributions"""
        return self.k_aux

    @property
    def p(self) -> torch.Tensor:
        """Returns the selection probabilities of the represented distributions."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.p_aux, lb=0.0, ub=1.0)  # type: ignore
    
    @k.setter
    def k(self, k: Union[int, List[int], np.ndarray, torch.Tensor]) -> None:
        r"""Sets the number of categories for each node

        Args: 
            k: 
                Positive integer or list of positive integeres representing the number of categories of the Categorical distributions.
                If a single integer is given it is broadcast to all nodes.
            
        Raises:
            ValueError: Invalid arguments.
        """
        if isinstance(k, int):
            k = torch.tensor([k for _ in range(self.n_out)])
        elif isinstance(k, list):
            k = torch.tensor(k)
        if k.ndim != 1:
            raise ValueError(f"Tensor of 'k' values for 'CategoricalLayer' is expected to be one-dimensional, but is {k.ndim}-dimensional")
        if k.shape[0] == 1:
            k = torch.hstack([k for _ in range(self.n_out)])
        if k.shape[0] != self.n_out:
            raise ValueError(f"Length of numpy array of 'k' values for 'CategoricalLayer' must match number of output nodes {self.n_out}, but is {k.shape[0]}")
        if torch.any(k<1) or not all(torch.isfinite(k)):
            raise ValueError(f"Values of 'k' for CategoricalLayer' distribution must be positive, but are: {k}")

        self.k_aux.data = k


    @p.setter
    def p(self, p: Union[List[float], List[List[float]], np.ndarray, torch.Tensor]) -> None:
        r"""Sets the success probability.

        Args:
            p:
                List of floating points, list of list of floats, one-dimensional NumPy array or PyTorch tensor representing the selection probabilities of the Categorical distributions between zero and one.
                If a flat list of floating point values is given it is broadcast to all nodes.

        Raises:
            ValueError: Invalid arguments.
        """
        if isinstance(p, (list, np.ndarray)) and isinstance(p[0], float):
            p = torch.tensor([p for _ in range(self.n_out)])
        elif isinstance(p, (list, np.ndarray)) and isinstance(p[0], (list, np.ndarray)):
            p = torch.tensor(p)
        if p.ndim != 2:
            raise ValueError(
                f"Tensor of 'p' values for 'CategoricalLayer' is expected to be two-dimensional, but is {p.ndim}-dimensional."
            )
        if p.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'p' values for 'CategoricalLayer' must match number of output nodes {self.n_out}, but is {p.shape[0]}"
            )
        if torch.any((p < 0.0).float()) or torch.any((p > 1.0).float()) or not torch.all(torch.isfinite(p)):
            raise ValueError(
                f"Values of 'p' for 'CategoricalLayer' distribution must to be between 0.0 and 1.0, but are: {p}"
            )
        if not len(self.k) == len(p) or not all([(self.k[i] == len(p[i]) for i in range(len(p)))]):
            raise ValueError(f"k and the length of p need to match, but were ({self.k}, {len(p)})")

        self.p_aux.data = proj_bounded_to_real(p, lb=0.0, ub=1.0)

    def dist(self, node_ids: Optional[List[int]] = None) -> D.Distribution:
        r"""Returns the PyTorch distributions represented by the leaf layer.

        Args:
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            ``torch.distributions.Categorical`` instance.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Categorical(probs=self.p[node_ids])

    def set_params(self, k: Union[int, List[int], np.ndarray] = 2, p: Optional[Union[int, float, List[float], np.ndarray, torch.Tensor]] = None) -> None:
        """Sets the parameters for the represented distributions.

        Bounded parameter ``p`` is projected onto the unbounded parameter ``p_aux``.

        TODO: projection function

        Args:
            k:
                A positive integer, a list of positive integers or one-dimensional NumPy array representing the number of categories of the distribution.
                Defaults to 2.
            p:
                A floating point, list of floats or one-dimensional NumPy array representing the selection probabilities of the Categorical distributions between zero and one.
                If a flat list of floating point values is given it is broadcast to all nodes.
                Defaults to uniformly distributed k categories.
        """
        self.k = k

        if isinstance(k, int) and k > 0:
            if p is None:
                p = [1./k for _ in range(k)]
            elif isinstance(p, list):
                if all(i is None for i in p):
                    p = [1./k for _ in range(k)]
            elif isinstance(p, np.ndarray):
                if np.all(np.isnan(p)):
                    p = [1./k for _ in range(k)]
        elif isinstance(k, list) or isinstance(k, np.ndarray) and np.all(k>0):
            if p is None:
                p = [[1./kk for _ in range(kk)] for kk in k]
            elif isinstance(p, list):
                if all(i is None for i in p):
                    p = [[1./kk for _ in range(kk)] for kk in k]
            elif isinstance(p, np.ndarray):
                if np.all(np.isnan(p)):
                    p = [[1./kk for _ in range(kk)] for kk in k]

        self.p = p

    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the parameters of the represented distribution.

        Returns:
            One-dimensional PyTorch tensor representing the success probabilities.
        """
        return (self.k, self.p,)

    def check_support(
        self,
        data: torch.Tensor,
        node_ids: Optional[List[int]] = None,
        is_scope_data: bool = False,
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or not instances are part of the supports of the Categorical distributions, which are:

        .. math::

            \text{supp}(\text{Categorical})=\{0, 1, ..., k-1\}

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional PyTorch tensor containing sample instances.
                Each row is regarded as a sample.
                Assumes that relevant data is located in the columns corresponding to the scope indices.
                Unless ``is_scope_data`` is set to True, it is assumed that the relevant data is located in the columns corresponding to the scope indices.
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.
            is_scope_data:
                Boolean indicating if the given data already contains the relevant data for the leafs' scope in the correct order (True) or if it needs to be extracted from the full data set.
                Note, that this should already only contain only the data according (and in order of) ``node_ids``.
                Defaults to False.

        Returns:
            Two dimensional PyTorch tensor indicating for each instance and node, whether they are part of the support (True) or not (False).
            Each row corresponds to an input sample.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        if is_scope_data:
            scope_data = data
        else:
            # all query scopes are univariate
            scope_data = data[:, [self.scopes_out[node_id].query[0] for node_id in node_ids]]

        # NaN values do not throw an error but are simply flagged as False
        valid = self.dist(node_ids).support.check(scope_data)  # type: ignore

        # nan entries (regarded as valid)
        nan_mask = torch.isnan(scope_data)

        # set nan_entries back to True
        valid[nan_mask] = True

        # check for infinite values
        valid[~nan_mask & valid] &= ~scope_data[~nan_mask & valid].isinf()

        return valid


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: CategoricalLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[CategoricalLayer, Categorical, None]:
    """Structural marginalization for ``CategoricalLayer`` objects in the ``torch`` backend.

    Structurally marginalizes the specified layer module.
    If the layer's scope contains non of the random variables to marginalize, then the layer is returned unaltered.
    If the layer's scope is fully marginalized over, then None is returned.

    Args:
        layer:
            Layer module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect here. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Unaltered leaf layer or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    marginalized_node_ids = []
    marginalized_scopes = []

    for i, scope in enumerate(layer.scopes_out):

        # compute marginalized query scope
        marg_scope = [rv for rv in scope.query if rv not in marg_rvs]

        # node not marginalized over
        if len(marg_scope) == 1:
            marginalized_node_ids.append(i)
            marginalized_scopes.append(scope)

    if len(marginalized_node_ids) == 0:
        return None
    elif len(marginalized_node_ids) == 1 and prune:
        node_id = marginalized_node_ids.pop()
        return Categorical(scope=marginalized_scopes[0], k=layer.k[node_id].item(), p=layer.p[node_id].detach().numpy().tolist())
    else:
        return CategoricalLayer(scope=marginalized_scopes, k=layer.k[marginalized_node_ids].detach(), p=layer.p[marginalized_node_ids].detach())


@dispatch(memoize=True)  # type: ignore
def toTorch(layer: BaseCategoricalLayer, dispatch_ctx: Optional[DispatchContext] = None) -> CategoricalLayer:
    """Conversion for ``CategoricalLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return CategoricalLayer(scope=layer.scopes_out, p=layer.p)


@dispatch(memoize=True)  # type: ignore
def toBase(layer: CategoricalLayer, dispatch_ctx: Optional[DispatchContext] = None) -> BaseCategoricalLayer:
    """Conversion for ``CategoricalLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseCategoricalLayer(scope=layer.scopes_out, p=layer.p.detach().numpy())
