"""Contains Categorical leaf layer for SPFlow in the ``base`` backend.
"""
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.base.structure.general.nodes.leaves.parametric.categorical import Categorical
from spflow.base.structure.module import Module
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class CategoricalLayer(Module):
    r"""Layer of multiple (univariate) Categorical distribution leaf nodes in the ``base`` backend.

    Represents multiple univariate Categorical distributions with independent scopes, each with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k)= p_k
        
    where
        - :math:`k` is an integer associated with a category
        - :math:`p_k` is the success probability of the associated category k in :math:`[0,1]`

    Attributes:
        k:
            One-dimensional array containing containing the number of categories for each of the independent Categorical distributions.
        p:
            Two-dimensional NumPy array containing the success probabilities for each of the independent Categorical distributions.
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``Categorical`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        k: Union[int, List[int], np.ndarray] = 2,
        p: Optional[Union[List[float], List[List[float]], np.ndarray]] = None,
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

        super().__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [Categorical(s) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(k, p)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out
    
    @property
    def k(self) -> np.ndarray:
        """Returns the number of categories of the represented distributions"""
        return np.array([node.k for node in self.nodes])

    @property
    def p(self) -> np.ndarray:
        """Returns the selection probabilities of the represented distributions."""
        return np.array([node.p for node in self.nodes])

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

    def set_params(self, k: Union[int, List[int], np.ndarray] = 2, p: Optional[Union[List[float], List[List[float]], np.ndarray]] = None) -> None:
        """Sets the parameters for the represented distributions.

        Args:
            k:
                A positive integer, a list of positive integers or one-dimensional NumPy array representing the number of categories of the distribution.
                Defaults to 2.
            p:
                A floating point, list of floats or one-dimensional NumPy array representing the selection probabilities of the Categorical distributions between zero and one.
                If a flat list of floating point values is given it is broadcast to all nodes.
                Defaults to uniformly distributed k categories.
        """
        if isinstance(k, int) and k > 0:
            if p is None:
                p = [1./k for _ in range(k)]
            elif isinstance(p, list):
                if all(i is None for i in p):
                    p = [1./k for _ in range(k)]
            elif isinstance(p, np.ndarray):
                if np.all(np.isnan(p)):
                    p = [1./k for _ in range(k)]
            k = np.array([k for _ in range(self.n_out)])
        elif isinstance(k, list) or isinstance(k, np.ndarray) and np.all(k>0):
            if p is None:
                p = [[1./kk for _ in range(kk)] for kk in k]
            elif isinstance(p, list):
                if all(i is None for i in p):
                    p = [[1./kk for _ in range(kk)] for kk in k]
            elif isinstance(p, np.ndarray):
                if np.all(np.isnan(p)):
                    p = [[1./kk for _ in range(kk)] for kk in k]
            k = np.array(k)
        else:
            raise ValueError(f"k needs to be a positive integer or list or numpy array thereof, but was {k}")
        if not type(p) in [list, np.ndarray]:
            raise ValueError(f"p needs to be of type list or numpy array and non-empty, but was {type(p), p}")
        if isinstance(p[0], float):
            p = [p for _ in range(self.n_out)]
        p = np.array(p)
        if p.ndim != 2:
            raise ValueError(
                f"Numpy array of 'p' values for 'CategoricalLayer' is expected to be two-dimensional, but is {p.ndim}-dimensional. {p}"
            )
        if p.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'p' values for 'CategoricalLayer' must match number of output nodes {self.n_out}, but is {p.shape[0]}"
            )
        for node_k, node_p, node in zip(k, p, self.nodes):
            node.set_params(node_k, node_p)

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the parameters of the represented distribution.

        Returns:
            Two-dimensional NumPy array representing the success probabilities.
        """
        return (self.k, self.p,)

    def dist(self, node_ids: Optional[List[int]] = None) -> List[rv_frozen]:
        r"""Returns the SciPy distributions represented by the leaf layer.

        Args:
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            List of ``scipy.stats.distributions.rv_frozen`` distributions.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return [self.nodes[i].dist for i in node_ids]

    def check_support(self, data: np.ndarray, node_ids: Optional[List[int]] = None) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or not instances are part of the supports of the Categorical distributions, which are:

        .. math::

            \text{supp}(\text{Categorical})=\{0, 1, ..., k-1\}

        Additionally, NaN values are regarded as being part of the support (they are marginalized over during inference).

        Args:
            data:
                Two-dimensional NumPy array containing sample instances.
                Each row is regarded as a sample.
                Assumes that relevant data is located in the columns corresponding to the scope indices.
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            Two dimensional NumPy array indicating for each instance and node, whether they are part of the support (True) or not (False).
            Each row corresponds to an input sample.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return np.concatenate([self.nodes[i].check_support(data) for i in node_ids], axis=1)


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: CategoricalLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[CategoricalLayer, Categorical, None]:
    """Structural marginalization for ``CategoricalLayer`` objects in the ``base`` backend.

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

    # marginalize nodes
    marg_scopes = []
    marg_params = []

    for node in layer.nodes:
        marg_node = marginalize(node, marg_rvs, prune=prune)

        if marg_node is not None:
            marg_scopes.append(marg_node.scope)
            marg_params.append(marg_node.get_params())

    if len(marg_scopes) == 0:
        return None
    elif len(marg_scopes) == 1 and prune:
        new_node = Categorical(marg_scopes[0], *marg_params[0])
        return new_node
    else:
        new_layer = CategoricalLayer(marg_scopes, *[np.array(p) for p in zip(*marg_params)])
        return new_layer
