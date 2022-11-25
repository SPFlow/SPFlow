"""Contains Bernoulli leaf layer for SPFlow in the ``base`` backend.
"""
from typing import List, Union, Optional, Iterable, Tuple
import numpy as np
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.data.scope import Scope
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.base.structure.module import Module
from spflow.base.structure.general.nodes.leaves.parametric.bernoulli import (
    Bernoulli,
)


class BernoulliLayer(Module):
    r"""Layer of multiple (univariate) Bernoulli distribution leaf nodes in the ``base`` backend.

    Represents multiple univariate Bernoulli distributions with independent scopes, each with the following probability mass function (PMF):

    .. math::

        \text{PMF}(k)=\begin{cases} p   & \text{if } k=1\\
                                    1-p & \text{if } k=0\end{cases}
        
    where
        - :math:`p` is the success probability in :math:`[0,1]`
        - :math:`k` is the outcome of the trial (0 or 1)

    Attributes:
        p:
            One-dimensional NumPy array containing the success probabilities for each of the independent Bernoulli distributions.
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``Bernoulli`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        p: Union[int, float, List[float], np.ndarray] = 0.5,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``BernoulliLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            p:
                Floating point, list of floats or one-dimensional NumPy array representing the success probabilities of the Bernoulli distributions between zero and one.
                If a single floating point value is given it is broadcast to all nodes.
                Defaults to 0.5.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'BernoulliLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError(
                    "List of scopes for 'BernoulliLayer' was empty."
                )

            self._n_out = len(scope)

        super().__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [Bernoulli(s) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(p)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def p(self) -> np.ndarray:
        """Returns the success probabilities of the represented distributions."""
        return np.array([node.p for node in self.nodes])

    @classmethod
    def accepts(self, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``BernoulliLayer`` can represent one or more univariate nodes with ``MetaType.discrete`` or ``BernoulliType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not Bernoulli.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(
        self, signatures: List[FeatureContext]
    ) -> "BernoulliLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``BernoulliLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not self.accepts(signatures):
            raise ValueError(
                f"'BernoulliLayer' cannot be instantiated from the following signatures: {signatures}."
            )

        p = []
        scopes = []

        for feature_ctx in signatures:

            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if domain == MetaType.Discrete:
                p.append(0.5)
            elif domain == FeatureTypes.Bernoulli:
                # instantiate object
                p.append(domain().p)
            elif isinstance(domain, FeatureTypes.Bernoulli):
                p.append(domain.p)
            else:
                raise ValueError(
                    f"Unknown signature domain {domain} for 'BernoulliLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return BernoulliLayer(scopes, p=p)

    def set_params(
        self, p: Union[int, float, List[float], np.ndarray] = 0.5
    ) -> None:
        """Sets the parameters for the represented distributions.

        Args:
            p:
                Floating point, list of floats or one-dimensional NumPy array representing the success probabilities of the Bernoulli distributions between zero and one.
                If a single floating point value is given it is broadcast to all nodes.
                Defaults to 0.5.
        """
        if isinstance(p, int) or isinstance(p, float):
            p = np.array([p for _ in range(self.n_out)])
        if isinstance(p, list):
            p = np.array(p)
        if p.ndim != 1:
            raise ValueError(
                f"Numpy array of 'p' values for 'BernoulliLayer' is expected to be one-dimensional, but is {p.ndim}-dimensional."
            )
        if p.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'p' values for 'BernoulliLayer' must match number of output nodes {self.n_out}, but is {p.shape[0]}"
            )
        for node_p, node in zip(p, self.nodes):
            node.set_params(node_p)

    def get_params(self) -> Tuple[np.ndarray]:
        """Returns the parameters of the represented distribution.

        Returns:
            One-dimensional NumPy array representing the success probabilities.
        """
        return (self.p,)

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

    def check_support(
        self, data: np.ndarray, node_ids: Optional[List[int]] = None
    ) -> np.ndarray:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or not instances are part of the supports of the Bernoulli distributions, which are:

        .. math::

            \text{supp}(\text{Bernoulli})=\{0,1\}

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

        return np.concatenate(
            [self.nodes[i].check_support(data) for i in node_ids], axis=1
        )


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: BernoulliLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[BernoulliLayer, Bernoulli, None]:
    """Structural marginalization for ``BernoulliLayer`` objects in the ``base`` backend.

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
        new_node = Bernoulli(marg_scopes[0], *marg_params[0])
        return new_node
    else:
        new_layer = BernoulliLayer(
            marg_scopes, *[np.array(p) for p in zip(*marg_params)]
        )
        return new_layer
