"""Contains Exponential leaf layer for SPFlow in the ``base`` backend.
"""
from typing import Iterable, List, Optional, Tuple, Type, Union

import numpy as np
from scipy.stats.distributions import rv_frozen  # type: ignore

from spflow.base.structure.general.nodes.leaves.parametric.exponential import (
    Exponential,
)
from spflow.tensorly.structure.spn.layers.leaves.parametric import ExponentialLayer as GeneralExponentialLayer
from spflow.tensorly.structure.module import Module
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureType, FeatureTypes
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


class ExponentialLayer(Module):
    r"""Layer of multiple (univariate) Exponential distribution leaf nodes in the ``base`` backend.

    Represents multiple univariate Exponential distributions with independent scopes, each with the following probability distribution function (PDF):

    .. math::
        
        \text{PDF}(x) = \begin{cases} \lambda e^{-\lambda x} & \text{if } x > 0\\
                                      0                      & \text{if } x <= 0\end{cases}
    
    where
        - :math:`x` is the input observation
        - :math:`\lambda` is the rate parameter
    
    Attributes:
        l:
            One-dimensional NumPy array containing the rate parameters (:math:`\lambda`) for each of the independent Exponential distributions.
        scopes_out:
            List of scopes representing the output scopes.
        nodes:
            List of ``Exponential`` objects for the nodes in this layer.
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        l: Union[int, float, List[float], np.ndarray] = 1.0,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``ExponentialLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            l:
                Floating point, list of floats or one-dimensional NumPy array representing the rate parameters (:math:`\lambda`) of the Exponential distributions (must be greater than 0).
                If a single floating point value is given it is broadcast to all nodes.
                Defaults to 1.0.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'ExponentialLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'ExponentialLayer' was empty.")

            self._n_out = len(scope)

        super().__init__(children=[], **kwargs)

        # create leaf nodes
        self.nodes = [Exponential(s) for s in scope]

        # compute scope
        self.scopes_out = scope

        # parse weights
        self.set_params(l)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def l(self) -> np.ndarray:
        """Returns the rate parameters of the represented distributions."""
        return np.array([node.l for node in self.nodes])

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``ExponentialLayer`` can represent one or more univariate nodes with ``MetaType.Continuous`` or ``ExponentialType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not Exponential.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "ExponentialLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``ExponentialLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(f"'ExponentialLayer' cannot be instantiated from the following signatures: {signatures}.")

        l = []
        scopes = []

        for feature_ctx in signatures:

            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if domain == MetaType.Continuous:
                l.append(1.0)
            elif domain == FeatureTypes.Exponential:
                # instantiate object
                l.append(domain().l)
            elif isinstance(domain, FeatureTypes.Exponential):
                l.append(domain.l)
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'ExponentialLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return ExponentialLayer(scopes, l=l)

    def set_params(self, l: Union[int, float, List[float], np.ndarray]) -> None:
        r"""Sets the parameters for the represented distributions.

        Args:
            l:
                Floating point, list of floats or one-dimensional NumPy array representing the rate parameters (:math:`\lambda`) of the Exponential distributions (must be greater than 0).
                If a single floating point value is given it is broadcast to all nodes.
        """
        if isinstance(l, int) or isinstance(l, float):
            l = np.array([float(l) for _ in range(self.n_out)])
        if isinstance(l, list):
            l = np.array(l)
        if l.ndim != 1:
            raise ValueError(
                f"Numpy array of 'l' values for 'ExponentialLayer' is expected to be one-dimensional, but is {l.ndim}-dimensional."
            )
        if l.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'l' values for 'ExponentialLayer' must match number of output nodes {self.n_out}, but is {l.shape[0]}"
            )

        for node_l, node in zip(l, self.nodes):
            node.set_params(node_l)

    def get_params(self) -> Tuple[np.ndarray]:
        """Returns the parameters of the represented distribution.

        Returns:
            One-dimensional NumPy array representing the rate parameters.
        """
        return (self.l,)

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

        Determines whether or note instances are part of the supports of the Exponential distributions, which are:

        .. math::

            \text{supp}(\text{Exponential})=[0,+\infty)

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
    layer: ExponentialLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[ExponentialLayer, Exponential, None]:
    r"""Structural marginalization for ``ExponentialLayer`` objects in the ``base`` backend.

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
        new_node = Exponential(marg_scopes[0], *marg_params[0])
        return new_node
    else:
        new_layer = ExponentialLayer(marg_scopes, *[np.array(p) for p in zip(*marg_params)])
        return new_layer

@dispatch(memoize=True)  # type: ignore
def updateBackend(leaf_node: ExponentialLayer, dispatch_ctx: Optional[DispatchContext] = None):
    """Conversion for ``SumNode`` from ``torch`` backend to ``base`` backend.

    Args:
        sum_node:
            Sum node to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return GeneralExponentialLayer(scope=leaf_node.scopes_out, l=leaf_node.l.detach().numpy())