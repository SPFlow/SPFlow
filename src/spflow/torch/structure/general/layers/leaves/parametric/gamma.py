"""Contains Gamma leaf layer for SPFlow in the ``torch`` backend.
"""
from typing import List, Union, Optional, Iterable, Tuple
from functools import reduce
import numpy as np
import torch
import torch.distributions as D
from torch.nn.parameter import Parameter

from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from spflow.meta.data.scope import Scope
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.torch.utils.projections import (
    proj_bounded_to_real,
    proj_real_to_bounded,
)
from spflow.torch.structure.module import Module
from spflow.torch.structure.general.nodes.leaves.parametric.gamma import Gamma
from spflow.base.structure.general.layers.leaves.parametric.gamma import (
    GammaLayer as BaseGammaLayer,
)


class GammaLayer(Module):
    r"""Layer of multiple (univariate) Gamma distribution leaf nodes in the 'base' backend.

    Represents multiple univariate Gamma distributions with independent scopes, each with the following probability distribution function (PDF):

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
            Unbounded one-dimensional PyTorch parameters that are projected to yield the actual shape parameters.
        alpha:
            One-dimensional PyTorch tensor representing the shape parameters (:math:`\alpha`) of the Gamma distributions, greater than 0 (projected from ``alpha_aux``).
        beta_aux:
            Unbounded one-dimensional PyTorch parameters that are projected to yield the actual rate parameters.
        beta:
            One-dimensional PyTorch tensor representing the rate parameters (:math:`\beta`) of the Gamma distributions, greater than 0 (projected from ``beta_aux``).
    """

    def __init__(
        self,
        scope: Union[Scope, List[Scope]],
        alpha: Union[int, float, List[float], np.ndarray, torch.Tensor] = 1.0,
        beta: Union[int, float, List[float], np.ndarray, torch.Tensor] = 1.0,
        n_nodes: int = 1,
        **kwargs,
    ) -> None:
        r"""Initializes ``GammaLayer`` object.

        Args:
            scope:
                Scope or list of scopes specifying the scopes of the individual distribution.
                If a single scope is given, it is used for all nodes.
            alpha:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor representing the shape parameters (:math:`\alpha`), greater than 0.
                If a single value is given it is broadcast to all nodes.
                Defaults to 1.0.
            beta:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor representing the rate parameters (:math:`\beta`), greater than 0.
                If a single value is given it is broadcast to all nodes.
                Defaults to 1.0.
            n_nodes:
                Integer specifying the number of nodes the layer should represent. Only relevant if a single scope is given.
                Defaults to 1.
        """
        if isinstance(scope, Scope):
            if n_nodes < 1:
                raise ValueError(
                    f"Number of nodes for 'GammaLayer' must be greater or equal to 1, but was {n_nodes}"
                )

            scope = [scope for _ in range(n_nodes)]
            self._n_out = n_nodes
        else:
            if len(scope) == 0:
                raise ValueError("List of scopes for 'GammaLayer' was empty.")

            self._n_out = len(scope)

        for s in scope:
            if len(s.query) != 1:
                raise ValueError("Size of query scope must be 1 for all nodes.")
            if len(s.evidence) != 0:
                raise ValueError(
                    f"Evidence scope for 'GammaLayer' should be empty, but was {s.evidence}."
                )

        super().__init__(children=[], **kwargs)

        # register auxiliary torch parameter for rate l of each implicit node
        self.alpha_aux = Parameter()
        self.beta_aux = Parameter()

        # compute scope
        self.scopes_out = scope
        self.combined_scope = reduce(
            lambda s1, s2: s1.join(s2), self.scopes_out
        )

        # parse weights
        self.set_params(alpha, beta)

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        """Checks if a specified signature can be represented by the module.

        ``GammaLayer`` can represent one or more univariate nodes with ``MetaType.Continuous`` or ``GammaType`` domains.

        Returns:
            Boolean indicating whether the module can represent the specified signature (True) or not (False).
        """
        # leaf has at least one output
        if len(signatures) < 1:
            return False

        for signature in signatures:
            if not Gamma.accepts([signature]):
                return False

        return True

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]) -> "GammaLayer":
        """Creates an instance from a specified signature.

        Returns:
            ``GammaLayer`` instance.

        Raises:
            Signatures not accepted by the module.
        """
        if not cls.accepts(signatures):
            raise ValueError(
                f"'GammaLayer' cannot be instantiated from the following signatures: {signatures}."
            )

        alpha = []
        beta = []
        scopes = []

        for feature_ctx in signatures:

            domain = feature_ctx.get_domains()[0]

            # read or initialize parameters
            if domain == MetaType.Continuous:
                alpha.append(1.0)
                beta.append(1.0)
            elif domain == FeatureTypes.Gamma:
                # instantiate object
                alpha.append(domain().alpha)
                beta.append(domain().beta)
            elif isinstance(domain, FeatureTypes.Gamma):
                alpha.append(domain.alpha)
                beta.append(domain.beta)
            else:
                raise ValueError(
                    f"Unknown signature type {domain} for 'GammaLayer' that was not caught during acception checking."
                )

            scopes.append(feature_ctx.scope)

        return GammaLayer(scopes, alpha=alpha, beta=beta)

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Equal to the number of nodes represented by the layer."""
        return self._n_out

    @property
    def alpha(self) -> torch.Tensor:
        """Returns the shape parameters of the represented distributions."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.alpha_aux, lb=0.0)  # type: ignore

    @property
    def beta(self) -> torch.Tensor:
        """Returns the rate parameters of the represented distributions."""
        # project auxiliary parameter onto actual parameter range
        return proj_real_to_bounded(self.beta_aux, lb=0.0)  # type: ignore

    def dist(self, node_ids: Optional[List[int]] = None) -> D.Distribution:
        r"""Returns the PyTorch distributions represented by the leaf layer.

        Args:
            node_ids:
                Optional list of integers specifying the indices (and order) of the nodes' distribution to return.
                Defaults to None, in which case all nodes distributions selected.

        Returns:
            ``torch.distributions.Gamma`` instance.
        """
        if node_ids is None:
            node_ids = list(range(self.n_out))

        return D.Gamma(
            concentration=self.alpha[node_ids], rate=self.beta[node_ids]
        )

    def set_params(
        self,
        alpha: Union[int, float, List[float], np.ndarray, torch.Tensor],
        beta: Union[int, float, List[float], np.ndarray, torch.Tensor],
    ) -> None:
        r"""Sets the parameters for the represented distributions.

        TODO: projection function

        Args:
            alpha:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor representing the shape parameters (:math:`\alpha`), greater than 0.
            beta:
                Floating point, list of floats or one-dimensional NumPy array or PyTorch tensor representing the rate parameters (:math:`\beta`), greater than 0.
        """
        if isinstance(alpha, int) or isinstance(alpha, float):
            alpha = torch.tensor([alpha for _ in range(self.n_out)])
        elif isinstance(alpha, list) or isinstance(alpha, np.ndarray):
            alpha = torch.tensor(alpha)
        if alpha.ndim != 1:
            raise ValueError(
                f"Numpy array of 'alpha' values for 'GammaLayer' is expected to be one-dimensional, but is {alpha.ndim}-dimensional."
            )
        if alpha.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'alpha' values for 'GammaLayer' must match number of output nodes {self.n_out}, but is {alpha.shape[0]}"
            )

        if torch.any(alpha <= 0.0) or not torch.any(torch.isfinite(alpha)):
            raise ValueError(
                f"Values of 'alpha' for 'GammaLayer' must be greater than 0, but was: {alpha}"
            )

        if isinstance(beta, int) or isinstance(beta, float):
            beta = torch.tensor([beta for _ in range(self.n_out)])
        elif isinstance(beta, list) or isinstance(beta, np.ndarray):
            beta = torch.tensor(beta)
        if beta.ndim != 1:
            raise ValueError(
                f"Numpy array of 'beta' values for 'GammaLayer' is expected to be one-dimensional, but is {beta.ndim}-dimensional."
            )
        if beta.shape[0] != self.n_out:
            raise ValueError(
                f"Length of numpy array of 'beta' values for 'GammaLayer' must match number of output nodes {self.n_out}, but is {beta.shape[0]}"
            )

        if torch.any(beta <= 0.0) or not torch.any(torch.isfinite(beta)):
            raise ValueError(
                f"Value of 'beta' for 'GammaLayer' must be greater than 0, but was: {beta}"
            )

        self.alpha_aux.data = proj_bounded_to_real(alpha, lb=0.0)
        self.beta_aux.data = proj_bounded_to_real(beta, lb=0.0)

    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the parameters of the represented distribution.

        Returns:
            Tuple of two one-dimensional PyTorch tensors representing the shape and rate parameters, respectively.
        """
        return (self.alpha, self.beta)

    def check_support(
        self,
        data: torch.Tensor,
        node_ids: Optional[List[int]] = None,
        is_scope_data: bool = False,
    ) -> torch.Tensor:
        r"""Checks if specified data is in support of the represented distributions.

        Determines whether or note instances are part of the supports of the Gamma distributions, which are:

        .. math::

            \text{supp}(\text{Gamma})=(0,+\infty)

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
            scope_data = data[
                :, [self.scopes_out[node_id].query[0] for node_id in node_ids]
            ]

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
    layer: GammaLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Union[GammaLayer, Gamma, None]:
    """Structural marginalization for ``GammaLayer`` objects in the ``torch`` backend.

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
        return Gamma(
            scope=marginalized_scopes[0],
            alpha=layer.alpha[node_id].item(),
            beta=layer.beta[node_id].item(),
        )
    else:
        return GammaLayer(
            scope=marginalized_scopes,
            alpha=layer.alpha[marginalized_node_ids].detach(),
            beta=layer.beta[marginalized_node_ids].detach(),
        )


@dispatch(memoize=True)  # type: ignore
def toTorch(
    layer: BaseGammaLayer, dispatch_ctx: Optional[DispatchContext] = None
) -> GammaLayer:
    """Conversion for ``GammaLayer`` from ``base`` backend to ``torch`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return GammaLayer(
        scope=layer.scopes_out, alpha=layer.alpha, beta=layer.beta
    )


@dispatch(memoize=True)  # type: ignore
def toBase(
    layer: GammaLayer, dispatch_ctx: Optional[DispatchContext] = None
) -> BaseGammaLayer:
    """Conversion for ``GammaLayer`` from ``torch`` backend to ``base`` backend.

    Args:
        layer:
            Leaf to be converted.
        dispatch_ctx:
            Dispatch context.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    return BaseGammaLayer(
        scope=layer.scopes_out,
        alpha=layer.alpha.detach().numpy(),
        beta=layer.beta.detach().numpy(),
    )
