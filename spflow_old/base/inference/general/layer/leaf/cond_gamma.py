"""Contains inference methods for ``CondGammaLayer`` leaves for SPFlow in the ``base`` backend.
"""
from typing import Optional

import numpy as np

from spflow.base.structure.general.layer.leaf.cond_gamma import (
    CondGammaLayer,
)
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
    layer: CondGammaLayer,
    data: np.ndarray,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> np.ndarray:
    r"""Computes log-likelihoods for ``CondGammaLayer`` leaves in the ``base`` backend given input data.

    Log-likelihood for ``CondGammaLayer`` is given by the logarithm of its individual probability distribution functions (PDFs):

    .. math::

        \log(\text{PDF}(x) = \begin{cases} \log(\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}) & \text{if } x > 0\\
                                           \log(0) & \text{if } x <= 0\end{cases}

    where
        - :math:`x` is the input observation
        - :math:`\Gamma` is the Gamma function
        - :math:`\alpha` is the shape parameter
        - :math:`\beta` is the rate parameter

    Missing values (i.e., NaN) are marginalized over.

    Args:
        node:
            Leaf node to perform inference for.
        data:
            Two-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Two-dimensional NumPy array containing the log-likelihoods of the input data for the sum node.
        Each row corresponds to an input sample.

    Raises:
        ValueError: Data outside of support.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # retrieve values for 'alpha','beta'
    alpha_values, beta_values = layer.retrieve_params(data, dispatch_ctx)

    for node, alpha, beta in zip(layer.nodes, alpha_values, beta_values):
        dispatch_ctx.update_args(node, {"alpha": alpha, "beta": beta})

    # weight child log-likelihoods (sum in log-space) and compute log-sum-exp
    return np.concatenate(
        [
            log_likelihood(
                node,
                data,
                check_support=check_support,
                dispatch_ctx=dispatch_ctx,
            )
            for node in layer.nodes
        ],
        axis=1,
    )
