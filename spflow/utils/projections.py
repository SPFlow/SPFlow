"""Projections between unbounded and bounded intervals.

Used for internal projections of PyTorch parameters.
"""
from typing import Optional, Union

import torch
from torch import Tensor


def proj_convex_to_real(x: Tensor) -> Tensor:
    r"""Projects normalized convex coefficients (e.g., weights) to real numbers.

    This projection is used when learning weight parameters that must remain
    in the probability simplex (sum to 1, all non-negative). It maps from the
    simplex to the unconstrained real line using the log function.

    .. math::

        f(x) = \ln(x)

    where :math:`x` is a tensor with non-negative values that sum to 1.

    Args:
        x: PyTorch tensor with values in [0, 1] representing normalized weights
           or probabilities (e.g., output of softmax).

    Returns:
        PyTorch tensor of same shape as ``x`` with the log-transformed values.
        These values can be unconstrained real numbers used for gradient-based
        optimization.

    Note:
        This is the inverse of proj_real_to_convex. The two functions form a
        bijection between the probability simplex and the real line (excluding 0).
    """
    # convex coefficients are already normalized, so taking the log is sufficient
    return torch.log(x)


def proj_real_to_convex(x: Tensor) -> Tensor:
    r"""Projects real numbers onto the probability simplex using softmax.

    This projection is used when parameterizing weight variables that must remain
    normalized (sum to 1, all non-negative). It maps from the unconstrained real line
    to the probability simplex using the softmax function.

    .. math::

        f(x) = \text{softmax}(x) = \frac{e^{x_i}}{\sum_j e^{x_j}}

    This ensures all output values are in [0, 1] and sum to 1 across the last dimension.

    Args:
        x: PyTorch tensor representing unconstrained real numbers.

    Returns:
        PyTorch tensor of same shape as ``x`` with normalized values (summing to 1
        along the last dimension). All values are in [0, 1].

    Note:
        This is the inverse of proj_convex_to_real. The two functions form a
        bijection between the probability simplex and the real line (excluding 0).
    """
    return torch.softmax(x, dim=-1)


def proj_real_to_bounded(
    x: Tensor,
    lb: Optional[Union[float, Tensor]] = None,
    ub: Optional[Union[float, Tensor]] = None,
) -> Tensor:
    r"""Projects real numbers onto a bounded interval.

    .. math::

        f(x)=\begin{cases} \sigma(x)(u-l)+l & \text{if domain bounded by } l,u \\
                           e^x+l            & \text{if domain lower bounded by } l \\
                           -e^x+u           & \text{if domain upper bounded by } u \end{cases}
    
    where
        - :math:`x`: bounded input
        - :math:`l`: (possible) lower bound of input domain
        - :math:`u`: (possible) upper bound of input domain
    
    Args:
        x:
            PyTorch tensor representing unbounded inputs.
        lb:
            Float or scalar PyTorch Tensor defining the lower bound (:math:`l`) of the codomain/projection or None (default), in which case the projection is not lower bounded.
        ub:
            Float or scalar PyTorch Tensor defining the lower bound (:math:`u`) of the codomain/projection or None (default), in which case the projection is not upper bounded.
    
    Returns:
        PyTorch tensor of same shape as ``x`` containing the projected values.
    """
    if lb is not None and ub is not None:
        # project to bounded interval
        return torch.sigmoid(x) * (ub - lb) + lb
    elif ub is None:
        # project to left-bounded interval
        return torch.exp(x) + lb
    else:
        # project to right-bounded interval
        return -torch.exp(x) + ub


def proj_bounded_to_real(
    x: Tensor,
    lb: Optional[Union[float, Tensor]] = None,
    ub: Optional[Union[float, Tensor]] = None,
) -> Tensor:
    r"""Projects a bounded interval onto the real numbers.

    .. math::

        f(x)=\begin{cases} \ln(\frac{x-l}{u-x}) & \text{if codomain bounded by } l,u \\
                           \ln(x-l)             & \text{if codomain lower bounded by } l \\
                           \ln(u-x)             & \text{if codomain upper bounded by } u \end{cases}
    
    where
        - :math:`x`: unbounded input
        - :math:`l`: (possible) lower bound of projection
        - :math:`u`: (possible) upper bound of projection
    
    Args:
        x:
            PyTorch tensor representing bounded inputs.
        lb:
            Float or scalar PyTorch Tensor defining the lower bound (:math:`l`) of the domain or None (default), in which case the input domain is not lower bounded.
        ub:
            Float or scalar PyTorch Tensor defining the lower bound (:math:`u`) of the domain or None (default), in which case the input domain is not upper bounded.
    
    Returns:
        PyTorch tensor of same shape as ``x`` containing the projected values.
    """
    if lb is not None and ub is not None:
        # project from bounded interval
        return torch.log((x - lb) / (ub - x))
    elif ub is None:
        # project from left-bounded interval
        return torch.log(x - lb)
    else:
        # project from right-bounded interval
        return torch.log(ub - x)
