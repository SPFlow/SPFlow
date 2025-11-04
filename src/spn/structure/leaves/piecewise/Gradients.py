import numpy as np


from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear
from spn.algorithms.Gradient import add_node_feature_gradient


def expand(array, left, right):
    expanded = np.append(array, right)
    expanded = np.append(left, expanded)
    return expanded


def piecewise_gradient(node, input_vals=None, dtype=np.float64):
    if input_vals is None:
        raise ValueError("Input to piecewise_gradient cannot be None")
    data = input_vals
    obs = data[:, node.scope[0]]
    gradient = np.full(input_vals.shape, np.nan)

    x_range = expand(np.array(node.x_range), -np.inf, np.inf)
    y_range = expand(np.array(node.y_range), 0, 0)

    loc = np.searchsorted(x_range, obs)
    upper = y_range[loc]
    lower = y_range[loc - 1]

    gradient[:, node.scope] = ((upper - lower) / (x_range[loc] - x_range[loc - 1])).reshape(-1, 1)

    return gradient


def add_piecewise_linear_gradient_support():
    add_node_feature_gradient(PiecewiseLinear, piecewise_gradient)
