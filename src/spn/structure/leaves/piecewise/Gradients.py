import numpy as np


from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear


def expand(array, left, right):
    expanded = np.append(array, right)
    expanded = np.append(left, expanded)
    return expanded


def piecewise_gradient(node, input_vals=None, dtype=np.float64):
    if input_vals is None:
        raise ValueError("Input to piecewise_gradient cannot be None")
    data = input_vals
    obs = data[:, node.scope[0]]
    marg_ids = np.isnan(obs)

    x_range = expand(np.array(node.x_range), -np.infty, np.infty)
    y_range = expand(np.array(node.y_range), 0, 0)

    loc = np.searchsorted(x_range, obs)
    upper = y_range[loc]
    lower = y_range[loc - 1]

    gradient = (upper - lower) / (x_range[loc] - x_range[loc - 1])
    gradient[marg_ids] = np.nan

    return gradient.reshape((-1, 1))


def add_piecewise_linear_gradient_support():
    add_node_gradients(PiecewiseLinear, piecewise_gradient)
