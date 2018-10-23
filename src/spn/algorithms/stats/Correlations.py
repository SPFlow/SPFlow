import numpy as np
from functools import reduce

from spn.algorithms.stats.Expectations import Expectation, _node_expectation, get_means, get_variances
from spn.algorithms.Inference import likelihood
from spn.algorithms.Marginalization import marginalize
from spn.algorithms.Condition import condition
from spn.structure.Base import Leaf, get_nodes_by_type, Sum, Product, eval_spn_bottom_up


def node_correlation(node, full_scope=0, dtype=np.float64):
    func = _node_expectation[type(node)]
    idx = node.scope[0]
    mat = np.zeros((full_scope, full_scope))
    mat[idx, idx] = func(node)
    return mat


def prod_correlation(node, children, full_scope=0, dtype=np.float64):
    assert node.scope is not None, 'Scope not set'
    means = np.array(children)

    def comb(x, y):
        d_x = np.diagonal(x)
        d_x = d_x.reshape((1, d_x.size))
        d_y = np.diagonal(y)
        d_y = d_y.reshape((1, d_y.size))
        return x + y + d_x.T.dot(d_y) + d_y.T.dot(d_x)
    means = reduce(comb, means, np.zeros(means[0].shape))
    return means


def sum_correlation(node, children, full_scope=0, dtype=np.float64):
    return np.sum(np.array([c * weight
                            for c, weight in zip(children, node.weights)]),
                  axis=0)


def get_full_correlation(spn, context):
    categoricals = context.get_categoricals()
    full_corr = get_correlation_matrix(spn)
    for cat in categoricals:
        full_corr[:, cat] = np.nan
        full_corr[cat, :] = np.nan
    cat_corr = get_categorical_correlation(spn, context)
    cat_cat_corr = get_mutual_information_correlation(spn, context)
    result = np.nansum([full_corr, cat_corr, cat_cat_corr], axis = 0)
    return result


def get_correlation_matrix(spn):
    size = len(spn.scope)
    covariance = get_covariance_matrix(spn)
    sigmas = np.sqrt(get_variances(spn)).reshape(1, size)
    sigma_matrix = sigmas.T.dot(sigmas)
    correlations = covariance / sigma_matrix

    return correlations


def get_covariance_matrix(spn):
    size = len(spn.scope)
    j_means = joined_means(spn)
    means = j_means.diagonal().reshape(1, size)
    squared_means = means.T.dot(means)
    covariance = j_means - squared_means
    diagonal_correlations = get_variances(spn).reshape(1, size)
    idx = np.diag_indices_from(covariance)
    covariance[idx] = diagonal_correlations

    return covariance


def get_categorical_correlation(spn, context):
    categoricals = context.get_categoricals()
    num_features = len(spn.scope)
    var = get_variances(spn)
    # all_vars = []
    full_matrix = np.zeros((num_features, num_features))
    full_matrix[:] = np.nan

    # OLD CODE
    for cat in categoricals:
        all_probs = []
        cat_vars = []
        query = np.full((1, num_features), np.nan)
        domain = context.get_domains_by_scope([cat])[0]
        for value in domain:
            query[:, cat] = value
            cond_spn = condition(spn, query)
            prob = likelihood(spn, query)
            cond_var = get_variances(cond_spn)
            cat_vars.append(cond_var)
            all_probs.append(prob)
        cat_vars = np.array(cat_vars)
        cat_vars = np.insert(cat_vars, cat, values=np.nan, axis=2)
        cat_vars = cat_vars.reshape((cat_vars.shape[0], cat_vars.shape[2]))
        all_probs = np.array(all_probs).reshape(-1, 1)
        all_probs /= np.sum(all_probs)
        total_var = np.sum(cat_vars * all_probs, axis=0)
        result = 1 - (total_var/var)
        full_matrix[:, cat] = result
        full_matrix[cat, :] = result
        for cat2 in categoricals:
            if cat != cat2:
                full_matrix[cat, cat2] = np.nan
                full_matrix[cat2, cat] = np.nan
            else:
                full_matrix[cat, cat] = 1
    assert np.all(np.logical_or(full_matrix > -0.0001, np.isnan(full_matrix)))
    full_matrix[full_matrix < 0] = 0
    return np.sqrt(full_matrix)


def get_mutual_information_correlation(spn, context):
    categoricals = context.get_categoricals()
    num_features = len(spn.scope)

    correlation_matrix = []

    for x in range(num_features):
        if x not in categoricals:
            correlation_matrix.append([np.nan] * num_features)
        else:
            x_correlation = [np.nan] * num_features
            x_range = context.get_domains_by_scope([x])[0]
            spn_x = marginalize(spn, [x])
            query_x = np.array([[np.nan] * num_features] * len(x_range))
            query_x[:, x] = x_range
            for y in categoricals:
                if x == y:
                    x_correlation[x] = 1
                    continue
                #TODO: still need correct code for setting proper array indices
                spn_y = marginalize(spn, [y])
                spn_xy = marginalize(spn, [x, y])
                y_range = context.get_domains_by_scope([y])[0]
                query_y = np.array([[np.nan] * num_features] * len(y_range))
                query_y[:, y] = y_range
                query_xy = np.array([[np.nan] * num_features] * (
                            len(x_range + 1) * (len(y_range + 1))))
                xy = np.mgrid[x_range[0]:x_range[-1]:len(x_range) * 1j,
                              y_range[0]:y_range[-1]:len(y_range) * 1j]
                xy = xy.reshape(2, -1)
                query_xy[:, x] = xy[0, :]
                query_xy[:, y] = xy[1, :]
                results_xy = likelihood(spn_xy, query_xy)
                results_xy = results_xy.reshape(len(x_range), len(y_range))
                results_x = likelihood(spn_x, query_x)
                results_y = likelihood(spn_y, query_y)

                xx, yy = np.mgrid[0:len(x_range) - 1:len(x_range) * 1j,
                         0:len(y_range) - 1:len(y_range) * 1j]
                xx = xx.astype(int)
                yy = yy.astype(int)

                grid_results_x = results_x[xx]
                grid_results_y = results_y[yy]
                grid_results_xy = results_xy

                log = np.log(grid_results_xy / (
                    np.multiply(grid_results_x, grid_results_y)))
                prod = np.prod(np.array([log, grid_results_xy]), axis=0)

                log_x = np.log(results_x)
                log_y = np.log(results_y)

                entropy_x = -1 * np.sum(np.multiply(log_x, results_x))
                entropy_y = -1 * np.sum(np.multiply(log_y, results_y))

                x_correlation.append(np.sum(prod) / np.sqrt(entropy_x * entropy_y))
            correlation_matrix.append(x_correlation)

    return np.array(correlation_matrix)


def joined_means(spn):
    """Compute the joined mean:

        E[XY]

    TODO: Currently, only unconditional correlatios are implemented

    Keyword arguments:
    spn -- the spn to compute the probabilities from
    """

    node_functions = {type(leaf): node_correlation
                      for leaf in get_nodes_by_type(spn, Leaf)}
    node_functions.update({Sum: sum_correlation,
                           Product: prod_correlation})

    expectation = eval_spn_bottom_up(spn, node_functions, full_scope=len(spn.scope))
    return expectation
