"""
Created on March 20, 2018

@author: Alejandro Molina
"""

import numpy as np
from networkx import from_numpy_matrix, connected_components

from sklearn.feature_extraction.text import TfidfTransformer

from spn.structure.StatisticalTypes import MetaType


def preproc(data, ds_context, pre_proc, ohe):
    if pre_proc:
        f = None
        if pre_proc == "tf-idf":
            f = lambda data: TfidfTransformer().fit_transform(data)
        elif ds_context == "log+1":
            f = lambda data: np.log(data + 1)
        elif ds_context == "sqrt":
            f = lambda data: np.sqrt(data)

        if f is not None:
            data = np.copy(data)
            data[:, ds_context.distribution_family == "poisson"] = f(
                data[:, ds_context.distribution_family == "poisson"]
            )

    if ohe:
        data = getOHE(data, ds_context)

    return data


def getOHE(data, ds_context):
    cols = []
    for f in range(data.shape[1]):
        data_col = data[:, f]

        if ds_context.meta_types[f] != MetaType.DISCRETE:
            cols.append(data_col)
            continue

        domain = ds_context.domains[f]

        dataenc = np.zeros((data_col.shape[0], len(domain)), dtype=data.dtype)

        dataenc[data_col[:, None] == domain[None, :]] = 1

        assert np.all((np.sum(dataenc, axis=1) == 1)), "one hot encoding bug {} {}".format(domain, data_col)

        cols.append(dataenc)

    return np.column_stack(cols)


def clusters_by_adjacency_matrix(adm, threshold, n_features):
    adm[adm < threshold] = 0

    adm[adm > 0] = 1

    result = np.zeros(n_features)
    for i, c in enumerate(connected_components(from_numpy_matrix(adm))):
        result[list(c)] = i + 1

    return result


def split_data_by_clusters(data, clusters, scope, rows=True):
    unique_clusters = np.unique(clusters)
    result = []

    nscope = np.asarray(scope)

    for uc in unique_clusters:
        if rows:
            local_data = data[clusters == uc, :]
            proportion = local_data.shape[0] / data.shape[0]
            result.append((local_data, scope, proportion))
        else:
            local_data = data[:, clusters == uc].reshape((data.shape[0], -1))
            proportion = local_data.shape[1] / data.shape[1]
            result.append((local_data, nscope[clusters == uc].tolist(), proportion))

    return result


def split_conditional_data_by_clusters(data, clusters, scope, rows=True):
    print(clusters)

    unique_clusters = np.unique(clusters)
    result = []

    local_scope = np.arange(len(scope))
    nscope = np.asarray(scope)

    dataOut = data[:, local_scope]
    dataIn = data[:, len(scope) :]

    # print(np.shape(data), np.shape(dataOut), np.shape(dataIn))
    for uc in unique_clusters:
        if rows:
            raise NotImplementedError
        else:
            local_data = np.concatenate((dataOut[:, clusters == uc].reshape((data.shape[0], -1)), dataIn), axis=1)
            proportion = local_data.shape[1] / data.shape[1]
            result.append((local_data, nscope[clusters == uc].tolist(), proportion))
            # print(uc)
            # print('cplit cols', np.shape(data), np.shape(local_data), np.shape(dataOut[:, clusters == uc].reshape((data.shape[0], -1))), np.shape(dataIn))
    return result
