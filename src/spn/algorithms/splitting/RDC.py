'''
Created on March 20, 2018

@author: Alejandro Molina
'''

import numpy as np
from sklearn.cluster import KMeans

from spn.algorithms.splitting.Base import split_data_by_clusters, clusters_by_adjacency_matrix

_rpy_initialized = False


def init_rpy():
    global _rpy_initialized
    if _rpy_initialized:
        return
    _rpy_initialized = True

    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    import os
    path = os.path.dirname(__file__)

    robjects.r("options(warn=-1)")

    with open(path + "/rdc.R", "r") as rfile:
        code = ''.join(rfile.readlines())
        robjects.r(code)

    numpy2ri.activate()


def get_RDC_transform(data, meta_types, ohe=False, k=10, s=1 / 6):
    from rpy2 import robjects
    init_rpy()

    assert data.shape[1] == len(meta_types), "invalid parameters"

    r_meta_types = [mt.name.lower() for mt in meta_types]

    try:
        df = robjects.r["as.data.frame"](data)
        out = robjects.r["transformRDC"](df, ohe, r_meta_types, k, s)
        out = np.asarray(out)
    except Exception as e:
        np.savetxt("/tmp/errordata.txt", data)
        print(e)
        raise e

    return out


def get_RDC_adjacency_matrix(data, meta_types, ohe=False, linear=True):
    from rpy2 import robjects
    init_rpy()

    assert data.shape[1] == len(meta_types), "invalid parameters"

    r_meta_types = [mt.name.lower() for mt in meta_types]

    try:
        df = robjects.r["as.data.frame"](data)
        out = robjects.r["testRDC"](df, ohe, r_meta_types, linear)
        out = np.asarray(out)
    except Exception as e:
        np.savetxt("/tmp/errordata.txt", data)
        print(e)
        raise e

    return out


def get_split_cols_RDC(threshold=0.3, ohe=True, linear=True):
    def split_cols_RDC(local_data, ds_context, scope):
        adjm = get_RDC_adjacency_matrix(local_data, ds_context.get_meta_types_by_scope(scope), ohe, linear)

        clusters = clusters_by_adjacency_matrix(adjm, threshold, local_data.shape[1])

        return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_RDC


def get_split_rows_RDC(n_clusters=2, k=10, s=1 / 6, ohe=True, seed=17):
    def split_rows_RDC(local_data, ds_context, scope):
        data = get_RDC_transform(
            local_data, ds_context.get_meta_types_by_scope(scope), ohe, k=k, s=s)

        clusters = KMeans(n_clusters=n_clusters, random_state=seed, n_jobs=1).fit_predict(data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_RDC


############################################################################################
#
# Python version
#
import itertools

from networkx.algorithms.components.connected import connected_components
from networkx.convert_matrix import from_numpy_matrix
import scipy.stats

from sklearn.cross_decomposition import CCA
from spn.structure.StatisticalTypes import MetaType

CCA_MAX_ITER = 100


def ecdf(X):
    """
    Empirical cumulative distribution function
    for data X (one dimensional, if not it is linearized first)
    """
    # return scipy.stats.rankdata(X, method='max') / len(X)

    mv_ids = np.isnan(X)

    N = X.shape[0]
    X = X[~mv_ids]
    R = scipy.stats.rankdata(X, method='max') / len(X)
    X_r = np.zeros(N)
    X_r[~mv_ids] = R
    return X_r


def empirical_copula_transformation(data):
    ones_column = np.ones((data.shape[0], 1))
    data = np.concatenate((np.apply_along_axis(ecdf, 0, data), ones_column), axis=1)
    return data


def make_matrix(data):
    """
    Ensures data to be 2-dimensional
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]
    else:
        assert data.ndim == 2, "Data must be 2 dimensional {}".format(data.shape)

    return data


def ohe_data(data, domain):
    dataenc = np.zeros((data.shape[0], len(domain)))

    dataenc[data[:, None] == domain[None, :]] = 1

    #
    # this control fails when having missing data as nans
    if not np.any(np.isnan(data)):
        assert np.all((np.nansum(dataenc, axis=1) == 1)
                      ), "one hot encoding bug {} {} {}".format(domain, data,
                                                                np.nansum(dataenc, axis=1))

    return dataenc


def rdc_transformer(local_data,
                    meta_types,
                    domains,
                    k=None,
                    s=1. / 6.,
                    non_linearity=np.sin,
                    return_matrix=False,
                    ohe=True,
                    rand_gen=None):
    # print('rdc transformer', k, s, non_linearity)
    """
    Given a data_slice,
    return a transformation of the features data in it according to the rdc
    pipeline:
    1 - empirical copula transformation
    2 - random projection into a k-dimensional gaussian space
    3 - pointwise  non-linear transform
    """

    N, D = local_data.shape

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    #
    # precomputing transformations to reduce time complexity
    #

    #
    # FORCING ohe on all discrete features
    features = []
    for f in range(D):
        if meta_types[f] == MetaType.DISCRETE:
            features.append(ohe_data(local_data[:, f], domains[f]))
        else:
            features.append(local_data[:, f].reshape(-1, 1))
    # else:
    #     features = [data_slice.getFeatureData(f) for f in range(D)]

    #
    # NOTE: here we are setting a global k for ALL features
    # to be able to precompute gaussians
    if k is None:
        feature_shapes = [f.shape[1] if len(f.shape) > 1 else 1 for f in features]
        k = max(feature_shapes) + 1

    #
    # forcing two columness
    features = [make_matrix(f) for f in features]

    #
    # transform through the empirical copula
    features = [empirical_copula_transformation(f) for f in features]

    #
    # substituting nans with zero (the above step should have taken care of that)
    features = [np.nan_to_num(f) for f in features]

    #
    # random projection through a gaussian
    random_gaussians = [rand_gen.normal(size=(f.shape[1], k))
                        for f in features]

    rand_proj_features = [s / f.shape[1] * np.dot(f, N)
                          for f, N in zip(features,
                                          random_gaussians)]

    nl_rand_proj_features = [non_linearity(f)
                             for f in rand_proj_features]

    #
    # apply non-linearity
    if return_matrix:
        return np.concatenate(nl_rand_proj_features, axis=1)

    else:
        return [np.concatenate((f, np.ones((f.shape[0], 1))), axis=1)
                for f in nl_rand_proj_features]


def rdc_cca(indexes):
    i, j, rdc_features = indexes
    cca = CCA(n_components=1, max_iter=CCA_MAX_ITER)
    X_cca, Y_cca = cca.fit_transform(rdc_features[i], rdc_features[j])
    rdc = np.corrcoef(X_cca.T, Y_cca.T)[0, 1]
    #print(i, j, rdc)
    return rdc


def rdc_test(local_data,
             meta_types,
             domains,
             k=None,
             s=1. / 6.,
             non_linearity=np.sin,
             n_jobs=-1,
             rand_gen=None):
    n_features = local_data.shape[1]

    rdc_features = rdc_transformer(local_data,
                                   meta_types,
                                   domains,
                                   k=k,
                                   s=s,
                                   non_linearity=non_linearity,
                                   return_matrix=False,
                                   rand_gen=rand_gen)

    pairwise_comparisons = list(itertools.combinations(np.arange(n_features), 2))

    from joblib import Parallel, delayed
    rdc_vals = Parallel(n_jobs=n_jobs, max_nbytes=1024, backend="threading")(delayed(rdc_cca)((i, j, rdc_features))
                                                                for i, j in pairwise_comparisons)

    rdc_adjacency_matrix = np.zeros((n_features, n_features))

    for (i, j), rdc in zip(pairwise_comparisons, rdc_vals):
        rdc_adjacency_matrix[i, j] = rdc
        rdc_adjacency_matrix[j, i] = rdc

    #
    # setting diagonal to 1
    rdc_adjacency_matrix[np.diag_indices_from(rdc_adjacency_matrix)] = 1

    return rdc_adjacency_matrix


def getIndependentRDCGroups_py(local_data,
                               threshold,
                               meta_types,
                               domains,
                               k=None,
                               s=1. / 6.,
                               non_linearity=np.sin,
                               n_jobs=-2,
                               rand_gen=None):
    rdc_adjacency_matrix = rdc_test(local_data,
                                    meta_types,
                                    domains,
                                    k=k,
                                    s=s,
                                    non_linearity=non_linearity,
                                    n_jobs=n_jobs,
                                    rand_gen=rand_gen)

    #
    # Why is this necessary?
    #
    rdc_adjacency_matrix[np.isnan(rdc_adjacency_matrix)] = 0
    n_features = local_data.shape[1]

    #
    # thresholding
    rdc_adjacency_matrix[rdc_adjacency_matrix < threshold] = 0
    # print("thresholding", rdc_adjacency_matrix)

    #
    # getting connected components
    result = np.zeros(n_features)
    for i, c in enumerate(connected_components(from_numpy_matrix(rdc_adjacency_matrix))):
        result[list(c)] = i + 1

    return result


def get_split_cols_RDC_py(threshold=0.3, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2,
                          rand_gen=None):
    def split_cols_RDC_py(local_data, ds_context, scope):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        clusters = getIndependentRDCGroups_py(local_data,
                                              threshold,
                                              meta_types,
                                              domains,
                                              k=k,
                                              s=s,
                                              # ohe=True,
                                              non_linearity=non_linearity,
                                              n_jobs=n_jobs,
                                              rand_gen=rand_gen)

        return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_RDC_py


def get_split_rows_RDC_py(n_clusters=2, ohe=True, k=10, s=1 / 6,
                          non_linearity=np.sin, n_jobs=-2,
                          rand_gen=None):
    def split_rows_RDC_py(local_data, ds_context, scope):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        rdc_data = rdc_transformer(local_data,
                                   meta_types,
                                   domains,
                                   k=k,
                                   s=s,
                                   non_linearity=non_linearity,
                                   return_matrix=True,
                                   rand_gen=rand_gen)

        clusters = KMeans(n_clusters=n_clusters,
                          random_state=rand_gen, n_jobs=n_jobs).fit_predict(rdc_data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_RDC_py
