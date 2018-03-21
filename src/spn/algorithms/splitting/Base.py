'''
Created on March 20, 2018

@author: Alejandro Molina
'''

import numpy as np
from networkx import from_numpy_matrix, connected_components


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
            result.append((data[clusters == uc, :], scope))
        else:
            result.append((data[:, clusters == uc].reshape((data.shape[0],-1)), nscope[clusters == uc]))

    return result