'''
Created on March 20, 2018

@author: Alejandro Molina
'''
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from sklearn.cluster import KMeans

from src.spn.algorithms.splitting.Base import split_data_by_clusters, clusters_by_adjacency_matrix
import os
import numpy as np

path = os.path.dirname(__file__)

with open(path + "/rdc.R", "r") as pdnfile:
    code = ''.join(pdnfile.readlines())
    rdc = SignatureTranslatedAnonymousPackage(code, "testRDC")

numpy2ri.activate()


def get_RDC_transform(data, statistical_type, ohe=False, k=10, s=1 / 6):
    assert data.shape[1] == len(statistical_type), "invalid parameters"

    try:
        df = robjects.r["as.data.frame"](data)
        out = rdc.transformRDC(df, ohe, statistical_type, k, s)
        out = np.asarray(out)
    except Exception as e:
        np.savetxt("/tmp/errordata.txt", data)
        print(e)
        raise e

    return out


def get_RDC_adjacency_matrix(data, statistical_type, ohe=False, linear=True):
    assert data.shape[1] == len(statistical_type), "invalid parameters"

    try:
        df = robjects.r["as.data.frame"](data)
        out = rdc.testRDC(df, ohe, statistical_type, linear)
        out = np.asarray(out)
    except Exception as e:
        np.savetxt("/tmp/errordata.txt", data)
        print(e)
        raise e

    return out


def split_cols_RDC(local_data, ds_context, scope, threshold=0.8, ohe=False, linear=True):
    adjm = get_RDC_adjacency_matrix(local_data, ds_context.statistical_type[scope], ohe, linear)

    clusters = clusters_by_adjacency_matrix(adjm, threshold, local_data.shape[1])

    return split_data_by_clusters(local_data, clusters, scope, rows=False)


def split_rows_RDC(local_data, ds_context, scope, n_clusters=2, k=10, s=1 / 6, ohe=False, seed=17):
    data = get_RDC_transform(local_data, ds_context.statistical_type[scope], ohe, k=k, s=s)

    clusters = KMeans(n_clusters=n_clusters, random_state=seed, n_jobs=1).fit_predict(data)

    return split_data_by_clusters(local_data, clusters, scope, rows=True)



if __name__ == '__main__':
    import numpy as np

    p = path + "/../../../data/nips100.csv"

    print(p)

    nips = np.loadtxt(p, skiprows=1, delimiter=',')

    print(nips)

    ds_context = type('', (object,), {})()
    ds_context.statistical_type = np.asarray(["discrete"] * nips.shape[1])

    scope = [1, 2, 3, 5, 6, 7, 8, 9, 10]
    #a = split_rows_RDC(nips[:, scope], ds_context, scope)
    #print(len(a))

    a = split_rows_RDC(nips[20:80, :][:, scope], ds_context, scope)
    print(len(a), a)
