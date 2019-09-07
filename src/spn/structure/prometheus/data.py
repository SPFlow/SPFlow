# Some helper functions to make the entire thing run
import spn.structure.prometheus.nodes
import math
import numpy as np
import scipy
import scipy.cluster.hierarchy as hcluster
from scipy.cluster.vq import vq, kmeans, whiten

# Converts array indices from function-level indices to global-level objective indices. For example, consider the set [0,1,2,3] and suppose you have passed the [1,3]'s into a call. They will be treated as [0,1] and you will convert them back.
# The usage of the set functionality slows up the implementation a bit and
# is not strictly necessary. However, it's a good idea to keep them this
# way since this simplifies the code.


def eff(tempdat, scope):
    effdat = np.copy(tempdat[:, sorted(list(scope))])
    return effdat


def returnarr(arr, scope):
    q = []
    te = sorted(list(scope))
    for i in arr:
        q.append(te[i])
    return set(q)


def split(arr, k, scope):
    pholder, clusters = scipy.cluster.vq.kmeans2(
        arr[:, sorted(list(scope))], k, minit='points')
    big = []
    for i in range(0, len(set(clusters))):
        mask = np.where(clusters == i)
        print(np.shape(mask))
        mask = np.asarray(mask, dtype=np.int32)
        mask = mask.flatten()
        small = (arr[mask, :])
        print(np.shape(small))
        big.append(small)
        print(big)
    return big


def submat(mat, subset):
    ret = np.copy(mat[:, sorted(list(subset))])
    return ret[sorted(list(subset)), :]


def submean(mean, subset):
    m = np.copy(mean[sorted(list(subset))])
    return m
