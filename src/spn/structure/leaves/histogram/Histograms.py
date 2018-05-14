'''
Created on March 20, 2018

@author: Alejandro Molina
'''
import numpy as np

from spn.structure.Base import Leaf
from spn.structure.StatisticalTypes import MetaType

rpy_initialized = False


def init_rpy():
    if rpy_initialized:
        return

    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    import os
    path = os.path.dirname(__file__)
    with open(path + "/Histogram.R", "r") as rfile:
        code = ''.join(rfile.readlines())
        robjects.r(code)

    numpy2ri.activate()


class Histogram(Leaf):
    def __init__(self, breaks, densities, bin_repr_points, scope=None):
        Leaf.__init__(self, scope=scope)
        self.breaks = breaks
        self.densities = densities
        self.bin_repr_points = bin_repr_points

    @property
    def mode(self):
        areas = np.diff(self.breaks) * self.densities
        _x = np.argmax(areas)
        return self.bin_repr_points[_x]


def create_histogram_leaf(data, ds_context, scope, alpha=1.0):
    assert len(scope) == 1, "scope of univariate histogram for more than one variable?"
    assert data.shape[1] == 1, "data has more than one feature?"

    data = data[~np.isnan(data)]

    idx = scope[0]
    meta_type = ds_context.meta_types[idx]
    domain = ds_context.domains[idx]

    if data.shape[0] == 0 or np.var(data) <= 1e-10:
        # no data or all were nans
        maxx = np.max(domain)
        minx = np.min(domain)
        breaks = np.array([minx, maxx])
        densities = np.array([1 / (maxx - minx)])
        repr_points = np.array([minx + (maxx - minx) / 2])
        if meta_type == MetaType.DISCRETE:
            repr_points = repr_points.astype(int)

    else:
        if meta_type == MetaType.REAL:
            breaks, densities, repr_points = getHistogramVals(data)

        elif meta_type == MetaType.DISCRETE:
            breaks = np.array([d for d in domain] + [domain[-1] + 1])
            densities, breaks = np.histogram(data, bins=breaks, density=True)
            repr_points = domain

        else:
            raise Exception('Invalid statistical type: ' + meta_type)

    # laplace smoothing
    if alpha:
        n_samples = data.shape[0]
        n_bins = len(breaks) - 1
        counts = densities * n_samples
        densities = (counts + alpha) / (n_samples + n_bins * alpha)

    assert (len(densities) == len(breaks) - 1)

    return Histogram(breaks.tolist(), densities.tolist(), repr_points.tolist(), scope=idx)


def getHistogramVals(data):
    from rpy2 import robjects
    init_rpy()

    result = robjects.r["getHistogram"](data)
    breaks = np.asarray(result[0])
    densities = np.asarray(result[2])
    mids = np.asarray(result[3])

    return breaks, densities, mids


def add_domains(data, ds_context):
    domain = []

    for col in range(data.shape[1]):
        feature_meta_type = ds_context.meta_types[col]
        if feature_meta_type == MetaType.REAL:
            domain.append([np.min(data[:, col]), np.max(data[:, col])])
        elif feature_meta_type == MetaType.DISCRETE:
            domain.append(np.unique(data[:, col]))

    ds_context.domains = np.asanyarray(domain)
