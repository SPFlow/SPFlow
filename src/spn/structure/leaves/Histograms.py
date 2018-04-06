'''
Created on March 20, 2018

@author: Alejandro Molina
'''
import numpy as np

from spn.structure.Base import Leaf

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
    def __init__(self, breaks, densities):
        Leaf.__init__(self)
        self.breaks = breaks
        self.densities = densities


def create_histogram_leaf(data, ds_context, scope, alpha=1.0):
    assert len(scope) == 1, "scope of univariate histogram for more than one variable?"
    assert data.shape[1] == 1, "data has more than one feature?"

    idx = scope[0]
    statistical_type = ds_context.statistical_type[idx]
    domain = ds_context.domain[idx]

    if statistical_type == 'continuous':
        maxx = np.max(domain)
        minx = np.min(domain)

        if np.var(data) > 1e-10:
            breaks, densities, mids = getHistogramVals(data)
        else:
            breaks = np.array([minx, maxx])
            densities = np.array([1 / (maxx - minx)])

    elif statistical_type in {'discrete', 'categorical'}:
        breaks = np.array([d for d in domain] + [domain[-1] + 1])
        densities, breaks = np.histogram(data, bins=breaks, density=True)

    else:
        raise Exception('Invalid statistical type: ' + statistical_type)

    # laplacian smoothing
    if alpha:
        n_samples = data.shape[0]
        n_bins = len(breaks) - 1
        counts = densities * n_samples
        densities = (counts + alpha) / (n_samples + n_bins * alpha)

    assert (len(densities) == len(breaks) - 1)

    return Histogram(breaks.tolist(), densities.tolist())


def getHistogramVals(data):
    from rpy2 import robjects
    init_rpy()

    result = robjects["getHistogram"](data)
    breaks = np.asarray(result[0])
    densities = np.asarray(result[2])
    mids = np.asarray(result[3])

    return breaks, densities, mids


def add_domains(data, ds_context):
    domain = []

    for col in range(data.shape[1]):
        feature_type = ds_context.statistical_type[col]
        if feature_type == 'continuous':
            domain.append([np.min(data[:, col]), np.max(data[:, col])])
        elif feature_type in {'discrete', 'categorical'}:
            domain.append(np.unique(data[:, col]))

    ds_context.domain = np.asanyarray(domain)

