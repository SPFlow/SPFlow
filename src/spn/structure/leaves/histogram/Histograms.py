"""
Created on March 20, 2018

@author: Alejandro Molina
"""
from collections import namedtuple

import numpy as np

from spn.structure.Base import Leaf
from spn.structure.StatisticalTypes import MetaType, Type

rpy_initialized = False


def init_rpy():
    if rpy_initialized:
        return

    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    import os

    path = os.path.dirname(__file__)
    with open(path + "/Histogram.R", "r") as rfile:
        code = "".join(rfile.readlines())
        robjects.r(code)

    numpy2ri.activate()


class Histogram(Leaf):

    type = Type.CATEGORICAL
    property_type = namedtuple("Histogram", "breaks densities bin_repr_points")

    def __init__(self, breaks, densities, bin_repr_points, scope=None, type_=None, meta_type=MetaType.DISCRETE):
        Leaf.__init__(self, scope=scope)
        self.type = type(self).type if not type_ else type_
        self.meta_type = meta_type
        self.breaks = breaks
        self.densities = densities
        self.bin_repr_points = bin_repr_points

    @property
    def parameters(self):
        return __class__.property_type(
            breaks=self.breaks, densities=self.densities, bin_repr_points=self.bin_repr_points
        )


def create_histogram_leaf(data, ds_context, scope, alpha=1.0, hist_source="numpy"):
    assert len(scope) == 1, "scope of univariate histogram for more than one variable?"
    assert data.shape[1] == 1, "data has more than one feature?"

    data = data[~np.isnan(data)]

    idx = scope[0]
    meta_type = ds_context.meta_types[idx]
    domain = ds_context.domains[idx]

    assert not np.isclose(np.max(domain), np.min(domain)), "invalid domain, min and max are the same"

    if data.shape[0] == 0:
        # no data or all were nans
        maxx = np.max(domain)
        minx = np.min(domain)
        breaks = np.array([minx, maxx])
        densities = np.array([1 / (maxx - minx)])
        repr_points = np.array([minx + (maxx - minx) / 2])
        if meta_type == MetaType.DISCRETE:
            repr_points = repr_points.astype(int)

    elif np.var(data) == 0 and meta_type == MetaType.REAL:
        # one data point
        maxx = np.max(domain)
        minx = np.min(domain)
        breaks = np.array([minx, maxx])
        densities = np.array([1 / (maxx - minx)])
        repr_points = np.array([minx + (maxx - minx) / 2])
        if meta_type == MetaType.DISCRETE:
            repr_points = repr_points.astype(int)

    else:
        breaks, densities, repr_points = getHistogramVals(data, meta_type, domain, source=hist_source)

    # laplace smoothing
    if alpha:
        n_samples = data.shape[0]
        n_bins = len(breaks) - 1
        counts = densities * n_samples
        densities = (counts + alpha) / (n_samples + n_bins * alpha)

    assert len(densities) == len(breaks) - 1

    return Histogram(breaks.tolist(), densities.tolist(), repr_points.tolist(), scope=idx, meta_type=meta_type)


def getHistogramVals(data, meta_type, domain, source="numpy"):
    # check this: https://github.com/theodoregoetz/histogram

    if meta_type == MetaType.DISCRETE:
        # for discrete, we just have to count
        breaks = np.array([d for d in domain] + [domain[-1] + 1])
        densities, breaks = np.histogram(data, bins=breaks, density=True)
        repr_points = domain
        return breaks, densities, repr_points

    if source == "R":
        from rpy2 import robjects

        init_rpy()

        result = robjects.r["getHistogram"](data)
        breaks = np.asarray(result[0])
        densities = np.asarray(result[2])
        mids = np.asarray(result[3])

        return breaks, densities, mids

    if source == "kde":
        import statsmodels.api as sm

        kde = sm.nonparametric.KDEMultivariate(data, var_type="c", bw="cv_ls")
        bins = int((domain[1] - domain[0]) / kde.bw)
        bins = min(30, bins)
        cdf_x = np.linspace(domain[0], domain[1], 2 * bins)
        cdf_y = kde.cdf(cdf_x)
        breaks = np.interp(np.linspace(0, 1, bins), cdf_y, cdf_x)  # inverse cdf
        mids = ((breaks + np.roll(breaks, -1)) / 2.0)[:-1]

        densities = kde.pdf(mids)
        densities / np.sum(densities)

        if len(densities.shape) == 0:
            densities = np.array([densities])

        return breaks, densities, mids

    if source == "numpy":
        densities, breaks = np.histogram(data, bins="auto", density=True)
        mids = ((breaks + np.roll(breaks, -1)) / 2.0)[:-1]
        return breaks, densities, mids

    if source == "astropy":
        from astropy.stats import histogram

        densities, breaks = histogram(data, bins="blocks", density=True)
        mids = ((breaks + np.roll(breaks, -1)) / 2.0)[:-1]
        return breaks, densities, mids

    assert False, "unkown histogram method " + source
