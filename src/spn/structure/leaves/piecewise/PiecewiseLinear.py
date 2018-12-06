"""
Created on May 4, 2018

@author: Alejandro Molina
@author: Antonio Vergari
"""
from collections import namedtuple

import numpy as np

from spn.structure.Base import Leaf
from spn.structure.StatisticalTypes import MetaType, Type
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
import itertools


class PiecewiseLinear(Leaf):
    type = Type.REAL
    property_type = namedtuple("PiecewiseLinear", "x_range y_range bin_repr_points")

    def __init__(self, x_range, y_range, bin_repr_points, scope=None):
        Leaf.__init__(self, scope=scope)
        self._type = type
        self.x_range = x_range
        self.y_range = y_range

        self.bin_repr_points = bin_repr_points

    @property
    def parameters(self):
        return __class__.property_type(x_range=self.x_range, y_range=self.y_range, bin_repr_points=self.bin_repr_points)

    @property
    def mode(self):
        areas = np.zeros(len(self.x_range) - 1)
        for i in range(areas.shape[0]):
            areas[i] = np.trapz([self.y_range[i], self.y_range[i + 1]], [self.x_range[i], self.x_range[i + 1]])
        # areas = np.diff(self.breaks) * self.densities
        max_area = np.argmax(areas)
        max_x = np.argmax([self.y_range[max_area], self.y_range[max_area + 1]]) + max_area
        return self.x_range[max_x]

    @property
    def mean(self):
        # Use the histogram values to compute the mean
        y_range_norm = self.y_range / np.sum(self.y_range)

        mean = 0.0
        for k, x in enumerate(self.x_range):
            mean += x * y_range_norm[k]

        return mean

    @property
    def types(self):
        return self._type


def isotonic_unimodal_regression_R(x, y):
    """
    Perform unimodal isotonic regression via the Iso package in R
    """
    from rpy2.robjects.packages import importr
    from rpy2 import robjects
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()
    # n_instances = x.shape[0]
    # assert y.shape[0] == n_instances

    importr("Iso")
    z = robjects.r["ufit"](y, x=x, type="b")
    iso_x, iso_y = np.array(z.rx2("x")), np.array(z.rx2("y"))

    return iso_x, iso_y


def create_piecewise_leaf(data, ds_context, scope, isotonic=False, prior_weight=0.1, hist_source="numpy"):
    assert len(scope) == 1, "scope of univariate Piecewise for more than one variable?"
    assert data.shape[1] == 1, "data has more than one feature?"

    idx = scope[0]
    meta_type = ds_context.meta_types[idx]

    hist = create_histogram_leaf(data, ds_context, scope, alpha=False, hist_source=hist_source)
    densities = hist.densities
    bins = hist.breaks
    repr_points = hist.bin_repr_points

    if meta_type == MetaType.REAL:
        EPS = 1e-8
        if len(densities) > 1:

            def pairwise(iterable):
                "s -> (s0,s1), (s1,s2), (s2, s3), ..."
                a, b = itertools.tee(iterable)
                next(b, None)
                return zip(a, b)

            x = [bins[0] - EPS] + [b0 + (b1 - b0) / 2 for (b0, b1) in pairwise(bins)] + [bins[-1] + EPS]
        else:
            assert len(bins) == 2
            x = [bins[0] - EPS] + [(bins[0] + (bins[1] - bins[0]) / 2)] + [bins[-1] + EPS]

    elif meta_type == MetaType.DISCRETE:
        tail_width = 1
        x = [b for b in bins[:-1]]
        x = [x[0] - tail_width] + x + [x[-1] + tail_width]

    else:
        raise Exception("Invalid statistical type: " + meta_type)

    y = [0.0] + [d for d in densities] + [0.0]

    assert len(densities) == len(bins) - 1
    assert len(x) == len(y), (len(x), len(y))
    x, y = np.array(x), np.array(y)

    if isotonic:
        x, y = isotonic_unimodal_regression_R(x, y)

    auc = np.trapz(y, x)
    y = y / auc

    node = PiecewiseLinear(x.tolist(), y.tolist(), repr_points, scope=scope)

    if prior_weight is None:
        return node

    uniform_data = np.zeros_like(data)
    uniform_data[:] = np.nan
    uniform_hist = create_histogram_leaf(uniform_data, ds_context, scope, alpha=False)
    return prior_weight * uniform_hist + (1 - prior_weight) * node
