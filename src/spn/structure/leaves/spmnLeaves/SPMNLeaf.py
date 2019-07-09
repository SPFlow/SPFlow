import numpy as np
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Histograms import Histogram, create_histogram_leaf


# class SPMNLeaf(Histogram):
#
#     def __init__(self, scope):

def create_spmn_leaf(data, ds_context, scope):
    assert len(scope) == 1, "scope of univariate histogram for more than one variable?"
    assert data.shape[1] == 1, "data has more than one feature?"

    # data = data[~np.isnan(data)]

    idx = scope[0]
    meta_type = ds_context.meta_types[idx]

    if meta_type == MetaType.UTILITY:
        return Utility(data, ds_context, scope, idx)
    else:
        return create_histogram_leaf(data, ds_context, scope)


class Utility(Histogram):

    def __init__(self, data, ds_context, scope, idx):

        hist = create_histogram_leaf(data, ds_context, scope)
        Histogram.__init__(self, hist.breaks, hist.densities, hist.bin_repr_points, scope=idx, type_=None, meta_type=MetaType.UTILITY)
