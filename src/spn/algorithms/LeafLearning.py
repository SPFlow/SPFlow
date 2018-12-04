"""
@author: Claas Voelcker
"""
from spn.structure.Base import Leaf
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf, Histogram
from spn.structure.leaves.parametric.Parametric import create_parametric_leaf, Parametric
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf, PiecewiseLinear

# from spn.structure.leaves.conditional.Conditional import create_conditional_leaf, Conditional


def learn_leaf_from_context(data, ds_context, scope):
    """
    Wrapper function to infer leaf type from the context object
    :param data: np.array: the data slice
    :param ds_context: Context: the context oobject for the data/spn
    :param scope: List: the scope of the variables
    :return: a correct leaf
    """
    assert len(scope) == 1, "scope for more than one variable?"
    idx = scope[0]

    conditional_type = ds_context.parametric_types[idx]
    assert issubclass(conditional_type, Leaf), "no instance of leaf "

    if issubclass(conditional_type, Parametric):
        return create_parametric_leaf(data, ds_context, scope)
    if issubclass(conditional_type, Conditional):
        return create_conditional_leaf(data, ds_context, scope)
    if issubclass(conditional_type, Histogram):
        return create_histogram_leaf(data, ds_context, scope)
    if issubclass(conditional_type, PiecewiseLinear):
        return create_piecewise_leaf(data, ds_context, scope)
    raise Exception("No fitting leaf type found")
