from spn.algorithms.Inference import log_likelihood
from spn.io.Text import spn_to_str_equation
from spn.structure.Base import Leaf, Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Histograms import add_domains, create_histogram_leaf
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.structure.leaves.parametric.Text import add_parametric_text_support
import numpy as np

from spn.structure.leaves.piecewise.Expectation import piecewise_expectation
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf

_node_expectation = {}


def add_node_expectation(node_type, lambda_func):
    _node_expectation[node_type] = lambda_func

def Expectation(spn, feature_scope, evidence, ds_context, node_expectation=_node_expectation):
    def leaf_expectation(node, data, dtype=np.float64, node_log_likelihood=None):
        if node.scope[0] == feature_scope:
            t_node = type(node)
            if t_node in node_expectation:
                return node_expectation[type(node)](node, ds_context)
            else:
                raise Exception('Node type unknown: ' + str(t_node))

        return log_likelihood(node, evidence)

    log_expectation = log_likelihood(spn, evidence, node_log_likelihood={Leaf: leaf_expectation})

    return np.exp(log_expectation)


if __name__ == '__main__':
    add_parametric_text_support()

    spn = 0.3 * (Gaussian(1.0, 1.0, scope=[0]) * Gaussian(5.0, 1.0, scope=[1])) + \
          0.7 * (Gaussian(10.0, 1.0, scope=[0]) * Gaussian(15.0, 1.0, scope=[1]))

    print(spn_to_str_equation(spn))

    data = np.array([0, 0, 1, 3]).reshape(-1, 1)
    data = np.random.rand(5000).reshape(-1, 1)

    ds_context = Context(meta_types=[MetaType.DISCRETE])
    ds_context = Context(meta_types=[MetaType.REAL])
    add_domains(data, ds_context)

    node = create_piecewise_leaf(data, ds_context, scope=[0], prior_weight=None)

    exp = piecewise_expectation(node)

    print("DISCRETE", "mean should be ", np.mean(data), "is", exp)
