# import unittest
import numpy as np
import sys

from spn.structure.leaves.conditional.MPE import add_conditional_mpe_support

sys.path.append("/home/shao/simple_spn/simple_spn/src")

from spn.algorithms.Inference import log_likelihood, conditional_log_likelihood
from spn.algorithms.LearningWrappers import learn_conditional
from spn.io.Text import spn_to_str_equation
from spn.structure.Base import Context, Sum, assign_ids
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.conditional.Inference import add_conditional_inference_support
from spn.structure.leaves.conditional.Conditional import Conditional_Poisson, Conditional_Bernoulli


if __name__ == "__main__":
    add_conditional_inference_support()

    np.random.seed(42)
    dataIn = np.random.randint(low=0, high=3, size=600).reshape(-1, 2)
    dataOut = np.random.randint(low=0, high=3, size=1200).reshape(-1, 4)
    data = np.concatenate((dataOut, dataIn), axis=1)
    assert data.shape[1] == dataIn.shape[1] + dataOut.shape[1], "invalid column size"
    assert data.shape[0] == dataIn.shape[0] == dataOut.shape[0], "invalid row size"

    ds_context = Context(meta_types=[MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE])
    ds_context.add_domains(dataOut)
    ds_context.parametric_types = [Conditional_Poisson, Conditional_Poisson, Conditional_Poisson, Conditional_Poisson]

    scope = list(range(dataOut.shape[1]))

    spn = Sum()

    for label, count in zip(*np.unique(data[:, 2], return_counts=True)):
        branch = learn_conditional(data, ds_context, scope, min_instances_slice=10000)
        spn.children.append(branch)
        spn.weights.append(count / data.shape[0])

    spn.scope.extend(branch.scope)

    assign_ids(spn)

    print(spn)

    mpe_test = data[[0, 1, 2], :].astype(float)
    mpe_test[:, 0] = np.nan

    from spn.algorithms.MPE import mpe

    add_conditional_mpe_support()

    print(mpe(spn, mpe_test)[:, 0])

    # class TestBase(unittest.TestCase):
#
#     def test_bfs(self):
#         add_parametric_inference_support()
#
#         np.random.seed(42)
#         data = np.random.randint(low=0, high=3, size=600).reshape(-1, 3)
#
#         # print(data)
#
#         ds_context = Context(meta_types=[MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE])
#         ds_context.add_domains(data)
#         ds_context.parametric_type = [Poisson, Poisson, Categorical]
#
#         spn = Sum()
#
#         for label, count in zip(*np.unique(data[:, 2], return_counts=True)):
#             branch = learn_parametric(data[data[:, 2] == label, :], ds_context, min_instances_slice=10000)
#             spn.children.append(branch)
#             spn.weights.append(count / data.shape[0])
#
#         spn.scope.extend(branch.scope)
#
# if __name__ == '__main__':
#     unittest.main()
