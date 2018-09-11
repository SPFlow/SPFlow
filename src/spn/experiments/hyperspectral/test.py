import numpy as np

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric
from spn.io.Text import spn_to_str_equation
from spn.structure.Base import Context, Sum
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import Poisson, Categorical
from spn.structure.leaves.parametric.Text import add_parametric_text_support

if __name__ == '__main__':
    add_parametric_inference_support()
    add_parametric_text_support()

    np.random.seed(42)
    data = np.random.randint(low=0, high=3, size=600).reshape(-1, 3)

    #print(data)

    ds_context = Context(meta_types=[MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE])
    ds_context.add_domains(data)
    ds_context.parametric_types = [Poisson, Poisson, Categorical]

    spn = Sum()

    for label, count in zip(*np.unique(data[:, 2], return_counts=True)):
        branch = learn_parametric(data[data[:, 2] == label, :], ds_context, min_instances_slice=10000)
        spn.children.append(branch)
        spn.weights.append(count / data.shape[0])

    spn.scope.extend(branch.scope)

    print(spn)


    print(spn_to_str_equation(spn))

    print(log_likelihood(spn, data))


