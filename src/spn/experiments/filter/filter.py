from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_mspn
from numpy import genfromtxt
import numpy as np

from spn.algorithms.Marginalization import marginalize
from spn.structure.Base import Context, Sum
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Inference import add_histogram_inference_support
from spn.structure.leaves.piecewise.Inference import add_piecewise_inference_support
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf


def plot_density(spn, data):
    import matplotlib.pyplot as plt
    import numpy as np

    x_max = data[:,0].max()
    x_min = data[:,0].min()
    y_max = data[:,1].max()
    y_min = data[:,1].min()

    nbinsx = x_max - x_min
    nbinsy = y_max - y_min
    xi, yi = np.mgrid[x_min:x_max:nbinsx * 1j, y_min:y_max:nbinsy * 1j]


    spn_input = np.vstack([xi.flatten(), yi.flatten()])


    marg_spn = marginalize(spn, set([0,1]))

    zi = log_likelihood(marg_spn, spn_input.T).T

    # Make the plot
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
    plt.show()

    # Change color palette
    #plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r)
    #plt.show()


if __name__ == '__main__':
    add_piecewise_inference_support()
    add_histogram_inference_support()

    data = genfromtxt('20180511-for-SPN.csv', delimiter=',', skip_header=True)[:, [0, 1, 3]]

    print(data)

    ds_context = Context(meta_types=[MetaType.REAL, MetaType.REAL, MetaType.DISCRETE])
    ds_context.add_domains(data)

    spn = Sum()

    for label, count in zip(*np.unique(data[:, 2], return_counts=True)):
        branch = learn_mspn(data[data[:, 2] == label,:], ds_context, min_instances_slice=10000, leaves=create_piecewise_leaf)
        spn.children.append(branch)
        spn.weights.append(count / data.shape[0])

    spn.scope.extend(branch.scope)

    plot_density(spn, data)