from joblib import Memory
from matplotlib.colors import LogNorm, PowerNorm

from spn.algorithms.Inference import log_likelihood, likelihood
from spn.algorithms.LearningWrappers import learn_mspn, learn_parametric
from numpy import genfromtxt
import numpy as np

from spn.algorithms.Marginalization import marginalize
from spn.structure.Base import Context, Sum
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Inference import add_histogram_inference_support
from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical
from spn.structure.leaves.piecewise.Inference import add_piecewise_inference_support
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf
import matplotlib.cm as cm

memory = Memory(cachedir="cache", verbose=0, compress=9)

def plot_density(spn, data):
    import matplotlib.pyplot as plt
    import numpy as np

    x_max = data[:, 0].max()
    x_min = data[:, 0].min()
    y_max = data[:, 1].max()
    y_min = data[:, 1].min()

    nbinsx = int(x_max - x_min)/1
    nbinsy = int(y_max - y_min)/1
    xi, yi = np.mgrid[x_min:x_max:nbinsx * 1j, y_min:y_max:nbinsy * 1j]

    spn_input = np.vstack([xi.flatten(), yi.flatten()]).T

    marg_spn = marginalize(spn, set([0, 1]))

    zill = likelihood(marg_spn, spn_input)

    z = zill.reshape(xi.shape)

    # Make the plot
    # plt.pcolormesh(xi, yi, z)

    plt.imshow(z+1, extent=(x_min, x_max, y_min, y_max), cmap=cm.hot, norm=PowerNorm(gamma=1./5.))
    #plt.pcolormesh(xi, yi, z)
    plt.colorbar()
    plt.show()

    # Change color palette
    # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r)
    # plt.show()


if __name__ == '__main__':
    add_piecewise_inference_support()
    add_histogram_inference_support()

    data = genfromtxt('20180511-for-SPN.csv', delimiter=',', skip_header=True)[:, [0, 1, 3]]

    print(data)

    ds_context = Context(meta_types=[MetaType.REAL, MetaType.REAL, MetaType.DISCRETE])
    #ds_context.parametric_type = [Gaussian, Gaussian, Categorical]
    ds_context.add_domains(data)

    spn = Sum()


    def create_leaf(data, ds_context, scope):
        return create_piecewise_leaf(data, ds_context, scope, isotonic=False, prior_weight=None)


    for label, count in zip(*np.unique(data[:, 2], return_counts=True)):
        #branch = learn_parametric(data[data[:, 2] == label, :], ds_context, min_instances_slice=100, leaves=create_leaf, memory=memory)
        branch = learn_mspn(data[data[:, 2] == label, :], ds_context, min_instances_slice=100, leaves=create_leaf, memory=memory)
        spn.children.append(branch)
        spn.weights.append(count / data.shape[0])

    spn.scope.extend(branch.scope)

    print("learned")

    plot_density(spn, data)
