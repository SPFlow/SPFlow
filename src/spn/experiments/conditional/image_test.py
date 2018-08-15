from spn.data.datasets import get_binary_data, get_nips_data, get_mnist
#from os.path import dirname; path = dirname(__file__)
import numpy as np
#import sys; sys.path.append('/home/shao/simple_spn/simple_spn/src')

from spn.algorithms.Inference import log_likelihood, conditional_log_likelihood
from spn.algorithms.LearningWrappers import learn_conditional, learn_structure, learn_parametric
from spn.io.Text import spn_to_str_equation
from spn.structure.Base import Context, Sum
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.conditional.Inference import add_conditional_inference_support
from spn.structure.leaves.conditional.Conditional import Conditional_Poisson, Conditional_Bernoulli
from spn.structure.leaves.parametric.Parametric import Poisson
from spn.io.Graphics import plot_spn


from spn.algorithms.splitting.Clustering import get_split_rows_KMeans, get_split_rows_Gower
from spn.algorithms.splitting.RDC import get_split_cols_RDC, get_split_cols_RDC_py
from spn.algorithms.splitting.Random import get_split_cols_binary_random_partition, \
    get_split_rows_random_partition, get_split_cols_random_partition
from spn.structure.leaves.parametric.Parametric import create_parametric_leaf

import matplotlib
matplotlib.use('Agg')
import scipy

if __name__ == '__main__':
    images_tr, labels_tr, images_te, labels_te = get_mnist()
    print("mnist loaded")
    images = np.reshape(images_tr, (-1, 28, 28))
    downscaled_image = np.asarray([scipy.misc.imresize(image, (9,9)) for image in np.asarray(images)], dtype=int)
    #toimage = [scipy.misc.toimage(image) for image in downscaled_image]

    #add_conditional_inference_support()

    labels = np.asarray(labels_tr)
    zeros = downscaled_image[labels==0]
    ones = downscaled_image[labels==1]
    data = np.concatenate((zeros, ones), axis=0).reshape((-1, 81))


    ds_context = Context(meta_types=[MetaType.DISCRETE] * data.shape[1])
    ds_context.add_domains(data)
    ds_context.parametric_type = [Poisson] * data.shape[1]

    print("data ready", data.shape)
    #the following two options should be working now.
    spn = learn_structure(data, ds_context, get_split_rows_random_partition(np.random.RandomState(17)), get_split_cols_random_partition(np.random.RandomState(17)), create_parametric_leaf)
    spn = learn_parametric(data, ds_context, min_instances_slice=1000, ohe=False)
    print(spn)
    plot_spn(spn, 'basicspn.png')


    # np.random.seed(42)
    # assert data.shape[1] == dataIn.shape[1] + dataOut.shape[1], 'invalid column size'
    # assert data.shape[0] == dataIn.shape[0] == dataOut.shape[0], 'invalid row size'
    #
    # ds_context = Context(meta_types=[MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE])
    # ds_context.add_domains(dataOut)
    # ds_context.parametric_type = [Conditional_Poisson, Conditional_Poisson, Conditional_Poisson, Conditional_Poisson]
    #
    # scope = list(range(dataOut.shape[1]))
    #
    # spn = Sum()
    #
    # for label, count in zip(*np.unique(data[:, 2], return_counts=True)):
    #     branch = learn_parametric(data, ds_context, scope, min_instances_slice=10000)
    #     spn.children.append(branch)
    #     spn.weights.append(count / data.shape[0])
    #
    # spn.scope.extend(branch.scope)
    #
    #
    # print(spn)
    # plot_spn(spn, 'basicspn.png')
