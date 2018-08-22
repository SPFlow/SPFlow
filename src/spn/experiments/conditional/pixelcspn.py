import sys;

sys.path.append('/home/shao/simple_spn/simple_spn/src')

from spn.data.datasets import get_binary_data, get_nips_data, get_mnist
from os.path import dirname;

path = dirname(__file__)
import numpy as np; np.random.seed(42)

from spn.algorithms.LearningWrappers import learn_conditional, learn_structure, learn_parametric
from spn.structure.Base import Context, Sum
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.conditional.Inference import add_conditional_inference_support
from spn.structure.leaves.conditional.Conditional import Conditional_Poisson, Conditional_Bernoulli, Conditional_Gaussian
from spn.structure.leaves.parametric.Parametric import Poisson, Gaussian
from spn.io.Graphics import plot_spn

import matplotlib

matplotlib.use('Agg')
import scipy
import pickle

spn_file = path + "/spn_file_poisson"
cspn_file = path + "/cspn_file_poisson"

if __name__ == '__main__':
    images_tr, labels_tr, images_te, labels_te = get_mnist()
    images_tr = images_tr # / 255.
    images_te = images_te # / 255.
    print("mnist loaded")
    images = np.reshape(images_tr, (-1, 28, 28))

    downscaleto = 28
    domain = 'poisson'
    horizontal_middle = downscaleto // 2
    vertical_middle = downscaleto // 2

    downscaled_image = np.asarray([scipy.misc.imresize(image, (downscaleto, downscaleto)) for image in np.asarray(images)], dtype=int)
    labels = np.asarray(labels_tr)

    data = []
    for i in range(0, 10):
        data.extend(downscaled_image[labels == i][:700])
    data = np.asarray(data)

    poisson_noise = np.random.poisson(lam=5, size=data.shape)
    data += poisson_noise

    blocked_images = (data[:, :horizontal_middle, :vertical_middle].reshape(len(data), -1),
                      data[:, :horizontal_middle, vertical_middle:].reshape(len(data), -1),
                      data[:, horizontal_middle:, :vertical_middle].reshape(len(data), -1),
                      data[:, horizontal_middle:, vertical_middle:].reshape(len(data), -1))

    # spn
    ds_context = Context(meta_types=[MetaType.DISCRETE] * blocked_images[0].shape[1])
    ds_context.add_domains(blocked_images[0])
    ds_context.parametric_type = [Poisson] * blocked_images[0].shape[1]

    print("data ready", data.shape)
    #the following two options should be working now.
    # spn = learn_structure(upperimage, ds_context, get_split_rows_random_partition(np.random.RandomState(17)), get_split_cols_random_partition(np.random.RandomState(17)), create_parametric_leaf)
    spn = learn_parametric(blocked_images[0], ds_context, min_instances_slice=0.1*len(data), ohe=False)


    fileObject = open(path + "/spn_block", 'wb')
    pickle.dump(spn, fileObject)
    fileObject.close()

    cspn_army = []
    for i in range(3):
        # cspn
        dataIn = blocked_images[i]
        dataOut = blocked_images[i+1]

        # assert data.shape[1] == dataIn.shape[1] + dataOut.shape[1], 'invalid column size'
        # assert data.shape[0] == dataIn.shape[0] == dataOut.shape[0], 'invalid row size'

        ds_context = Context(meta_types=[MetaType.DISCRETE] * dataOut.shape[1])
        ds_context.add_domains(dataOut)
        ds_context.parametric_type = [Conditional_Poisson] * dataOut.shape[1]


        scope = list(range(dataOut.shape[1]))

        cspn = learn_conditional(np.concatenate((dataOut, dataIn), axis=1), ds_context, scope, min_instances_slice=0.1*len(data))
        cspn_army.append(cspn)
        print(cspn)
        plot_spn(cspn, 'basicspn.png')

        fileObject = open(path + "/cspn_block%s"%i, 'wb')
        pickle.dump(cspn, fileObject)
        fileObject.close()


    from spn.structure.leaves.conditional.Sampling import add_conditional_sampling_support
    add_conditional_inference_support()
    add_conditional_sampling_support()

    from numpy.random.mtrand import RandomState
    from spn.algorithms.Sampling import sample_instances
    num_samples = 30
    num_half_image_pixels = downscaleto*downscaleto//4
    block_a_samples = sample_instances(spn, np.array([[np.nan] * num_half_image_pixels] * num_samples).reshape(-1, num_half_image_pixels), RandomState(123))

    block_samples = block_a_samples
    final_samples = block_a_samples
    for i in range(3):
        samples_placholder = np.concatenate((np.array([[np.nan] * num_half_image_pixels] * block_samples.shape[0]).reshape(-1, num_half_image_pixels), block_samples), axis=1)
        sample_images = sample_instances(cspn_army[i], samples_placholder, RandomState(123))
        block_samples = sample_images[:, :num_half_image_pixels]
        final_samples = np.concatenate((final_samples, block_samples), axis=1)

    assert np.shape(final_samples) == (60000, 28, 28), 'samples has wrong shape %s' % np.shape(final_samples)

    for idx, image in enumerate(final_samples):
        if domain == 'gaussian':
            scipy.misc.imsave('sample_pixelcspn%s.png'%idx, image)
        elif domain == 'poisson':
            scipy.misc.imsave('sample_pixelcspn%s.png'%idx, image)
        else:
            raise NotImplementedError