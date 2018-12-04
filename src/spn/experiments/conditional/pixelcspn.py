import sys

sys.path.append("/home/shao/simple_spn/simple_spn/src")

from spn.data.datasets import get_binary_data, get_nips_data, get_mnist
from os.path import dirname

path = dirname(__file__)
import numpy as np

np.random.seed(42)

from spn.algorithms.LearningWrappers import learn_conditional, learn_structure, learn_parametric
from spn.structure.Base import Context, Sum
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.conditional.Inference import add_conditional_inference_support
from spn.structure.leaves.conditional.Conditional import (
    Conditional_Poisson,
    Conditional_Bernoulli,
    Conditional_Gaussian,
)
from spn.structure.leaves.parametric.Parametric import Poisson, Gaussian, Bernoulli, Categorical
from spn.io.Graphics import plot_spn

import matplotlib

matplotlib.use("Agg")
import scipy
import pickle

spn_file = path + "/spn_file_poisson"
cspn_file = path + "/cspn_file_poisson"


def one_hot(y):
    if len(y.shape) != 1:
        return y
    values = np.array(sorted(list(set(y))))
    return np.array([values == v for v in y], dtype=int)


if __name__ == "__main__":

    images_tr, labels_tr, images_te, labels_te = get_mnist()
    images_tr = images_tr  # / 255.
    images_te = images_te  # / 255.
    print("mnist loaded")
    images = np.reshape(images_tr, (-1, 28, 28))

    downscaleto = 28
    domain = "poisson"
    ohe = True
    horizontal_middle = downscaleto // 2
    vertical_middle = downscaleto // 2

    downscaled_image = np.asarray(
        [scipy.misc.imresize(image, (downscaleto, downscaleto)) for image in np.asarray(images)], dtype=int
    )
    labels = np.asarray(labels_tr)

    data = []
    data_labels = []
    for i in range(10):
        data.extend(downscaled_image[labels == i][:2000])
        data_labels.extend(labels_tr[labels == i][:2000])

    data = np.asarray(data)
    data_labels = np.asarray(data_labels).reshape(-1)
    if ohe is True:
        data_labels = one_hot(data_labels)

    poisson_noise = np.random.poisson(lam=5, size=data.shape)
    data += poisson_noise
    # data = data / float(np.max(data))

    blocked_images = (
        data[:, :horizontal_middle, :vertical_middle].reshape(len(data), -1),
        data[:, :horizontal_middle, vertical_middle:].reshape(len(data), -1),
        data[:, horizontal_middle:, :vertical_middle].reshape(len(data), -1),
        data[:, horizontal_middle:, vertical_middle:].reshape(len(data), -1),
    )

    # # spn
    # ds_context = Context(meta_types=[MetaType.REAL] * blocked_images[0].shape[1])
    # ds_context.add_domains(blocked_images[0])
    # ds_context.parametric_type = [Poisson] * blocked_images[0].shape[1]
    #
    # print("data ready", data.shape)
    # #the following two options should be working now.
    # # spn = learn_structure(upperimage, ds_context, get_split_rows_random_partition(np.random.RandomState(17)), get_split_cols_random_partition(np.random.RandomState(17)), create_parametric_leaf)
    # spn = learn_parametric(blocked_images[0], ds_context, min_instances_slice=0.1*len(data), ohe=False)

    # spn
    ds_context = Context(meta_types=[MetaType.DISCRETE] * 10)
    ds_context.add_domains(data_labels)
    ds_context.parametric_types = [Bernoulli] * blocked_images[0].shape[1]
    spn = learn_parametric(data_labels, ds_context, min_instances_slice=0.3 * len(data_labels))

    # first cspn
    dataIn = data_labels
    dataOut = blocked_images[0]
    ds_context = Context(meta_types=[MetaType.DISCRETE] * dataOut.shape[1])
    ds_context.add_domains(dataOut)
    ds_context.parametric_types = [Conditional_Poisson] * dataOut.shape[1]

    scope = list(range(dataOut.shape[1]))
    print(np.shape(dataIn), np.shape(dataOut))
    print(dataIn[0], dataOut[0])
    cspn_1st = learn_conditional(
        np.concatenate((dataOut, dataIn), axis=1), ds_context, scope, min_instances_slice=0.3 * len(data)
    )

    fileObject = open(path + "/cspn_1st", "wb")
    pickle.dump(cspn_1st, fileObject)
    fileObject.close()

    # a list of cspns
    cspn_army = []
    for i in range(3):
        print("cspn%s" % i)
        if i == 0:
            dataIn = blocked_images[i]
        else:
            dataIn = blocked_images[0]
            for j in range(1, i + 1):
                dataIn = np.concatenate((dataIn, blocked_images[j]), axis=1)
        dataIn = np.concatenate((dataIn, data_labels), axis=1)
        dataOut = blocked_images[i + 1]

        # assert data.shape[1] == dataIn.shape[1] + dataOut.shape[1], 'invalid column size'
        # assert data.shape[0] == dataIn.shape[0] == dataOut.shape[0], 'invalid row size'

        ds_context = Context(meta_types=[MetaType.DISCRETE] * dataOut.shape[1])
        ds_context.add_domains(dataOut)
        ds_context.parametric_types = [Conditional_Poisson] * dataOut.shape[1]

        scope = list(range(dataOut.shape[1]))
        cspn = learn_conditional(
            np.concatenate((dataOut, dataIn), axis=1), ds_context, scope, min_instances_slice=0.3 * len(data)
        )
        cspn_army.append(cspn)
        plot_spn(cspn, "basicspn%s.png" % i)

        fileObject = open(path + "/cspn_block%s" % i, "wb")
        pickle.dump(cspn, fileObject)
        fileObject.close()

    from spn.structure.leaves.conditional.Sampling import add_conditional_sampling_support

    add_conditional_inference_support()
    add_conditional_sampling_support()

    from numpy.random.mtrand import RandomState
    from spn.algorithms.Sampling import sample_instances

    num_samples = 30
    num_half_image_pixels = downscaleto * downscaleto // 4
    # block_samples_spn = sample_instances(spn, np.array([[np.nan] * num_half_image_pixels] * num_samples).reshape(-1, num_half_image_pixels), RandomState(123))
    annotation_spn = sample_instances(spn, np.array([[np.nan] * 10] * num_samples).reshape(-1, 10), RandomState(123))

    # sample 1st block
    samples_placholder = np.concatenate(
        (np.array([[np.nan] * num_half_image_pixels] * num_samples).reshape(-1, num_half_image_pixels), annotation_spn),
        axis=1,
    )
    block_samples_spn = sample_instances(cspn_1st, samples_placholder, RandomState(123))

    final_samples = np.zeros((num_samples, downscaleto, downscaleto))
    final_samples_block = [
        final_samples[:, :horizontal_middle, :vertical_middle],
        final_samples[:, :horizontal_middle, vertical_middle:],
        final_samples[:, horizontal_middle:, :vertical_middle],
        final_samples[:, horizontal_middle:, vertical_middle:],
    ]
    final_samples_block[0] = block_samples_spn.reshape(num_samples, horizontal_middle, vertical_middle)
    # tmp = []
    for i in range(3):
        current_block = final_samples_block[0].reshape(num_samples, -1)
        for j in range(1, i + 1):
            current_block = np.concatenate((current_block, final_samples_block[j].reshape(num_samples, -1)), axis=1)
        samples_placholder = np.concatenate(
            (
                np.array([[np.nan] * num_half_image_pixels] * num_samples).reshape(-1, num_half_image_pixels),
                current_block,
            ),
            axis=1,
        )
        sample_images = sample_instances(cspn_army[i], samples_placholder, RandomState(123))
        final_samples_block[i + 1] = sample_images[:, :num_half_image_pixels].reshape(
            num_samples, horizontal_middle, vertical_middle
        )
        # block_samples = np.concatenate((sample_images[:, -num_half_image_pixels:], sample_images[:, :-num_half_image_pixels]), axis=1)
        # block_samples = np.concatenate((sample_images[:, num_half_image_pixels:], sample_images[:, :num_half_image_pixels]), axis=1)
        # final_samples = np.concatenate((final_samples, block_samples), axis=1)

    final_samples[:, :horizontal_middle, :vertical_middle] = final_samples_block[0]
    final_samples[:, :horizontal_middle, vertical_middle:] = final_samples_block[1]
    final_samples[:, horizontal_middle:, :vertical_middle] = final_samples_block[2]
    final_samples[:, horizontal_middle:, vertical_middle:] = final_samples_block[3]
    print(final_samples[0])

    for idx, image in enumerate(final_samples):
        if domain == "gaussian":
            scipy.misc.imsave("sample_pixelcspn_gaussian%s.png" % idx, image)
        elif domain == "poisson":
            print(image)
            scipy.misc.imsave("sample_pixelcspn%s.png" % idx, image)
        else:
            raise NotImplementedError
