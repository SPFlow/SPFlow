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
from spn.structure.leaves.parametric.Parametric import Poisson, Gaussian
from spn.io.Graphics import plot_spn

import matplotlib

matplotlib.use("Agg")
import scipy
import pickle

spn_file = path + "/spn_file_poisson"
cspn_file = path + "/cspn_file_poisson"

if __name__ == "__main__":

    images_tr, labels_tr, images_te, labels_te = get_mnist()
    images_tr = images_tr  # / 255.
    images_te = images_te  # / 255.
    print("mnist loaded")
    images = np.reshape(images_tr, (-1, 28, 28))

    downscaleto = 18
    domain = "poisson"
    horizontal_middle = downscaleto // 2
    vertical_middle = downscaleto // 2

    downscaled_image = np.asarray(
        [scipy.misc.imresize(image, (downscaleto, downscaleto)) for image in np.asarray(images)], dtype=int
    )
    labels = np.asarray(labels_tr)

    data = []
    for i in range(10):
        data.extend(downscaled_image[labels == i][:700])
    data = np.asarray(data)

    poisson_noise = np.random.poisson(lam=10, size=data.shape)
    data += poisson_noise
    # data = data / float(np.max(data))

    blocked_images = (
        data[:, :horizontal_middle, :vertical_middle].reshape(len(data), -1),  # top left
        data[:, :horizontal_middle, vertical_middle:].reshape(len(data), -1),  # top right
        data[:, horizontal_middle:, :vertical_middle].reshape(len(data), -1),  # bottom left
        data[:, horizontal_middle:, vertical_middle:].reshape(len(data), -1),
    )  # bottom right

    """
    zeros = np.zeros(data.shape)
    print('zeros shape', np.shape(zeros))
    dataIn = data[:, :horizontal_middle, :vertical_middle].reshape(len(data), -1)
    dataOut = data[:, :horizontal_middle, vertical_middle:].reshape(len(data), -1)
    print(data[0])
    print("_______")
    print(dataIn[0])
    print("_______")
    print(dataOut[0])
    print("_______")
    zeros[:, :horizontal_middle, :vertical_middle] = dataIn.reshape(len(data), 4, 4)  #data[:, :horizontal_middle, :vertical_middle]
    zeros[:, :horizontal_middle, vertical_middle:] = dataOut.reshape(len(data), 4, 4) #data[:, :horizontal_middle, vertical_middle:]
    print(zeros[0], np.shape(zeros))    #print(np.concatenate((dataIn, dataOut), axis=1).reshape(len(dataIn), 4, 8)[0])
    """

    # spn
    ds_context = Context(meta_types=[MetaType.REAL] * blocked_images[0].shape[1])
    ds_context.add_domains(blocked_images[0])
    ds_context.parametric_types = [Poisson] * blocked_images[0].shape[1]

    print("data ready", data.shape)
    # the following two options should be working now.
    spn = learn_parametric(blocked_images[0], ds_context, min_instances_slice=0.1 * len(data), ohe=False)

    # cspn
    dataIn = blocked_images[0]  # data[:, :horizontal_middle, :vertical_middle].reshape(len(data), -1)
    dataOut = blocked_images[1]  # data[:, :horizontal_middle, vertical_middle:].reshape(len(data), -1)

    ds_context = Context(meta_types=[MetaType.REAL] * dataOut.shape[1])
    ds_context.add_domains(dataOut)
    ds_context.parametric_types = [Conditional_Poisson] * dataOut.shape[1]

    scope = list(range(dataOut.shape[1]))
    print(np.shape(dataIn), np.shape(dataOut))

    cspn = learn_conditional(
        np.concatenate((dataOut, dataIn), axis=1), ds_context, scope, min_instances_slice=0.4 * len(data)
    )
    plot_spn(cspn, "basicspn.png")

    # start sampling
    from spn.structure.leaves.conditional.Sampling import add_conditional_sampling_support

    add_conditional_inference_support()
    add_conditional_sampling_support()

    from numpy.random.mtrand import RandomState
    from spn.algorithms.Sampling import sample_instances

    num_samples = 30
    num_half_image_pixels = downscaleto * downscaleto // 4
    samples_placeholder = np.array([[np.nan] * num_half_image_pixels] * num_samples).reshape(-1, num_half_image_pixels)
    top_left_samples = sample_instances(spn, samples_placeholder, RandomState(123))

    samples_placeholder = np.concatenate(
        (np.array([[np.nan] * num_half_image_pixels] * top_left_samples.shape[0]), top_left_samples), axis=1
    )
    sample_images = sample_instances(cspn, samples_placeholder, RandomState(123))
    top_right_samples = sample_images[:, :num_half_image_pixels]

    # tmp = np.zeros((num_samples, 8, 8))
    # tmp[:, :4, :4] = top_left_samples.reshape(num_samples, 4, 4)
    # tmp[:, :4, 4:] = top_right_samples.reshape(num_samples, 4, 4)

    final_samples = np.zeros((num_samples, downscaleto, downscaleto))
    final_samples[:, :horizontal_middle, :vertical_middle] = top_left_samples.reshape(
        num_samples, horizontal_middle, vertical_middle
    )
    final_samples[:, :horizontal_middle, vertical_middle:] = top_right_samples.reshape(
        num_samples, horizontal_middle, vertical_middle
    )

    print("final_samples", final_samples[0])
    print("top_left_samples", top_left_samples[0])
    print("top_right_samples", top_right_samples[0])

    for idx, image in enumerate(final_samples):
        scipy.misc.imsave("sample_pixelcspn%s.png" % idx, image)
