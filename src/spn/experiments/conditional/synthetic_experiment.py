'''
Created on August 14, 2018

@author: Alejandro Molina
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import RandomState

from spn.algorithms.LearningWrappers import learn_conditional
from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import Context
from spn.structure.leaves.conditional.Conditional import Conditional_Poisson
from spn.structure.leaves.conditional.Inference import add_conditional_inference_support
from spn.structure.leaves.conditional.Sampling import add_conditional_sampling_support
import pickle
import os

def create_images_horizontal_lines(rows, cols):
    result = np.zeros((rows, rows, cols))
    for i in range(rows):
        result[i, i, :] = 1

    return result.reshape((rows, -1))


def plot_img(image, rows, cols):
    plt.imshow(image.reshape(rows, cols), cmap='Greys', interpolation='nearest')

    plt.show()


if __name__ == '__main__':
    add_conditional_inference_support()
    add_conditional_sampling_support()

    px = 10
    py = 20

    images = create_images_horizontal_lines(px, py)
    images2d = images.reshape(-1, px, py)
    middle = py // 2
    left = images2d[:, :, :middle].reshape((images.shape[0], -1))
    right = images2d[:, :, middle:].reshape((images.shape[0], -1))

    ds_context = Context(parametric_types=[Conditional_Poisson] * right.shape[1]).add_domains(right)

    scope = list(range(right.shape[1]))

    # In left, OUT right
    file_cache_path = "/tmp/csn.bin"
    if not os.path.isfile(file_cache_path) or True:
        cspn = learn_conditional(np.concatenate((right, left), axis=1), ds_context, scope, min_instances_slice=60000000)
        with open(file_cache_path, 'wb') as f:
            pickle.dump(cspn, f, pickle.HIGHEST_PROTOCOL)

    with open(file_cache_path, 'rb') as f:
        cspn = pickle.load(f)

    sample_input = np.concatenate((np.zeros_like(right)/0, left), axis=1)

    sample_images = sample_instances(cspn, sample_input, RandomState(123))


    # plot_img(images[1, :], px, py)
    plot_img(sample_input[0, :], px, py)

    plot_img(sample_images[0, :], px, py)
