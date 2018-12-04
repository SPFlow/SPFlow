"""
Created on August 14, 2018

@author: Alejandro Molina
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import RandomState

from spn.algorithms.LearningWrappers import learn_conditional, learn_parametric
from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import Context
from spn.structure.leaves.conditional.Conditional import Conditional_Poisson, Conditional_Bernoulli
from spn.structure.leaves.conditional.Inference import add_conditional_inference_support
from spn.structure.leaves.conditional.Sampling import add_conditional_sampling_support
import pickle
import os

from spn.structure.leaves.parametric.Parametric import Bernoulli


def create_images_horizontal_lines(rows, cols):
    result = np.zeros((rows, rows, cols))
    for i in range(rows):
        result[i, i, :] = 1

    return result.reshape((rows, -1))


def plot_img(image, rows, cols):
    plt.imshow(image.reshape(rows, cols), cmap="Greys", interpolation="nearest")

    plt.show()


if __name__ == "__main__":
    add_conditional_inference_support()
    add_conditional_sampling_support()

    px = 10
    py = 20

    images = create_images_horizontal_lines(px, py)

    images2d = images.reshape(-1, px, py)
    middle = py // 2
    left = images2d[:, :, :middle].reshape((images.shape[0], -1))
    right = images2d[:, :, middle:].reshape((images.shape[0], -1))

    # format: R|L
    conditional_training_data = np.concatenate((right.reshape(px, -1), left.reshape(px, -1)), axis=1)

    # In left, OUT right
    file_cache_path = "/tmp/cspn.bin"
    if not os.path.isfile(file_cache_path):
        spn_training_data = left.reshape(px, -1)
        spn_training_data = np.repeat(spn_training_data, 10, axis=0)
        ds_context = Context(parametric_types=[Bernoulli] * left.shape[1]).add_domains(spn_training_data)
        spn = learn_parametric(spn_training_data, ds_context, min_instances_slice=1)

        ds_context = Context(parametric_types=[Conditional_Bernoulli] * right.shape[1]).add_domains(right)
        scope = list(range(right.shape[1]))
        cspn = learn_conditional(conditional_training_data, ds_context, scope, min_instances_slice=60000000)
        with open(file_cache_path, "wb") as f:
            pickle.dump((cspn, spn), f, pickle.HIGHEST_PROTOCOL)

    with open(file_cache_path, "rb") as f:
        cspn, spn = pickle.load(f)

    def conditional_input_to_LR(input_images_in_rl):
        # format L|R
        images_to_lr = np.concatenate(
            (
                input_images_in_rl[:, input_images_in_rl.shape[1] // 2 :].reshape(input_images_in_rl.shape[0], px, -1),
                input_images_in_rl[:, : input_images_in_rl.shape[1] // 2].reshape(input_images_in_rl.shape[0], px, -1),
            ),
            axis=2,
        ).reshape(input_images_in_rl.shape[0], -1)
        return images_to_lr

    spn_input = np.zeros_like(right).reshape(px, -1) / 0

    sample_left = sample_instances(spn, spn_input, RandomState(123))

    sample_input = np.concatenate((np.zeros_like(right).reshape(px, -1) / 0, sample_left), axis=1)

    sample_plot = conditional_input_to_LR(sample_input)
    for r in range(sample_plot.shape[0]):
        plot_img(sample_plot[r], px, py)

    sample_images = sample_instances(cspn, sample_input, RandomState(123))

    sample_plot = conditional_input_to_LR(sample_images)
    for r in range(sample_plot.shape[0]):
        plot_img(sample_plot[r], px, py)
