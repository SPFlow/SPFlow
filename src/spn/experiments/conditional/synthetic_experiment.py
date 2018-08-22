'''
Created on August 14, 2018

@author: Alejandro Molina
'''
import numpy as np
import matplotlib.pyplot as plt

from spn.algorithms.LearningWrappers import learn_conditional
from spn.structure.Base import Context
from spn.structure.leaves.conditional.Conditional import Conditional_Poisson


def create_images_horizontal_lines(rows, cols):
    result = np.zeros((rows, rows, cols))
    for i in range(rows):
        result[i, i, :] = 1

    return result.reshape((rows, -1))


def plot_img(image, rows, cols):
    plt.imshow(image.reshape(rows, cols), cmap='Greys', interpolation='nearest')

    plt.show()


if __name__ == '__main__':
    px = 100
    py = 20

    images = create_images_horizontal_lines(px, py)
    images2d = images.reshape(-1, px, py)
    middle = py // 2
    left = images2d[:, :, :middle].reshape((images.shape[0], -1))
    right = images2d[:, :, middle:].reshape((images.shape[0], -1))


    ds_context = Context(parametric_types=[Conditional_Poisson] * right.shape[1]).add_domains(right)

    scope = list(range(right.shape[1]))

    #In left, OUT right
    cspn = learn_conditional(np.concatenate((right, left), axis=1), ds_context, scope, min_instances_slice=60000000)

    plot_img(images[1, :], 10, 20)
