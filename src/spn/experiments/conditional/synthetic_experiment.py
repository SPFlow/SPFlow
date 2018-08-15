'''
Created on August 14, 2018

@author: Alejandro Molina
'''
import numpy as np
import matplotlib.pyplot as plt


def create_images_horizontal_lines(rows, cols):
    result = np.zeros((rows, rows, cols))
    for i in range(rows):
        result[i, i, :] = 1

    return result.reshape((rows, -1))


def plot_img(image, rows, cols):
    plt.imshow(image.reshape(rows, cols), cmap='Greys', interpolation='nearest')

    plt.show()


if __name__ == '__main__':


    images = create_images_horizontal_lines(10, 20)
    plot_img(images[1, :], 10, 20)
