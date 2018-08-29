'''
Created on August 29, 2018

@author: Alejandro Molina
'''
from scipy.misc import imresize

from spn.data.datasets import get_mnist
import numpy as np


def get_imgs(dataset, size=(20, 20)):
    assert dataset.shape[1] == np.prod(size), "invalid image size for dataset size"
    return dataset.reshape(dataset.shape[0], size[0], size[1])


def get_blocks(imgs, num_blocks=(2, 2), blocks=[0, 1]):
    assert imgs.shape[1] % num_blocks[0] == 0, "invalid image size for num_blocks"
    assert imgs.shape[2] % num_blocks[1] == 0, "invalid image size for num_blocks"

    vsplits = np.split(imgs, num_blocks[0], axis=1)
    splits = [np.split(vs, num_blocks[1], axis=2) for vs in vsplits]
    blocks_imgs = np.concatenate(splits)
    ds = np.concatenate(blocks_imgs[blocks], axis=2)
    return ds.reshape(imgs.shape[0], -1)


def stitch_imgs(img_size=(20, 20), num_blocks=(2, 2)):
    block_size = (img_size[0] / num_blocks[0], img_size[1] / num_blocks[1])

    result = np.zeros(ds.shape[0], img_size[0], img_size[1])

    vsplits = np.split(imgs, num_blocks[0], axis=1)
    splits = [np.split(vs, num_blocks[1], axis=2) for vs in vsplits]
    blocks_imgs = np.concatenate(splits)

    return result


def rescale(ds, original_size=(28, 28), new_size=(20, 20)):
    assert ds.shape[1] == np.prod(original_size), "invalid image size for dataset size"
    assert np.all(np.array(new_size) > 0), "new_size should be positive"

    img_data = np.reshape(ds, (-1, original_size[0], original_size[1]))
    return np.asarray([imresize(image, new_size) for image in img_data], dtype=np.float64).reshape(ds.shape[0], -1)


def show_img(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    images_tr, labels_tr, images_te, labels_te = get_mnist()

    ds = images_tr[[0, 1, 2], :]

    ds = rescale(ds, original_size=(28, 28), new_size=(20, 20))

    imgs = get_imgs(ds, size=(20, 20))

    blocks = get_blocks(imgs, blocks=[0, 2])

    block_img = stitch_imgs(img_size=(20, 20), num_blocks=(2, 2))
    show_img(imgs[0])
    show_img(block_img[0])
