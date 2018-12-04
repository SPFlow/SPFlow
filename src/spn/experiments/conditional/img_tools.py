"""
Created on August 29, 2018

@author: Alejandro Molina
"""
from numpy.random.mtrand import RandomState
from scipy.misc import imresize, imsave

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
    ds = np.concatenate([b.reshape(imgs.shape[0], -1) for b in blocks_imgs[blocks]], axis=1)
    return ds, blocks


def stitch_imgs(imgs=0, img_size=(20, 20), num_blocks=(2, 2), blocks=None):
    block_size = (img_size[0] // num_blocks[0], img_size[1] // num_blocks[1])

    result = np.zeros((imgs, img_size[0], img_size[1]))

    result_idx = np.arange(0, np.prod(num_blocks)).reshape(num_blocks[0], num_blocks[1])

    for block_pos, block_values in blocks.items():
        if type(block_pos) == int:
            block_pos = [block_pos]
        sub_blocks = np.split(block_values, len(block_pos), axis=1)
        for bp, bv in zip(block_pos, sub_blocks):
            bv = bv.reshape(-1, block_size[0], block_size[1])
            idx, idy = np.where(result_idx == bp)
            idx = idx[0] * block_size[0]
            idy = idy[0] * block_size[1]
            result[:, idx : idx + block_size[0], idy : idy + block_size[1]] = bv

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


def save_img(img, path):
    imsave(path, img)


def standardize(imgs):
    return imgs / np.max(imgs)


def add_poisson_noise(imgs, seed=123):
    poisson_noise = RandomState(seed).poisson(lam=1, size=imgs.shape)
    return imgs + poisson_noise


def get_sub_blocks(block, inp=[1, 0], output=[0]):
    sub_blocks = np.split(block, len(inp), axis=1)
    res_blocks = [sub_blocks[inp.index(o)] for o in output]
    result = np.concatenate(res_blocks, axis=1)
    return result


def set_sub_block_nans(block, inp=[1, 0], nans=[0]):
    block_size = block.shape[1] // len(inp)
    for o in nans:
        clear_index = inp.index(o)
        rpos = clear_index * block_size
        block[:, rpos : rpos + block_size] = np.nan
    return block


if __name__ == "__main__":
    images_tr, labels_tr, images_te, labels_te = get_mnist()

    ds = images_tr[[0, 1, 2], :]

    ds = rescale(ds, original_size=(28, 28), new_size=(20, 40))

    imgs = get_imgs(ds, size=(20, 40))
    show_img(imgs[0])

    blocks0, _ = get_blocks(imgs, num_blocks=(2, 2), blocks=[0])
    blocks10, _ = get_blocks(imgs, num_blocks=(2, 2), blocks=[1, 0])
    blocks210, _ = get_blocks(imgs, num_blocks=(2, 2), blocks=[2, 1, 0])
    blocks3210, _ = get_blocks(imgs, num_blocks=(2, 2), blocks=[3, 2, 1, 0])

    block_img = stitch_imgs(
        blocks0.shape[0],
        img_size=(20, 40),
        num_blocks=(2, 2),
        blocks={(3, 2, 1, 0): set_sub_block_nans(blocks3210, inp=[3, 2, 1, 0], nans=[0])}
        # blocks={(0): blocks0,
        #        (1): get_sub_blocks(blocks10, inp=[1, 0], output=[1]),
        #        (2): get_sub_blocks(blocks210, inp=[2, 1, 0], output=[2]),
        #        (3): get_sub_blocks(blocks3210, inp=[3, 2, 1, 0], output=[3])}
    )

    show_img(block_img[0])
