from joblib import Memory

from spn.algorithms.MPE import mpe
from spn.experiments.conditional.img_tools import rescale, standardize, add_poisson_noise, get_blocks, get_imgs, \
    set_sub_block_nans, stitch_imgs, show_img, save_img
from spn.structure.leaves.conditional.MPE import add_conditional_mpe_support

from spn.data.datasets import get_mnist
import numpy as np

np.random.seed(42)

from spn.algorithms.LearningWrappers import learn_conditional
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.conditional.Inference import add_conditional_inference_support
from spn.structure.leaves.conditional.Conditional import Conditional_Gaussian
from numpy.random.mtrand import RandomState
from spn.algorithms.Sampling import sample_instances
from spn.structure.leaves.conditional.Sampling import add_conditional_sampling_support

add_conditional_inference_support()
add_conditional_sampling_support()
add_conditional_mpe_support()

memory = Memory(cachedir="/tmp/cspn_cache", verbose=0, compress=9)


def one_hot(y):
    if len(y.shape) != 1:
        return y
    values = np.array(sorted(list(set(y))))
    return np.array([values == v for v in y], dtype=int)


if __name__ == '__main__':

    images_tr, labels_tr, images_te, labels_te = get_mnist()
    data_tr = []
    data_labels_tr = []
    for i in range(10):
        data_tr.append(images_tr[labels_tr == i][:2000])
        data_labels_tr.append(labels_tr[labels_tr == i][:2000])
    data_tr = np.concatenate(data_tr, axis=0)
    data_labels_tr = np.concatenate(data_labels_tr, axis=0)

    print("mnist loaded")
    images = rescale(data_tr, original_size=(28, 28), new_size=(20, 20))
    images = get_imgs(images, size=(20, 20))
    images = standardize(images)
    images = add_poisson_noise(images)

    data_labels_tr = one_hot(data_labels_tr)

    # Learn cspns for image blocks like this:
    #   |0|1|
    #   |2|3|
    # P0(0|labels)
    # P1(1|0,labels)
    # P2(2|1,0,labels)
    # P3(3|2,1,0,labels)

    datasets = [
        # block of  0
        get_blocks(images, num_blocks=(2, 2), blocks=[0]),
        # block of  1|0
        get_blocks(images, num_blocks=(2, 2), blocks=[1, 0]),
        # block of  2|1,0
        get_blocks(images, num_blocks=(2, 2), blocks=[2, 1, 0]),
        # block of  3|2,1,0
        get_blocks(images, num_blocks=(2, 2), blocks=[3, 2, 1, 0])
    ]
    cspns = []
    for i, (tr_block, block_idx) in enumerate(datasets):
        cspn = learn_conditional(
            np.concatenate((tr_block, data_labels_tr), axis=1),
            Context(meta_types=[MetaType.REAL] * tr_block.shape[1],
                    parametric_types=[Conditional_Gaussian] * tr_block.shape[1]).add_domains(tr_block),
            scope=list(range(tr_block.shape[1])),
            min_instances_slice=3 * tr_block.shape[0], memory=memory)
        cspns.append(cspn)

        # plot MPEs
        cspn_query = np.concatenate((set_sub_block_nans(tr_block[0:10, :].reshape(10, -1), inp=block_idx, nans=[i]),
                                     np.eye(10, 10)),
                                    axis=1)
        mpe_result = mpe(cspn, cspn_query)

        img_blocks = stitch_imgs(mpe_result.shape[0], img_size=(20, 20), num_blocks=(2, 2),
                                 blocks={tuple(block_idx): mpe_result[:, 0:-10]})
        for c in range(10):
            save_img(img_blocks[c],
                     "imgs_pixelcspn_annotations/mpe_cspn_%s_class_%s.png" % ("".join(map(str, block_idx)), c))
