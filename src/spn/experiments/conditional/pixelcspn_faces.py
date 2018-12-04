from joblib import Memory

from spn.algorithms.MPE import mpe
from spn.experiments.conditional.img_tools import (
    rescale,
    standardize,
    add_poisson_noise,
    get_blocks,
    get_imgs,
    set_sub_block_nans,
    stitch_imgs,
    show_img,
    save_img,
)
from spn.structure.leaves.conditional.MPE import add_conditional_mpe_support

from spn.data.datasets import get_mnist
import numpy as np

np.random.seed(42)

from spn.algorithms.LearningWrappers import learn_conditional, learn_parametric
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.conditional.Inference import add_conditional_inference_support
from spn.structure.leaves.conditional.Conditional import Conditional_Gaussian
from spn.structure.leaves.parametric.Parametric import Gaussian
from numpy.random.mtrand import RandomState
from spn.algorithms.Sampling import sample_instances
from spn.structure.leaves.conditional.Sampling import add_conditional_sampling_support

add_conditional_inference_support()
add_conditional_sampling_support()
add_conditional_mpe_support()

memory = Memory(cachedir="/tmp/cspn_face_cache", verbose=0, compress=9)


from sklearn.datasets import fetch_olivetti_faces


def one_hot(y):
    if len(y.shape) != 1:
        return y
    values = np.array(sorted(list(set(y))))
    return np.array([values == v for v in y], dtype=int)


if __name__ == "__main__":

    faces = fetch_olivetti_faces()

    print("faces loaded")
    images = faces["images"]
    # images = images * 256
    # images = add_poisson_noise(images)
    # images = standardize(images)

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
        get_blocks(images, num_blocks=(2, 2), blocks=[3, 2, 1, 0]),
    ]
    cspns = []
    mpe_query_blocks = None
    sample_query_blocks = None
    for i, (tr_block, block_idx) in enumerate(datasets):
        if i == 0:
            # spn
            ds_context = Context(meta_types=[MetaType.REAL] * tr_block.shape[1])
            ds_context.add_domains(tr_block)
            ds_context.parametric_types = [Gaussian] * tr_block.shape[1]

            cspn = learn_parametric(tr_block, ds_context, min_instances_slice=2 * len(tr_block), ohe=False)

        else:
            cspn = learn_conditional(
                tr_block,
                Context(
                    meta_types=[MetaType.REAL] * tr_block.shape[1],
                    parametric_types=[Conditional_Gaussian] * tr_block.shape[1],
                ).add_domains(tr_block),
                scope=list(range(datasets[0][0].shape[1])),
                min_instances_slice=0.5 * tr_block.shape[0],
                memory=memory,
            )
        cspns.append(cspn)

        if mpe_query_blocks is None:
            # first time, we only care about the structure to put nans
            mpe_query_blocks = np.zeros_like(tr_block[0:10, :].reshape(10, -1))
            sample_query_blocks = np.zeros_like(tr_block[0:10, :].reshape(10, -1))
        else:
            # i+1 time: we set the previous mpe values as evidence
            mpe_query_blocks = np.zeros_like(np.array(tr_block[0:10, :].reshape(10, -1)))
            mpe_query_blocks[:, -(mpe_result.shape[1]) :] = mpe_result

            sample_query_blocks = np.zeros_like(np.array(tr_block[0:10, :].reshape(10, -1)))
            sample_query_blocks[:, -(sample_query_blocks.shape[1]) :] = sample_result

        cspn_mpe_query = set_sub_block_nans(mpe_query_blocks, inp=block_idx, nans=[block_idx[0]])
        mpe_result = mpe(cspn, cspn_mpe_query)

        mpe_img_blocks = stitch_imgs(
            mpe_result.shape[0], img_size=(64, 64), num_blocks=(2, 2), blocks={tuple(block_idx): mpe_result}
        )

        cspn_sample_query = set_sub_block_nans(sample_query_blocks, inp=block_idx, nans=[block_idx[0]])
        sample_result = sample_instances(cspn, cspn_sample_query, RandomState(123))

        sample_img_blocks = stitch_imgs(
            mpe_result.shape[0], img_size=(64, 64), num_blocks=(2, 2), blocks={tuple(block_idx): sample_result}
        )

        for c in range(10):
            mpe_fname = (
                "/home/shao/simple_spn/simple_spn/src/spn/experiments/conditional/faces_pixelcspn_annotations/mpe_cspn_%s_class_%s.png"
                % ("".join(map(str, block_idx)), c)
            )
            save_img(mpe_img_blocks[c], mpe_fname)

            sample_fname = (
                "/home/shao/simple_spn/simple_spn/src/spn/experiments/conditional/faces_pixelcspn_annotations/sample_cspn_%s_class_%s.png"
                % ("".join(map(str, block_idx)), c)
            )
            save_img(sample_img_blocks[c], sample_fname)
