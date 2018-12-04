import logging

import numpy as np
from joblib import Memory

from spn.algorithms.MPE import mpe
from spn.experiments.conditional.img_tools import get_blocks, set_sub_block_nans, stitch_imgs, save_img
from spn.structure.leaves.conditional.MPE import add_conditional_mpe_support

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
import os

add_conditional_inference_support()
add_conditional_sampling_support()
add_conditional_mpe_support()

memory = Memory(location="/tmp/cspn_face_cache", verbose=10, compress=9)

from sklearn.datasets import fetch_olivetti_faces


def one_hot(y):
    if len(y.shape) != 1:
        return y
    values = np.array(sorted(list(set(y))))
    return np.array([values == v for v in y], dtype=int)


if __name__ == "__main__":

    output_path = os.path.dirname(os.path.abspath(__file__)) + "/imgs_pixelcspn_faces/"

    logging.basicConfig(level=logging.DEBUG)
    logging.captureWarnings(True)

    faces = fetch_olivetti_faces()

    images = faces["images"]
    print("faces loaded", images.shape)

    # Split Images Vertically in 16 blocks
    img_size = (64, 64)
    num_v_blocks = 16
    num_blocks = (num_v_blocks, 1)

    datasets = []
    for i in range(num_v_blocks):
        block_ids = np.arange(i, -1, -1)
        datasets.append((get_blocks(images, num_blocks=num_blocks, blocks=block_ids.tolist()), 1))

    num_mpes = 1
    num_samples = 10

    cspns = []
    mpe_query_blocks = None
    sample_query_blocks = None
    for i, ((tr_block, block_idx), conditional_blocks) in enumerate(datasets):
        print("learning", i)
        conditional_features_count = (tr_block.shape[1] // len(block_idx)) * conditional_blocks
        if i == 0:
            # spn
            ds_context = Context(meta_types=[MetaType.REAL] * tr_block.shape[1])
            ds_context.add_domains(tr_block)
            ds_context.parametric_types = [Gaussian] * tr_block.shape[1]

            cspn = learn_parametric(tr_block, ds_context, min_instances_slice=20, ohe=False, memory=memory)
        else:
            cspn = learn_conditional(
                tr_block,
                Context(
                    meta_types=[MetaType.REAL] * tr_block.shape[1],
                    parametric_types=[Conditional_Gaussian] * tr_block.shape[1],
                ).add_domains(tr_block),
                scope=list(range(conditional_features_count)),
                min_instances_slice=30,
                memory=memory,
            )
        cspns.append(cspn)
        print("done")

        # for i, ((tr_block, block_idx), conditional_blocks) in enumerate(datasets):
        #    cspn = cspns[i]
        if i == 0:
            # first time, we only care about the structure to put nans
            mpe_query_blocks = np.zeros_like(tr_block[0:num_mpes, :].reshape(num_mpes, -1))
            sample_query_blocks = np.zeros_like(tr_block[0:num_samples, :].reshape(num_samples, -1))
        else:
            # i+1 time: we set the previous mpe values as evidence
            mpe_query_blocks = np.zeros_like(np.array(tr_block[0:num_mpes, :].reshape(num_mpes, -1)))
            mpe_query_blocks[:, -(mpe_result.shape[1]) :] = mpe_result

            sample_query_blocks = np.zeros_like(np.array(tr_block[0:num_samples, :].reshape(num_samples, -1)))
            sample_query_blocks[:, -(sample_result.shape[1]) :] = sample_result

        cspn_mpe_query = set_sub_block_nans(mpe_query_blocks, inp=block_idx, nans=block_idx[0:conditional_blocks])
        mpe_result = mpe(cspn, cspn_mpe_query)

        mpe_img_blocks = stitch_imgs(
            mpe_result.shape[0], img_size=img_size, num_blocks=num_blocks, blocks={tuple(block_idx): mpe_result}
        )

        cspn_sample_query = set_sub_block_nans(sample_query_blocks, inp=block_idx, nans=block_idx[0:conditional_blocks])
        sample_result = sample_instances(cspn, cspn_sample_query, RandomState(123))

        sample_img_blocks = stitch_imgs(
            sample_result.shape[0], img_size=img_size, num_blocks=num_blocks, blocks={tuple(block_idx): sample_result}
        )

        for j in range(num_mpes):
            mpe_fname = output_path + "mpe_%s_%s.png" % ("-".join(map(str, block_idx)), j)
            save_img(mpe_img_blocks[j], mpe_fname)

        for j in range(num_samples):
            sample_fname = output_path + "sample_%s_%s.png" % ("-".join(map(str, block_idx)), j)
            save_img(sample_img_blocks[j], sample_fname)
