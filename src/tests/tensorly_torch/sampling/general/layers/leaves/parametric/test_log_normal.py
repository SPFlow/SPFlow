import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.general.layers.leaves.parametric.general_log_normal import LogNormalLayer
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_log_normal import LogNormal
from spflow.torch.structure.general.nodes.leaves.parametric.log_normal import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_sampling(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)

    layer = LogNormalLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        mean=[0.2, 1.8, 0.2],
        std=[0.01, 0.05, 0.01],
    )

    nodes = [
        LogNormal(Scope([0]), mean=0.2, std=0.01),
        LogNormal(Scope([1]), mean=1.8, std=0.05),
        LogNormal(Scope([0]), mean=0.2, std=0.01),
    ]

    # make sure sampling fron non-overlapping scopes works
    sample(layer, 1, sampling_ctx=SamplingContext([0], [[0, 1]]))
    sample(layer, 1, sampling_ctx=SamplingContext([0], [[2, 1]]))
    # make sure sampling from overlapping scopes does not works
    tc.assertRaises(
        ValueError,
        sample,
        layer,
        1,
        sampling_ctx=SamplingContext([0], [[0, 2]]),
    )
    tc.assertRaises(
        ValueError,
        sample,
        layer,
        1,
        sampling_ctx=SamplingContext([0], [[]]),
    )

    layer_samples = sample(
        layer,
        10000,
        sampling_ctx=SamplingContext(
            list(range(10000)),
            [[0, 1] for _ in range(5000)] + [[2, 1] for _ in range(5000, 10000)],
        ),
    )
    nodes_samples = tl.concatenate(
        [
            tl.concatenate([sample(nodes[0], 5000), sample(nodes[2], 5000)], axis=0),
            sample(nodes[1], 10000)[:, [1]],
        ],
        axis=1,
    )

    expected_mean = tl.exp(tl.tensor([0.2 + (0.01**2) / 2, 1.8 + (0.05**2) / 2]))
    tc.assertTrue(np.allclose(tl.mean(nodes_samples, axis=0), expected_mean, atol=0.01, rtol=0.1))
    tc.assertTrue(
        np.allclose(
            tl.mean(layer_samples, axis=0),
            tl.mean(nodes_samples, axis=0),
            atol=0.01,
            rtol=0.1,
        )
    )

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = LogNormalLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        mean=[0.2, 1.8, 0.2],
        std=[0.01, 0.05, 0.01],
    )

    # get samples
    samples = sample(
        layer,
        10000,
        sampling_ctx=SamplingContext(
            list(range(10000)),
            [[0, 1] for _ in range(5000)] + [[2, 1] for _ in range(5000, 10000)],
        ),
    )
    np_samples = tl_toNumpy(samples)
    mean = np_samples.mean(axis=0)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):

            # for each backend update the model and draw the samples
            layer_updated = updateBackend(layer)
            samples_updated = sample(
                layer_updated,
                10000,
                sampling_ctx=SamplingContext(
                    list(range(10000)),
                    [[0, 1] for _ in range(5000)] + [[2, 1] for _ in range(5000, 10000)],
                ),
            )
            mean_updated = tl_toNumpy(samples_updated).mean(axis=0)

            # check if model and updated model produce similar samples
            tc.assertTrue(
                np.allclose(
                    mean,
                    mean_updated,
                    atol=0.01,
                    rtol=0.1,
                )
            )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
