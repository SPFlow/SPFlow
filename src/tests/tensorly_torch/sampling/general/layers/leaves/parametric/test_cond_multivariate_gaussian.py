import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.general.layers.leaves.parametric.general_cond_multivariate_gaussian import CondMultivariateGaussianLayer
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_multivariate_gaussian import CondMultivariateGaussian
from spflow.torch.structure.general.nodes.leaves.parametric.cond_multivariate_gaussian import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_sampling(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)

    layer = CondMultivariateGaussianLayer(
        scope=[Scope([0, 1], [4]), Scope([2, 3], [4]), Scope([0, 1], [4])],
        cond_f=lambda data: {
            "mean": [[0.8, 0.3], [0.2, -0.1], [0.8, 0.3]],
            "cov": [
                [[0.13, 0.08], [0.08, 0.05]],
                [[0.17, 0.054], [0.054, 0.0296]],
                [[0.13, 0.08], [0.08, 0.05]],
            ],
        },
    )

    nodes = [
        CondMultivariateGaussian(
            Scope([0, 1], [4]),
            cond_f=lambda data: {
                "mean": [0.8, 0.3],
                "cov": [[0.13, 0.08], [0.08, 0.05]],
            },
        ),
        CondMultivariateGaussian(
            Scope([2, 3], [4]),
            cond_f=lambda data: {
                "mean": [0.2, -0.1],
                "cov": [[0.17, 0.054], [0.054, 0.0296]],
            },
        ),
        CondMultivariateGaussian(
            Scope([0, 1], [4]),
            cond_f=lambda data: {
                "mean": [0.8, 0.3],
                "cov": [[0.13, 0.08], [0.08, 0.05]],
            },
        ),
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
            sample(nodes[1], 10000)[:, [2, 3]],
        ],
        axis=1,
    )

    expected_mean = tl.tensor([0.8, 0.3, 0.2, -0.1])
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

    layer = CondMultivariateGaussianLayer(
        scope=[Scope([0, 1], [4]), Scope([2, 3], [4]), Scope([0, 1], [4])],
        cond_f=lambda data: {
            "mean": [[0.8, 0.3], [0.2, -0.1], [0.8, 0.3]],
            "cov": [
                [[0.13, 0.08], [0.08, 0.05]],
                [[0.17, 0.054], [0.054, 0.0296]],
                [[0.13, 0.08], [0.08, 0.05]],
            ],
        },
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
