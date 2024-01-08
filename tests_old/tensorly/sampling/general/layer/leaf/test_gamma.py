import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.modules.module import sample
from spflow.structure.general.layer.leaf import GammaLayer
from spflow.modules.node import Gamma
from spflow.torch.structure.general.node.leaf.gamma import updateBackend
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_layer_sampling(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)

    layer = GammaLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        alpha=[0.2, 1.8, 0.2],
        beta=[1.9, 0.4, 1.9],
    )

    nodes = [
        Gamma(Scope([0]), alpha=0.2, beta=1.9),
        Gamma(Scope([1]), alpha=1.8, beta=0.4),
        Gamma(Scope([0]), alpha=0.2, beta=1.9),
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

    expected_mean = tl.tensor([0.2 / 1.9, 1.8 / 0.4])
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
    torch.set_default_dtype(torch.float32)
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = GammaLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        alpha=[0.2, 1.8, 0.2],
        beta=[1.9, 0.4, 1.9],
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
    np_samples = tle.toNumpy(samples)
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
            mean_updated = tle.toNumpy(samples_updated).mean(axis=0)

            # check if model and updated model produce similar samples
            tc.assertTrue(
                np.allclose(
                    mean,
                    mean_updated,
                    atol=0.01,
                    rtol=0.1,
                )
            )


def test_change_dtype(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_dtype(torch.float32)

    layer = GammaLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        alpha=[0.2, 1.8, 0.2],
        beta=[1.9, 0.4, 1.9],
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

    tc.assertTrue(samples.dtype == tl.float32)
    layer.to_dtype(tl.float64)
    samples = sample(
        layer,
        10000,
        sampling_ctx=SamplingContext(
            list(range(10000)),
            [[0, 1] for _ in range(5000)] + [[2, 1] for _ in range(5000, 10000)],
        ),
    )
    tc.assertTrue(samples.dtype == tl.float64)


def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = GammaLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        alpha=[0.2, 1.8, 0.2],
        beta=[1.9, 0.4, 1.9],
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
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer.to_device, cuda)
        return

    tc.assertTrue(samples.device.type == "cpu")
    layer.to_device(cuda)
    samples = sample(
        layer,
        10000,
        sampling_ctx=SamplingContext(
            list(range(10000)),
            [[0, 1] for _ in range(5000)] + [[2, 1] for _ in range(5000, 10000)],
        ),
    )
    tc.assertTrue(samples.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()