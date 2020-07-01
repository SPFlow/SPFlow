#!/usr/bin/env python3

import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal as TorchNormal
from torch.nn import functional as F

from spn.algorithms.layerwise import layers, distributions
from spn.algorithms.layerwise.type_checks import (
    check_valid,
    OutOfBoundsException,
    InvalidTypeException,
    InvalidStackedSpnConfigurationException,
)
from spn.algorithms.layerwise.utils import SamplingContext, provide_evidence


class TestLayerwiseImplementation(unittest.TestCase):
    """Testcases taht ensure, that inference methods for Sum, Product and Leaf layers are working as expected."""

    def test_sum_layer(self):
        """Test the forward pass of a sum layer"""

        # Setup layer
        in_channels = 8
        out_channels = 7
        in_features = 3
        num_repetitions = 5
        sum_layer = layers.Sum(
            in_channels=in_channels, out_channels=out_channels, in_features=in_features, num_repetitions=num_repetitions
        )

        w = torch.rand(in_features, in_channels, out_channels, num_repetitions)

        # Set the sum layer parameters
        sum_layer.weights = nn.Parameter(w)

        # Apply softmax once again since Sum forward pass uses F.log_softmax internally to project random weights
        # back into valid ranges
        w = F.softmax(w, dim=1)

        # Setup test input
        batch_size = 16
        x = torch.rand(size=(batch_size, in_features, in_channels, num_repetitions))

        # Expected outcome
        expected_result = torch.zeros(batch_size, in_features, out_channels, num_repetitions)
        for n in range(batch_size):
            for d in range(in_features):
                for oc in range(out_channels):
                    for r in range(num_repetitions):
                        expected_result[n, d, oc, r] = x[n, d, :, r] @ w[d, :, oc, r]

        # Do forward pass: apply log as sum layer operates in log space. Exp() afterwards to make it comparable to the
        # expected result
        result = sum_layer(x.log()).exp()

        # Run assertions
        self.assertTrue(result.shape[0] == batch_size)
        self.assertTrue(result.shape[1] == in_features)
        self.assertTrue(result.shape[2] == out_channels)
        self.assertTrue(result.shape[3] == num_repetitions)
        self.assertTrue(((result - expected_result).abs() < 1e-6).all())

    def test_product_layer(self):
        """Test the product layer forward pass."""

        # Setup product layer
        in_features = 9
        cardinality = 3
        num_repetitions = 5
        prod_layer = layers.Product(in_features=in_features, cardinality=cardinality, num_repetitions=num_repetitions)

        # Setup test input
        batch_size = 16
        in_channels = 3
        x = torch.rand(size=(batch_size, in_features, in_channels, num_repetitions))

        # Expected result:
        expected_result = torch.ones(batch_size, in_features // cardinality, in_channels, num_repetitions)
        for n in range(batch_size):
            for d in range(0, in_features, cardinality):
                for c in range(in_channels):
                    for r in range(num_repetitions):
                        for i in range(cardinality):
                            expected_result[n, d // cardinality, c, r] *= x[n, d + i, c, r]

        # Actual result
        result = prod_layer(x.log()).exp()

        # Run assertions
        self.assertTrue(result.shape[0] == batch_size)
        self.assertTrue(result.shape[1] == in_features // cardinality)
        self.assertTrue(result.shape[2] == in_channels)
        self.assertTrue(result.shape[3] == num_repetitions)
        self.assertTrue(((result - expected_result).abs() < 1e-6).all())

    def test_normal_leaf_layer(self):
        """Test the normal leaf layer."""
        # Setup leaf layer
        out_channels = 7
        in_features = 8
        num_repetitions = 5
        leaf = distributions.Normal(out_channels=out_channels, in_features=in_features, num_repetitions=num_repetitions)

        # Setup test input
        batch_size = 3
        x = torch.rand(size=(batch_size, in_features))

        # Setup artificial means and scale matrices
        means = torch.randn(1, in_features, out_channels, num_repetitions)
        scale = torch.rand(1, in_features, out_channels, num_repetitions)

        # Use scipy norm to get pdfs
        # Expected result
        expected_result = torch.zeros(batch_size, in_features, out_channels, num_repetitions)

        # Repetition 1
        for n in range(batch_size):
            for d in range(in_features):
                for c in range(out_channels):
                    for r in range(num_repetitions):
                        expected_result[n, d, c, r] = TorchNormal(
                            loc=means[0, d, c, r], scale=scale[0, d, c, r]
                        ).log_prob(x[n, d])

        # Perform forward pass in leaf
        leaf.means.data = means
        leaf.stds.data = scale
        result = leaf(x)

        # Make assertions
        self.assertEqual(result.shape[0], batch_size)
        self.assertEqual(result.shape[1], in_features)
        self.assertEqual(result.shape[2], out_channels)
        self.assertTrue(((result - expected_result).abs() < 1e-6).all())


class TestLayerwiseSampling(unittest.TestCase):
    """Testcases that ensure that sampling methods for Sum, Product and Leaf layers are working as expected."""

    def test_sum_shape_as_root_node(self):
        """Check that the sum node has the correct sampling shape when used as root."""
        n = 5
        num_repetitions = 1
        for in_channels in [1, 5, 10]:
            for in_features in [1, 5, 10]:
                sum_layer = layers.Sum(
                    in_channels=in_channels, out_channels=1, in_features=in_features, num_repetitions=num_repetitions
                )
                ctx = SamplingContext(n=n)
                ctx = sum_layer.sample(context=ctx)
                self.assertTrue(ctx.parent_indices.shape[0] == n)
                self.assertTrue(ctx.parent_indices.shape[1] == in_features)

    def test_product_shape_as_root_node(self):
        """Check that the product node has the correct sampling shape when used as root."""
        prod_layer = layers.Product(in_features=10, cardinality=2, num_repetitions=1)
        ctx = SamplingContext(n=5)
        ctx = prod_layer.sample(context=ctx)
        self.assertTrue(ctx.parent_indices.shape[0] == 5)
        self.assertTrue(ctx.parent_indices.shape[1] == 1)

    def test_sum_as_intermediate_node(self):
        """Check that sum node returns the correct sample indices when used as indermediate node."""
        # Some values for the sum layer
        in_features = 10
        in_channels = 3
        out_channels = 5
        num_repetitions = 7
        n = 2
        parent_indices = torch.randint(out_channels, size=(n, in_features))

        # Create sum layer
        sum_layer = layers.Sum(
            in_features=in_features, in_channels=in_channels, out_channels=out_channels, num_repetitions=num_repetitions
        )

        # Choose `in_features` number of random indexes from 0 to in_channels-1 which will have probability of 1.0 in
        # the sum layer weight tensor
        rand_indxs = torch.randint(in_channels, size=(in_features, num_repetitions))
        rep_idxs = torch.randint(num_repetitions, size=(n,))

        # Artificially set sum weights (probabilities) to 1.0
        weights = torch.zeros(in_features, in_channels, out_channels, num_repetitions)
        for r in range(num_repetitions):
            weights[range(in_features), rand_indxs[:, r], :, r] = 1.0
        sum_layer.weights = nn.Parameter(torch.log(weights))

        # Perform sampling
        ctx = SamplingContext(n=n, parent_indices=parent_indices, repetition_indices=rep_idxs)
        sum_layer.sample(context=ctx)

        # Assert that the sample indexes are those where the weights were set to 1.0
        for i in range(n):
            self.assertTrue((rand_indxs[:, rep_idxs[i]] == ctx.parent_indices[i, :]).all())

    def test_prod_as_intermediate_node(self):
        # Product layer values
        in_features = 10
        num_samples = 5
        num_repetitions = 5
        for cardinality in range(2, in_features):
            prod_layer = layers.Product(
                in_features=in_features, cardinality=cardinality, num_repetitions=num_repetitions
            )

            # Example parent indexes
            parent_indices = torch.randint(high=5, size=(num_samples, in_features))

            # Create expected indexes: each index is repeated #cardinality times
            pad = (cardinality - in_features % cardinality) % cardinality
            expected_sample_indices = []
            for j in range(num_samples):

                sample_i_indices = []
                for i in parent_indices[j, :]:
                    sample_i_indices += [i] * cardinality

                # Remove padding
                if pad > 0:
                    sample_i_indices = sample_i_indices[:-pad]

                # Add current sample
                expected_sample_indices.append(sample_i_indices)

            # As tensor
            expected_sample_indices = torch.tensor(expected_sample_indices)

            # Sample
            ctx = SamplingContext(n=num_samples, parent_indices=parent_indices)
            prod_layer.sample(context=ctx)
            self.assertTrue((expected_sample_indices == ctx.parent_indices).all())

    def test_normal_leaf(self):
        # Setup leaf layer
        out_channels = 10
        in_features = 10
        num_repetitions = 5
        leaf = distributions.Normal(out_channels=out_channels, in_features=in_features, num_repetitions=num_repetitions)

        # Set leaf layer mean to some random int
        leaf.means.data = torch.randint(
            low=-100, high=100, size=(1, in_features, out_channels, num_repetitions)
        ).float()
        # Set leaf layer std to 0 such that the samples will all be the mean (so we can actually make assertions in the end)
        leaf.stds.data = torch.zeros(size=(1, in_features, out_channels, num_repetitions)).float()

        # Create some random indices into the out_channels axis
        parent_indices = torch.randint(high=out_channels, size=(1, in_features,))
        repetition_indices = torch.randint(high=num_repetitions, size=(1,))

        # Perform sampling
        ctx = SamplingContext(n=1, parent_indices=parent_indices, repetition_indices=repetition_indices)
        result = leaf.sample(context=ctx)

        # Expected sampling
        expected_result = leaf.means.data[:, range(in_features), parent_indices, repetition_indices[0]]

        # Run assertions
        self.assertTrue(((result - expected_result).abs() < 1e-6).all())

    def test_spn_sampling(self):

        # Define SPN
        leaf = distributions.Normal(in_features=2 ** 3, out_channels=5, num_repetitions=1)
        sum_1 = layers.Sum(in_channels=5, in_features=2 ** 3, out_channels=20, num_repetitions=1)
        prd_1 = layers.Product(in_features=2 ** 3, cardinality=2, num_repetitions=1)
        sum_2 = layers.Sum(in_channels=20, in_features=2 ** 2, out_channels=20, num_repetitions=1)
        prd_2 = layers.Product(in_features=2 ** 2, cardinality=2, num_repetitions=1)
        sum_3 = layers.Sum(in_channels=20, in_features=2 ** 1, out_channels=20, num_repetitions=1)
        prd_3 = layers.Product(in_features=2 ** 1, cardinality=2, num_repetitions=1)
        sum_4 = layers.Sum(in_channels=20, in_features=2 ** 0, out_channels=1, num_repetitions=1)

        # Test forward pass
        x_test = torch.randn(1, 2 ** 3)

        x_test = leaf(x_test)
        x_test = sum_1(x_test)
        x_test = prd_1(x_test)
        x_test = sum_2(x_test)
        x_test = prd_2(x_test)
        x_test = sum_3(x_test)
        x_test = prd_3(x_test)
        res = sum_4(x_test)

        # Sampling pass
        ctx = SamplingContext(n=1000)
        sum_4.sample(context=ctx)
        prd_3.sample(context=ctx)
        sum_3.sample(context=ctx)
        prd_2.sample(context=ctx)
        sum_2.sample(context=ctx)
        prd_1.sample(context=ctx)
        sum_1.sample(context=ctx)
        samples = leaf.sample(context=ctx)

    def test_spn_mpe(self):

        # Define SPN
        leaf = distributions.Normal(in_features=2 ** 3, out_channels=5, num_repetitions=1)
        sum_1 = layers.Sum(in_channels=5, in_features=2 ** 3, out_channels=20, num_repetitions=1)
        prd_1 = layers.Product(in_features=2 ** 3, cardinality=2, num_repetitions=1)
        sum_2 = layers.Sum(in_channels=20, in_features=2 ** 2, out_channels=20, num_repetitions=1)
        prd_2 = layers.Product(in_features=2 ** 2, cardinality=2, num_repetitions=1)
        sum_3 = layers.Sum(in_channels=20, in_features=2 ** 1, out_channels=20, num_repetitions=1)
        prd_3 = layers.Product(in_features=2 ** 1, cardinality=2, num_repetitions=1)
        sum_4 = layers.Sum(in_channels=20, in_features=2 ** 0, out_channels=1, num_repetitions=1)

        sum_1._enable_input_cache()
        sum_2._enable_input_cache()
        sum_3._enable_input_cache()
        sum_4._enable_input_cache()

        # Test forward pass
        x_test = torch.randn(1, 2 ** 3)

        x_test = leaf(x_test)
        x_test = sum_1(x_test)
        x_test = prd_1(x_test)
        x_test = sum_2(x_test)
        x_test = prd_2(x_test)
        x_test = sum_3(x_test)
        x_test = prd_3(x_test)
        res = sum_4(x_test)

        ctx = SamplingContext(n=x_test.shape[0], is_mpe=True)
        sum_4.sample(context=ctx)
        prd_3.sample(context=ctx)
        sum_3.sample(context=ctx)
        prd_2.sample(context=ctx)
        sum_2.sample(context=ctx)
        prd_1.sample(context=ctx)
        sum_1.sample(context=ctx)

        # Should be the same
        mpe_1 = leaf.sample(context=ctx)
        mpe_2 = leaf.sample(context=ctx)
        mpe_3 = leaf.sample(context=ctx)
        self.assertTrue(((mpe_1 - mpe_2).abs() < 1e-6).all())
        self.assertTrue(((mpe_2 - mpe_3).abs() < 1e-6).all())


class TestTypeChecks(unittest.TestCase):
    def test_valid(self):
        # Ints
        check_valid(0, int, 0)
        check_valid(np.int64(0), int, 0)
        check_valid(np.int32(0), int, 0)
        check_valid(np.int16(0), int, 0)
        check_valid(np.int8(0), int, 0)
        check_valid(torch.tensor(0).int(), int, 0)
        check_valid(torch.tensor(0).long(), int, 0)

        # Floats
        check_valid(1.0, float, 0)
        check_valid(np.float64(1.0), float, 0)
        check_valid(np.float32(1.0), float, 0)
        check_valid(np.float16(1.0), float, 0)
        check_valid(torch.tensor(1.0).half(), float, 0)
        check_valid(torch.tensor(1.0).float(), float, 0)
        check_valid(torch.tensor(1.0).double(), float, 0)

    def test_invalid_range(self):
        with self.assertRaises(OutOfBoundsException):
            check_valid(0, int, 1, 2)

        with self.assertRaises(OutOfBoundsException):
            check_valid(0.0, float, 1.0, 2.0)

        with self.assertRaises(OutOfBoundsException):
            check_valid(2, int, 0, 1)

    def test_invalid_type(self):
        with self.assertRaises(InvalidTypeException):
            check_valid(0, float, 0, 1)

        with self.assertRaises(InvalidTypeException):
            check_valid(0.0, int, 0, 1)

        with self.assertRaises(InvalidTypeException):
            check_valid(np.int64(0), float, 0, 1)

        with self.assertRaises(InvalidTypeException):
            check_valid(torch.tensor(0).int(), float, 0, 1)


class TestRATLayerwise(unittest.TestCase):
    def test_rat_forward(self):
        from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn
        from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpnConfig
        from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal

        # Setup RatSpn
        config = RatSpnConfig()
        config.F = 16
        config.R = 13
        config.D = 3
        config.C = 2
        config.I = 11
        config.S = 12
        config.dropout = 0.0
        config.leaf_base_class = RatNormal

        spn = RatSpn(config)

        # Generate data
        batch_size = 32
        x = torch.randn(batch_size, config.F)

        # Forward pass
        result = spn(x)

        # Make assertions on the shape
        self.assertEqual(result.shape[0], batch_size)
        self.assertEqual(result.shape[1], config.C)

    def test_rat_sampling(self):
        from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn
        from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpnConfig
        from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal

        # Setup RatSpn
        config = RatSpnConfig()
        config.F = 16
        config.R = 13
        config.D = 3
        config.C = 2
        config.I = 11
        config.S = 12
        config.dropout = 0.0
        config.leaf_base_class = RatNormal
        spn = RatSpn(config)

        # Sample
        n = 10
        samples = spn.sample(n=n)
        self.assertTrue(samples.shape[0] == n)
        self.assertTrue(samples.shape[1] == config.F)

        # Conditional sampling
        x = torch.randn(n, config.F)
        x[:, 0 : config.F // 2] = float("nan")
        spn.sample(evidence=x)

    def test_rat_mpe(self):
        from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn
        from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpnConfig
        from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal

        # Setup RatSpn
        config = RatSpnConfig()
        config.F = 16
        config.R = 13
        config.D = 3
        config.C = 2
        config.I = 11
        config.S = 12
        config.dropout = 0.0
        config.leaf_base_class = RatNormal
        spn = RatSpn(config)

        # Conditional MPE
        x = torch.randn(10, config.F)
        x[:, 0 : config.F // 2] = float("nan")
        mpe_1 = spn.mpe(evidence=x)
        mpe_2 = spn.mpe(evidence=x)
        mpe_3 = spn.mpe(evidence=x)
        self.assertTrue(((mpe_1 - mpe_2).abs() < 1e-6).all())
        self.assertTrue(((mpe_2 - mpe_3).abs() < 1e-6).all())



if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    unittest.main()
