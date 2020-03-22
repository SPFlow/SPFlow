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


class TestLayerwiseImplementation(unittest.TestCase):
    """Testcases taht ensure, that inference methods for Sum, Product and Leaf layers are working as expected."""

    def test_sum_layer(self):
        """Test the forward pass of a sum layer"""

        # Setup layer
        in_channels = 2
        out_channels = 2
        in_features = 2
        sum_layer = layers.Sum(in_channels=in_channels, out_channels=out_channels, in_features=in_features)

        w = torch.zeros(in_features, in_channels, out_channels)

        # Weights feat: 0, out_channel: 0
        w[0, 0, 0] = 0.3
        w[0, 1, 0] = 0.7
        self.assertTrue(w[0, :, 0].sum() == 1.0)

        # Weights feat: 1, out_channel: 0
        w[1, 0, 0] = 0.8
        w[1, 1, 0] = 0.2
        self.assertTrue(w[1, :, 0].sum() == 1.0)

        # Weights feat: 0, out_channel: 1
        w[0, 0, 1] = 0.25
        w[0, 1, 1] = 0.75
        self.assertTrue(w[0, :, 1].sum() == 1.0)

        # Weights feat: 1, out_channel: 1
        w[1, 0, 1] = 0.3
        w[1, 1, 1] = 0.7
        self.assertTrue(w[1, :, 1].sum() == 1.0)

        # Set the sum layer parameters
        sum_layer.sum_weights = nn.Parameter(w)

        # Apply softmax once again since Sum forward pass uses F.log_softmax internally to project random weights
        # back into valid ranges
        w = F.softmax(w, dim=1)

        # Setup test input
        batch_size = 2
        x = torch.rand(size=(batch_size, in_features, in_channels))

        # Expected outcome
        expected_result = torch.zeros(batch_size, in_features, out_channels)

        expected_result[0, 0, 0] = x[0, 0, 0] * w[0, 0, 0] + x[0, 0, 1] * w[0, 1, 0]
        expected_result[0, 1, 0] = x[0, 1, 0] * w[1, 0, 0] + x[0, 1, 1] * w[1, 1, 0]

        expected_result[0, 0, 1] = x[0, 0, 0] * w[0, 0, 1] + x[0, 0, 1] * w[0, 1, 1]
        expected_result[0, 1, 1] = x[0, 1, 0] * w[1, 0, 1] + x[0, 1, 1] * w[1, 1, 1]

        expected_result[1, 0, 0] = x[1, 0, 0] * w[0, 0, 0] + x[1, 0, 1] * w[0, 1, 0]
        expected_result[1, 1, 0] = x[1, 1, 0] * w[1, 0, 0] + x[1, 1, 1] * w[1, 1, 0]

        expected_result[1, 0, 1] = x[1, 0, 0] * w[0, 0, 1] + x[1, 0, 1] * w[0, 1, 1]
        expected_result[1, 1, 1] = x[1, 1, 0] * w[1, 0, 1] + x[1, 1, 1] * w[1, 1, 1]

        # Do forward pass: apply log as sum layer operates in log space. Exp() afterwards to make it comparable to the
        # expected result
        result = sum_layer(x.log()).exp()

        # Run assertions
        self.assertTrue(result.shape[0] == batch_size)
        self.assertTrue(result.shape[1] == in_features)
        self.assertTrue(result.shape[2] == out_channels)
        self.assertTrue(((result - expected_result).abs() < 1e-6).all())

    def test_product_layer(self):
        """Test the product layer forward pass."""

        # Setup product layer
        in_features = 4
        cardinality = 2
        prod_layer = layers.Product(in_features=in_features, cardinality=cardinality)

        # Setup test input
        batch_size = 2
        in_channels = 2
        x = torch.rand(size=(batch_size, in_features, in_channels))

        # Expected result:
        expected_result = torch.zeros(batch_size, in_features // 2, in_channels)
        expected_result[0, 0, 0] = x[0, 0, 0] * x[0, 1, 0]
        expected_result[0, 1, 0] = x[0, 2, 0] * x[0, 3, 0]
        expected_result[0, 0, 1] = x[0, 0, 1] * x[0, 1, 1]
        expected_result[0, 1, 1] = x[0, 2, 1] * x[0, 3, 1]

        expected_result[1, 0, 0] = x[1, 0, 0] * x[1, 1, 0]
        expected_result[1, 1, 0] = x[1, 2, 0] * x[1, 3, 0]
        expected_result[1, 0, 1] = x[1, 0, 1] * x[1, 1, 1]
        expected_result[1, 1, 1] = x[1, 2, 1] * x[1, 3, 1]

        # Actual result
        result = prod_layer(x.log()).exp()

        # Run assertions
        self.assertTrue(result.shape[0] == batch_size)
        self.assertTrue(result.shape[1] == in_features // 2)
        self.assertTrue(result.shape[2] == in_channels)
        self.assertTrue(((result - expected_result).abs() < 1e-6).all())

    def test_normal_leaf_layer(self):
        """Test the normal leaf layer."""
        # Setup leaf layer
        multiplicity = 2
        in_features = 2
        leaf = distributions.Normal(multiplicity=multiplicity, in_features=in_features)

        # Setup test input
        batch_size = 2
        x = torch.rand(size=(batch_size, in_features))

        # Setup artificial means and scale matrices
        means = torch.zeros(1, in_features, multiplicity)
        means[0, 0, 0] = 2.5
        means[0, 0, 1] = 5.0
        means[0, 1, 0] = -0.5
        means[0, 1, 1] = -2.0

        scale = torch.zeros(1, in_features, multiplicity)
        scale[0, 0, 0] = 1.0
        scale[0, 0, 1] = 5.0
        scale[0, 1, 0] = 2.0
        scale[0, 1, 1] = 0.1

        # Use scipy norm to get pdfs
        # Expected result
        expected_result = torch.zeros(batch_size, in_features, multiplicity)
        expected_result[0, 0, 0] = TorchNormal(loc=means[0, 0, 0], scale=scale[0, 0, 0]).log_prob(x[0, 0])
        expected_result[0, 0, 1] = TorchNormal(loc=means[0, 0, 1], scale=scale[0, 0, 1]).log_prob(x[0, 0])
        expected_result[0, 1, 0] = TorchNormal(loc=means[0, 1, 0], scale=scale[0, 1, 0]).log_prob(x[0, 1])
        expected_result[0, 1, 1] = TorchNormal(loc=means[0, 1, 1], scale=scale[0, 1, 1]).log_prob(x[0, 1])

        expected_result[1, 0, 0] = TorchNormal(loc=means[0, 0, 0], scale=scale[0, 0, 0]).log_prob(x[1, 0])
        expected_result[1, 0, 1] = TorchNormal(loc=means[0, 0, 1], scale=scale[0, 0, 1]).log_prob(x[1, 0])
        expected_result[1, 1, 0] = TorchNormal(loc=means[0, 1, 0], scale=scale[0, 1, 0]).log_prob(x[1, 1])
        expected_result[1, 1, 1] = TorchNormal(loc=means[0, 1, 1], scale=scale[0, 1, 1]).log_prob(x[1, 1])

        # Perform forward pass in leaf
        leaf.means.data = means
        leaf.stds.data = scale
        result = leaf(x)

        # Make assertions
        self.assertEqual(result.shape[0], batch_size)
        self.assertEqual(result.shape[1], in_features)
        self.assertEqual(result.shape[2], multiplicity)
        self.assertTrue(((result - expected_result).abs() < 1e-6).all())


class TestLayerwiseSampling(unittest.TestCase):
    """Testcases that ensure that sampling methods for Sum, Product and Leaf layers are working as expected."""

    def test_sum_shape_as_root_node(self):
        """Check that the sum node has the correct sampling shape when used as root."""
        for in_channels in [1, 5, 10]:
            for in_features in [1, 5, 10]:
                sum_layer = layers.Sum(in_channels=in_channels, out_channels=1, in_features=in_features)
                samples = sum_layer.sample(n=5)
                self.assertTrue(samples.shape[0] == 5)
                self.assertTrue(samples.shape[1] == in_features)

    def test_product_shape_as_root_node(self):
        """Check that the product node has the correct sampling shape when used as root."""
        prod_layer = layers.Product(in_features=10, cardinality=2)
        samples = prod_layer.sample(n=5)
        self.assertTrue(samples.shape[0] == 5)
        self.assertTrue(samples.shape[1] == 1)

    def test_sum_as_intermediate_node(self):
        """Check that sum node returns the correct sample indices when used as indermediate node."""
        # Some values for the sum layer
        in_features = 10
        in_channels = 50
        out_channels = 50
        n = 2
        parent_indices = torch.randint(out_channels, size=(n, in_features))

        # Create sum layer
        sum_layer = layers.Sum(in_features=in_features, in_channels=in_channels, out_channels=out_channels)

        # Choose `in_features` number of random indexes from 0 to in_channels-1 which will have probability of 1.0 in
        # the sum layer weight tensor
        rand_indxs = torch.randint(in_channels, size=(in_features,))

        # Artificially set sum weights (probabilities) to 1.0
        weights = torch.zeros(in_features, in_channels, out_channels)
        weights[range(in_features), rand_indxs, :] = 1.0
        sum_layer.sum_weights = nn.Parameter(torch.log(weights))

        # Perform sampling
        sample_indices = sum_layer.sample(indices=parent_indices)

        # Assert that the sample indexes are those where the weights were set to 1.0
        self.assertTrue((rand_indxs == sample_indices).all())

    def test_prod_as_intermediate_node(self):
        # Product layer values
        in_features = 10
        num_samples = 5
        for cardinality in range(2, in_features):
            prod_layer = layers.Product(in_features=in_features, cardinality=cardinality)

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
            sample_indices = prod_layer.sample(indices=parent_indices)
            self.assertTrue((expected_sample_indices == sample_indices).all())

    def test_normal_leaf(self):
        # Setup leaf layer
        multiplicity = 10
        in_features = 10
        leaf = distributions.Normal(multiplicity=multiplicity, in_features=in_features)

        # Set leaf layer mean to some random int
        leaf.means.data = torch.randint(low=-100, high=100, size=(1, in_features, multiplicity)).float()
        # Set leaf layer std to 0 such that the samples will all be the mean (so we can actually make assertions in the end)
        leaf.stds.data = torch.zeros(size=(1, in_features, multiplicity)).float()

        # Create some random indices into the multiplicity axis
        parent_indices = torch.randint(high=multiplicity, size=(1, in_features,))

        # Perform sampling
        result = leaf.sample(indices=parent_indices)

        # Expected sampling
        expected_result = leaf.means.data[:, range(in_features), parent_indices]

        # Run assertions
        self.assertTrue(((result - expected_result).abs() < 1e-6).all())

    def test_spn_sampling(self):

        # Define SPN
        leaf = distributions.Normal(in_features=2 ** 3, multiplicity=5)
        sum_1 = layers.Sum(in_channels=5, in_features=2 ** 3, out_channels=20)
        prd_1 = layers.Product(in_features=2 ** 3, cardinality=2)
        sum_2 = layers.Sum(in_channels=20, in_features=2 ** 2, out_channels=20)
        prd_2 = layers.Product(in_features=2 ** 2, cardinality=2)
        sum_3 = layers.Sum(in_channels=20, in_features=2 ** 1, out_channels=20)
        prd_3 = layers.Product(in_features=2 ** 1, cardinality=2)
        sum_4 = layers.Sum(in_channels=20, in_features=2 ** 0, out_channels=1)

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
        x = sum_4.sample(n=1000)
        x = prd_3.sample(indices=x)
        x = sum_3.sample(indices=x)
        x = prd_2.sample(indices=x)
        x = sum_2.sample(indices=x)
        x = prd_1.sample(indices=x)
        x = sum_1.sample(indices=x)
        x = leaf.sample(indices=x)


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
    def test_rat_sampling(self):
        from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpnConstructor

        # Setup RAT
        in_features = 32
        num_classes = 10
        num_sums = 10
        num_input_dists = 20
        num_splits = 5

        rg = RatSpnConstructor(in_features=in_features, C=num_classes, S=num_sums, I=num_input_dists, dropout=0.0)
        for _ in range(0, num_splits):
            rg.random_split(num_parts=2, num_recursions=4)

        model = rg.build()
        model.train()

        # Sample
        samples = model.sample(n=10)
        self.assertTrue(samples.shape[0] == 10)
        self.assertTrue(samples.shape[1] == 32)


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    unittest.main()
