from spflow.modules.leaf import Normal
from tests.fixtures import auto_set_test_seed
import unittest

from itertools import product

from spflow.exceptions import InvalidParameterCombinationError, ScopeError
from spflow.meta.data import Scope
import pytest
from spflow.meta.dispatch import init_default_sampling_context, init_default_dispatch_context, SamplingContext
from spflow import log_likelihood, sample, marginalize, sample_with_evidence
from spflow.learn import expectation_maximization
from spflow.learn import train_gradient_descent
from spflow.modules import Sum, ElementwiseProduct
from tests.utils.leaves import make_normal_leaf, make_normal_data, make_leaf
from spflow.modules.learn_spn import learn_spn
import torch

out_features = 5
out_channels = 2

def clustering_fn(x):
    # split into two approximately equal sized clusters
    mask = torch.zeros(x.shape[0])
    mask[int(x.shape[0] / 2) :] = 1
    return mask


def partitioning_fn(x):
    ids = torch.zeros(x.shape[1])

    if not partitioning_fn.alternate or partitioning_fn.partition:
        # split into two approximately equal sized partitions
        partitioning_fn.partition = False
        ids[: int(x.shape[1] / 2)] = 1
    else:
        partitioning_fn.partition = True
    return ids

def test_learn_1():

    data = make_normal_data(num_samples=100, out_features=out_features)
    scope = Scope(list(range(out_features)))


    normal_layer = Normal(scope= scope, out_channels=out_channels)

    # ----- min_features_slice > scope size (no splitting or clustering) -----

    #partitioning_fn.alternate = True
    #partitioning_fn.partition = True

    spn = learn_spn(
        data,
        leaf_modules=normal_layer,
        fit_params=False,
        min_features_slice=4,
    )

def test_learn_spn_2():
    # set seed
    torch.manual_seed(0)
    #np.random.seed(0)
    #random.seed(0)

    data = make_normal_data(num_samples=100, out_features=50)
    scope = Scope(list(range(50)))
    normal_layer = Normal(scope=scope, out_channels=out_channels)

    partitioning_fn.alternate = True
    partitioning_fn.partition = True

    spn = learn_spn(
        data,
        leaf_modules=normal_layer,
        partitioning_method=partitioning_fn,
        clustering_method=clustering_fn,
        fit_params=False,
        min_instances_slice=51,
    )
    test = 5