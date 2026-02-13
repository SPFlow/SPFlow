"""Fail-fast tests for non-targeted differentiable sampling paths."""

import pytest
import torch

from spflow.exceptions import UnsupportedOperationError
from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.modules.leaves.cltree import CLTree
from spflow.modules.sos.socs import SOCS
from spflow.zoo.pic.integral import Integral
from spflow.zoo.sos.signed_categorical import SignedCategorical
from spflow.utils.sampling_context import DifferentiableSamplingContext
from tests.utils.leaves import DummyLeaf, make_leaf


def test_cltree_rsample_fail_fast_with_explicit_message():
    leaf = CLTree(scope=Scope([0, 1]), K=2)

    with pytest.raises(UnsupportedOperationError, match="does not support differentiable sampling"):
        leaf.rsample(num_samples=2)


def test_socs_rsample_fail_fast_with_explicit_message():
    component = make_leaf(cls=Normal, out_channels=1, scope=Scope([0]))
    model = SOCS([component])

    with pytest.raises(UnsupportedOperationError, match="does not support differentiable sampling"):
        model.rsample(num_samples=2)


def test_signed_categorical_rsample_fail_fast_with_explicit_message():
    leaf = SignedCategorical(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        K=2,
        weights=torch.tensor([[[[0.2, -0.3]]]], dtype=torch.get_default_dtype()),
    )

    with pytest.raises(UnsupportedOperationError, match="does not support differentiable sampling"):
        leaf.rsample(num_samples=2)


def test_integral_rsample_fail_fast_with_explicit_message():
    child = make_leaf(cls=DummyLeaf, out_channels=2, scope=Scope([0]))
    module = Integral(
        input_module=child,
        latent_scope=Scope([1]),
        integrated_latent_scope=Scope([2]),
        function=None,
    )

    with pytest.raises(UnsupportedOperationError, match="does not support differentiable sampling"):
        module.rsample(num_samples=2)
