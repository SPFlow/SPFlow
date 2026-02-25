import pytest
import torch

from spflow.meta import Scope
from spflow.modules.products import ElementwiseProduct, OuterProduct, Product
from tests.modules.module_contract_data import PRODUCT_PARAMS
from tests.modules.test_helpers.builders import build_product
from tests.utils.leaves import DummyLeaf, make_data, make_leaf

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("in_channels,out_features,num_reps", PRODUCT_PARAMS)
def test_product_loglikelihood_contract(in_channels: int, out_features: int, num_reps: int):
    module = build_product(in_channels=in_channels, out_features=out_features, num_repetitions=num_reps)
    data = make_data(cls=DummyLeaf, out_features=out_features)
    lls = module.log_likelihood(data)
    # Product collapses its full scope into a single output feature.
    assert lls.shape == (data.shape[0], 1, module.out_shape.channels, num_reps)
    assert torch.isfinite(lls).all()


@pytest.mark.parametrize("cls", [ElementwiseProduct, OuterProduct])
@pytest.mark.parametrize("in_channels,out_features,num_reps", PRODUCT_PARAMS)
def test_base_product_loglikelihood_contract(cls, in_channels: int, out_features: int, num_reps: int):
    scope_a = Scope(list(range(0, out_features)))
    scope_b = Scope(list(range(out_features, out_features * 2)))
    inputs = [
        make_leaf(
            cls=DummyLeaf,
            out_channels=in_channels,
            scope=scope_a,
            num_repetitions=num_reps,
        ),
        make_leaf(
            cls=DummyLeaf,
            out_channels=in_channels,
            scope=scope_b,
            num_repetitions=num_reps,
        ),
    ]
    module = cls(inputs=inputs)
    data = make_data(cls=DummyLeaf, out_features=out_features * len(inputs))
    lls = module.log_likelihood(data)
    # Outer/elementwise variants differ in feature layout, but both must stay numerically stable.
    assert torch.isfinite(lls).all()
