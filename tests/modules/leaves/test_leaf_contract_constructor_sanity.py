import pytest
import torch

from tests.modules.leaves.leaf_contract_data import LEAF_PARAMS
from tests.utils.leaves import make_leaf

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "leaf_cls,out_features,out_channels, num_reps",
    LEAF_PARAMS,
)
def test_constructor_valid_params(leaf_cls, out_features: int, out_channels: int, num_reps):
    module_a = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    module_a_param_dict = module_a.params()

    # params() must be sufficient to recreate an equivalent module instance.
    module_b = leaf_cls(scope=module_a.scope, **module_a_param_dict)

    state_a = module_a.state_dict()
    state_b = module_b.state_dict()
    # Matching keys guards against silently dropped or renamed constructor parameters.
    assert state_a.keys() == state_b.keys()
    for name in state_a:
        assert torch.isfinite(state_b[name]).all()
        torch.testing.assert_close(state_b[name], state_a[name])


@pytest.mark.parametrize(
    "leaf_cls,out_features,out_channels, num_reps",
    LEAF_PARAMS,
)
def test_constructor_nan_param(leaf_cls, out_features: int, out_channels: int, num_reps):
    module_a = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    nan_params = module_a.params()
    # Inject NaNs in every parameter to ensure validation is parameter-agnostic.
    for key, value in nan_params.items():
        nan_params[key] = torch.full_like(value, torch.nan)

    with pytest.raises(ValueError):
        leaf_cls(scope=module_a.scope, **nan_params).distribution()


@pytest.mark.parametrize(
    "leaf_cls,out_features,out_channels, num_reps",
    LEAF_PARAMS,
)
def test_constructor_inf_param(leaf_cls, out_features: int, out_channels: int, num_reps):
    module_a = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    inf_params = module_a.params()
    # Infinite parameters should fail at distribution construction, not at sampling time.
    for key, value in inf_params.items():
        inf_params[key] = torch.full_like(value, torch.inf)

    with pytest.raises(ValueError):
        leaf_cls(scope=module_a.scope, **inf_params).distribution()


@pytest.mark.parametrize(
    "leaf_cls,out_features,out_channels, num_reps",
    LEAF_PARAMS,
)
def test_constructor_neginf_param(leaf_cls, out_features: int, out_channels: int, num_reps):
    module_a = make_leaf(
        leaf_cls, out_channels=out_channels, out_features=out_features, num_repetitions=num_reps
    )

    neginf_params = module_a.params()
    # Negative infinities hit different domain checks than +inf in several leaves.
    for key, value in neginf_params.items():
        neginf_params[key] = torch.full_like(value, -1 * torch.inf)

    with pytest.raises(ValueError):
        leaf_cls(scope=module_a.scope, **neginf_params).distribution()
