import itertools

import pytest
import torch

from spflow.utils.cache import Cache
from spflow.zoo.sos import ExpSOCS, ExpSOSModel, SOCS, SOSModel
from spflow.zoo.sos.signed_categorical import SignedCategorical
from spflow.zoo.sos.socs import _signed_eval


def _all_nary(num_variables: int, arity: int) -> torch.Tensor:
    xs = list(itertools.product(range(arity), repeat=num_variables))
    return torch.tensor(xs, dtype=torch.get_default_dtype())


def _check_normalized(ll: torch.Tensor, atol: float = 1e-5) -> None:
    flat = ll.squeeze(-1).squeeze(-1).squeeze(1)
    assert flat.dim() == 1
    assert torch.isfinite(flat).all()
    z = torch.logsumexp(flat, dim=0).exp()
    torch.testing.assert_close(z, torch.tensor(1.0, dtype=z.dtype, device=z.device), rtol=0.0, atol=atol)


@pytest.mark.parametrize(
    "num_variables,num_squares,num_units,structured,input_layer",
    list(
        itertools.product(
            [9, 12],
            [1, 4],
            [1, 2],
            [False, True],
            ["categorical", "embedding"],
        )
    ),
)
def test_sos_model_discrete_normalization_exhaustive(
    num_variables: int,
    num_squares: int,
    num_units: int,
    structured: bool,
    input_layer: str,
):
    kwargs = {"num_categories": 2} if input_layer == "categorical" else {"num_states": 2}
    model = SOSModel(
        num_variables,
        num_input_units=num_units,
        num_sum_units=num_units,
        input_layer=input_layer,
        input_layer_kwargs=kwargs,
        num_squares=num_squares,
        region_graph="rnd-bt",
        structured_decomposable=structured,
        complex=False,
        seed=7,
    )

    x = _all_nary(num_variables, 2)
    ll = model.log_likelihood(x)
    _check_normalized(ll, atol=2e-5)


@pytest.mark.parametrize("num_variables", [9, 12])
def test_sos_model_qt_path_discrete_normalization(num_variables: int):
    image_shape = (1, 3, 3) if num_variables == 9 else (1, 4, 3)
    model = SOSModel(
        num_variables,
        image_shape=image_shape,
        num_input_units=1,
        num_sum_units=1,
        input_layer="categorical",
        input_layer_kwargs={"num_categories": 2},
        num_squares=1,
        region_graph="qt",
        structured_decomposable=True,
        seed=11,
    )

    x = _all_nary(num_variables, 2)
    ll = model.log_likelihood(x)
    _check_normalized(ll, atol=2e-5)


def test_sos_model_complex_matches_bruteforce_tiny_discrete():
    model = SOSModel(
        2,
        num_input_units=2,
        num_sum_units=2,
        input_layer="embedding",
        input_layer_kwargs={"num_states": 2},
        num_squares=1,
        region_graph="rnd-bt",
        structured_decomposable=True,
        complex=True,
        seed=13,
    )

    x = _all_nary(2, 2)
    ll = model.log_likelihood(x).squeeze(-1).squeeze(-1).squeeze(1)

    ((real, imag),) = model.complex_component_pairs
    cache_r = Cache()
    cache_i = Cache()
    logabs_r, sign_r = _signed_eval(real, x, cache_r)
    logabs_i, sign_i = _signed_eval(imag, x, cache_i)

    vr = sign_r.to(dtype=ll.dtype) * torch.exp(logabs_r)
    vi = sign_i.to(dtype=ll.dtype) * torch.exp(logabs_i)
    unnorm = vr.squeeze(-1).squeeze(-1).squeeze(1).pow(2) + vi.squeeze(-1).squeeze(-1).squeeze(1).pow(2)
    expected = torch.log(unnorm / unnorm.sum())

    torch.testing.assert_close(ll, expected, rtol=1e-5, atol=1e-6)


def test_sos_model_enables_non_monotonic_categorical_inputs_when_requested():
    model = SOSModel(
        3,
        num_input_units=2,
        num_sum_units=2,
        input_layer="categorical",
        input_layer_kwargs={"num_categories": 2},
        num_squares=1,
        region_graph="rnd-bt",
        structured_decomposable=True,
        non_monotonic_inputs=True,
        seed=29,
    )

    has_signed_leaf = any(isinstance(m, SignedCategorical) for c in model.components for m in c.modules())
    assert has_signed_leaf


@pytest.mark.parametrize(
    "num_variables,num_units,input_layer,complex",
    list(itertools.product([9, 12], [1, 2], ["categorical", "embedding"], [False, True])),
)
def test_exp_sos_model_discrete_normalization_exhaustive(
    num_variables: int,
    num_units: int,
    input_layer: str,
    complex: bool,
):
    kwargs = {"num_categories": 2} if input_layer == "categorical" else {"num_states": 2}
    model = ExpSOSModel(
        num_variables,
        num_input_units=num_units,
        num_sum_units=num_units,
        mono_num_input_units=2,
        mono_num_sum_units=2,
        input_layer=input_layer,
        input_layer_kwargs=kwargs,
        region_graph="rnd-bt",
        structured_decomposable=True,
        complex=complex,
        seed=17,
    )

    x = _all_nary(num_variables, 2)
    ll = model.log_likelihood(x)
    _check_normalized(ll, atol=2e-5)


@pytest.mark.parametrize("complex", [False, True])
def test_sos_model_matches_equivalent_low_level_socs(complex: bool):
    model = SOSModel(
        3,
        num_input_units=2,
        num_sum_units=2,
        input_layer="embedding",
        input_layer_kwargs={"num_states": 2},
        num_squares=2,
        region_graph="rnd-bt",
        structured_decomposable=True,
        complex=complex,
        seed=19,
    )

    manual = SOCS(model.components)
    x = _all_nary(3, 2)
    ll_model = model.log_likelihood(x)
    ll_manual = manual.log_likelihood(x)
    torch.testing.assert_close(ll_model, ll_manual, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("complex", [False, True])
def test_exp_sos_model_matches_equivalent_low_level_exp_socs(complex: bool):
    model = ExpSOSModel(
        3,
        num_input_units=2,
        num_sum_units=2,
        mono_num_input_units=2,
        mono_num_sum_units=2,
        input_layer="embedding",
        input_layer_kwargs={"num_states": 2},
        region_graph="rnd-bt",
        structured_decomposable=True,
        complex=complex,
        seed=23,
    )

    manual = ExpSOCS(monotone=model.monotone, components=model.components)
    x = _all_nary(3, 2)
    ll_model = model.log_likelihood(x)
    ll_manual = manual.log_likelihood(x)
    torch.testing.assert_close(ll_model, ll_manual, rtol=1e-6, atol=1e-6)
