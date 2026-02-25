import torch

from spflow.learn.build_socs import build_abs_weight_proposal, build_complex_socs, build_socs
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.leaves.normal import Normal
from spflow.modules.ops.cat import Cat
from spflow.modules.products.product import Product
from spflow.modules.sums.signed_sum import SignedSum
from spflow.modules.sums.sum import Sum
from spflow.utils.compatibility import check_socs_compatibility


def test_build_socs_creates_compatible_components_and_signed_sums():
    scope = Scope([0])
    leaves = [
        Bernoulli(scope=scope, out_channels=1, num_repetitions=1, probs=torch.tensor([[[0.2]]])),
        Bernoulli(scope=scope, out_channels=1, num_repetitions=1, probs=torch.tensor([[[0.8]]])),
    ]
    template = Sum(inputs=Cat(inputs=leaves, dim=2), out_channels=1, num_repetitions=1)

    model = build_socs(template, num_components=3, signed=True, noise_scale=0.1, flip_prob=0.5, seed=0)
    assert len(model.components) == 3
    check_socs_compatibility(model)

    # signed=True is a contract that every sum-like node uses signed weights.
    for comp in model.components:
        assert any(isinstance(m, SignedSum) for m in comp.modules())

    x = torch.tensor([[0.0], [1.0]])
    ll = model.log_likelihood(x)
    assert torch.isfinite(ll).all()


import pytest

from spflow.exceptions import InvalidParameterError


def _make_template() -> Product:
    left = Sum(inputs=Bernoulli(scope=Scope([0]), out_channels=1), out_channels=1)
    right = Sum(inputs=Bernoulli(scope=Scope([1]), out_channels=1), out_channels=1)
    return Product([left, right])


def test_build_socs_parameter_validation_and_rng_none_paths():
    template = _make_template()

    with pytest.raises(InvalidParameterError):
        build_socs(template, num_components=0)
    with pytest.raises(InvalidParameterError):
        build_socs(template, num_components=1, noise_scale=-1.0)
    with pytest.raises(InvalidParameterError):
        build_socs(template, num_components=1, flip_prob=2.0)

    unsigned = build_socs(template, num_components=2, signed=False)
    assert not any(isinstance(m, SignedSum) for c in unsigned.components for m in c.modules())

    # seed=None should still use stochastic conversion paths without explicit RNG.
    model = build_socs(template, num_components=2, signed=True, noise_scale=0.1, flip_prob=0.5, seed=None)
    for comp in model.components:
        assert any(isinstance(m, SignedSum) for m in comp.modules())


def test_build_socs_invalid_parameters_raise():
    template = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)
    with pytest.raises(InvalidParameterError):
        build_socs(template, num_components=0)
    with pytest.raises(InvalidParameterError):
        build_socs(template, num_components=1, noise_scale=-0.1)
    with pytest.raises(InvalidParameterError):
        build_socs(template, num_components=1, flip_prob=1.1)


def test_build_socs_unsigned_and_random_generator_none_branch():
    leaf_a = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1, probs=torch.tensor([[[0.3]]]))
    leaf_b = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1, probs=torch.tensor([[[0.7]]]))
    template = Sum(inputs=Cat(inputs=[leaf_a, leaf_b], dim=2), out_channels=1, num_repetitions=1)

    unsigned = build_socs(template, num_components=2, signed=False)
    assert not any(isinstance(m, SignedSum) for c in unsigned.components for m in c.modules())

    signed = build_socs(template, num_components=2, signed=True, noise_scale=0.1, flip_prob=0.5, seed=None)
    assert any(isinstance(m, SignedSum) for c in signed.components for m in c.modules())


def test_build_abs_weight_proposal_validation_and_recursive_signed_replacement():
    template = _make_template()
    signed_component = build_socs(
        template, num_components=1, signed=True, noise_scale=0.0, flip_prob=0.0, seed=0
    ).components[0]

    with pytest.raises(InvalidParameterError):
        build_abs_weight_proposal(signed_component, eps=0.0)

    proposal = build_abs_weight_proposal(signed_component, eps=1e-6)
    assert not any(isinstance(m, SignedSum) for m in proposal.modules())
    assert any(isinstance(m, Sum) for m in proposal.modules())


def test_build_abs_weight_proposal_eps_and_nested_signedsum_conversion():
    leaf_a = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1, probs=torch.tensor([[[0.2]]]))
    leaf_b = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1, probs=torch.tensor([[[0.8]]]))
    signed = SignedSum(
        inputs=Cat(inputs=[leaf_a, leaf_b], dim=2),
        out_channels=1,
        num_repetitions=1,
        weights=torch.tensor([[[[1.0]], [[-0.5]]]]),
    )

    with pytest.raises(InvalidParameterError):
        build_abs_weight_proposal(signed, eps=0.0)

    proposal = build_abs_weight_proposal(signed, eps=1e-5)
    assert not any(isinstance(m, SignedSum) for m in proposal.modules())


def test_build_complex_socs_validation_and_success():
    real = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)
    imag_same = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)
    imag_scope = Normal(scope=Scope([1]), out_channels=1, num_repetitions=1)
    imag_shape = Normal(scope=Scope([0]), out_channels=2, num_repetitions=1)

    with pytest.raises(InvalidParameterError):
        build_complex_socs(real, imag_scope)
    with pytest.raises(InvalidParameterError):
        build_complex_socs(real, imag_shape)

    model = build_complex_socs(real, imag_same)
    assert len(model.components) == 2


def test_build_socs_replaces_nested_sum_nodes():
    same_scope = Scope([0])
    sum_branch = Sum(
        inputs=[
            Bernoulli(scope=same_scope, out_channels=1, num_repetitions=1),
            Bernoulli(scope=same_scope, out_channels=1, num_repetitions=1),
        ],
        out_channels=1,
        num_repetitions=1,
    )
    template = Product([sum_branch, Normal(scope=Scope([1]), out_channels=1, num_repetitions=1)])

    model = build_socs(template, num_components=1, signed=True, noise_scale=0.0, flip_prob=0.5, seed=7)
    assert any(isinstance(m, SignedSum) for m in model.components[0].modules())


def test_build_abs_weight_proposal_replaces_nested_signedsum_nodes():
    same_scope = Scope([0])
    signed_branch = SignedSum(
        inputs=[
            Bernoulli(scope=same_scope, out_channels=1, num_repetitions=1),
            Bernoulli(scope=same_scope, out_channels=1, num_repetitions=1),
        ],
        out_channels=1,
        num_repetitions=1,
        weights=torch.tensor([[[[1.0]], [[-0.5]]]]),
    )
    component = Product([signed_branch, Normal(scope=Scope([1]), out_channels=1, num_repetitions=1)])
    proposal = build_abs_weight_proposal(component)
    assert not any(isinstance(m, SignedSum) for m in proposal.modules())
