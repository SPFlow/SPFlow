import torch

from spflow.learn.build_socs import build_socs
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.ops.cat import Cat
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

    # All sums should be converted to SignedSum when signed=True.
    for comp in model.components:
        assert any(isinstance(m, SignedSum) for m in comp.modules())

    x = torch.tensor([[0.0], [1.0]])
    ll = model.log_likelihood(x)
    assert torch.isfinite(ll).all()


# Additional branch-focused build_socs tests
import pytest

from spflow.exceptions import InvalidParameterError
from spflow.learn.build_socs import build_abs_weight_proposal
from spflow.modules.products.product import Product


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

    # seed=None exercises rand_like/randn_like branch in conversion.
    model = build_socs(template, num_components=2, signed=True, noise_scale=0.1, flip_prob=0.5, seed=None)
    for comp in model.components:
        assert any(isinstance(m, SignedSum) for m in comp.modules())


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
