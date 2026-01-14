import torch

from spflow.exp.sos import build_socs
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.ops.cat import Cat
from spflow.exp.sos import SignedSum
from spflow.modules.sums.sum import Sum
from spflow.exp.sos import check_socs_compatibility


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
