import torch

from spflow.meta.data.scope import Scope
from spflow.modules.leaves.normal import Normal
from spflow.modules.sums.signed_sum import SignedSum
from spflow.utils.cache import Cache


def test_signed_sum_signed_eval_shapes_and_finiteness():
    leaf_a = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)
    leaf_b = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)

    # Cat(dim=2) will create in_channels=2
    w = torch.tensor([[[[1.0]], [[-0.5]]]])  # (F=1, IC=2, OC=1, R=1)
    node = SignedSum(inputs=[leaf_a, leaf_b], out_channels=1, num_repetitions=1, weights=w)

    x = torch.randn(7, 1)
    cache = Cache()
    logabs, sign = node.signed_logabs_and_sign(x, cache=cache)

    assert logabs.shape == (7, 1, 1, 1)
    assert sign.shape == (7, 1, 1, 1)
    assert torch.isfinite(logabs).all()
    assert ((sign == -1) | (sign == 0) | (sign == 1)).all()
