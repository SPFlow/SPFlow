import itertools
import math

import torch

from spflow.meta.data.scope import Scope
from spflow.modules.leaves.categorical import Categorical
from spflow.utils.cache import Cache
from spflow.zoo.sos import ExpSOCS, SignedCategorical, log_self_inner_product_scalar


def _all_binary(num_variables: int) -> torch.Tensor:
    return torch.tensor(
        list(itertools.product([0, 1], repeat=num_variables)), dtype=torch.get_default_dtype()
    )


def test_signed_categorical_signed_eval_and_inner_product():
    leaf = SignedCategorical(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        K=2,
        weights=torch.tensor([[[[0.2, -0.5]]]], dtype=torch.get_default_dtype()),
    )

    x = _all_binary(1)
    logabs, sign = leaf.signed_logabs_and_sign(x, cache=Cache())

    vals = sign.to(dtype=torch.get_default_dtype()) * torch.exp(logabs)
    expected = torch.tensor([0.2, -0.5], dtype=vals.dtype)
    torch.testing.assert_close(vals.squeeze(-1).squeeze(-1).squeeze(1), expected)

    log_z = log_self_inner_product_scalar(leaf)
    expected_log_z = math.log(0.2 * 0.2 + 0.5 * 0.5)
    torch.testing.assert_close(log_z, torch.tensor(expected_log_z, dtype=log_z.dtype, device=log_z.device))


def test_signed_categorical_triple_product_path_in_exp_socs():
    signed = SignedCategorical(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        K=2,
        weights=torch.tensor([[[[1.0, -0.2]]]], dtype=torch.get_default_dtype()),
    )
    mono = Categorical(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        K=2,
        probs=torch.tensor([[[[0.7, 0.3]]]], dtype=torch.get_default_dtype()),
    )

    model = ExpSOCS(monotone=mono, components=[signed])
    x = _all_binary(1)
    cache = Cache()
    ll = model.log_likelihood(x, cache=cache).squeeze(-1).squeeze(-1).squeeze(1)

    z = 0.7 * (1.0**2) + 0.3 * ((-0.2) ** 2)
    expected = torch.tensor(
        [math.log((0.7 * (1.0**2)) / z), math.log((0.3 * ((-0.2) ** 2)) / z)],
        dtype=ll.dtype,
        device=ll.device,
    )
    torch.testing.assert_close(ll, expected, rtol=1e-6, atol=1e-6)

    log_z = cache.extras["exp_socs_logZ"]
    torch.testing.assert_close(
        log_z,
        torch.tensor(math.log(z), dtype=log_z.dtype, device=log_z.device),
        rtol=1e-7,
        atol=1e-10,
    )
