import torch
from torch import Tensor, nn

from spflow.meta import Scope
from spflow.modules.leaves import Normal


class LinearNormalParams(nn.Module):
    """Simple conditional Normal parameters: loc = 2 * x, fixed scale."""

    def forward(self, evidence: Tensor) -> dict[str, Tensor]:
        loc = 2.0 * evidence[:, 0:1]
        scale = torch.full_like(loc, 0.1)

        loc = loc.unsqueeze(2).unsqueeze(-1)  # (batch, features=1, channels=1, reps=1)
        scale = scale.unsqueeze(2).unsqueeze(-1)
        return {"loc": loc, "scale": scale}


def test_conditional_leaf_mpe_uses_evidence() -> None:
    leaf = Normal(
        scope=Scope(query=[0], evidence=[1]),
        out_channels=1,
        num_repetitions=1,
        parameter_fn=LinearNormalParams(),
    )

    data = torch.tensor(
        [
            [float("nan"), 1.0],
            [float("nan"), -2.0],
        ],
        dtype=torch.float32,
    )

    mpe = leaf.sample(data=data.clone(), is_mpe=True)
    assert torch.allclose(mpe[:, 0], torch.tensor([2.0, -4.0], dtype=torch.float32))

