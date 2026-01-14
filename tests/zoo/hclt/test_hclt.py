import torch

from spflow.zoo.hclt import learn_hclt_binary


def test_learn_hclt_binary_builds_and_scores() -> None:
    torch.manual_seed(0)
    data = torch.randint(0, 2, (32, 8), dtype=torch.float32)

    model = learn_hclt_binary(data, num_hidden_cats=3, num_trees=2, init="uniform")
    ll = model.log_likelihood(data)

    assert tuple(ll.shape) == (32, 1, 1, 1)
