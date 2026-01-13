import torch

from spflow.learn.hclt import learn_hclt_categorical


def test_learn_hclt_categorical_builds_and_scores() -> None:
    torch.manual_seed(0)
    data = torch.randint(0, 4, (64, 10), dtype=torch.float32)

    model = learn_hclt_categorical(data, num_hidden_cats=3, num_cats=4, num_trees=2, init="uniform")
    ll = model.log_likelihood(data)

    assert tuple(ll.shape) == (64, 1, 1, 1)

