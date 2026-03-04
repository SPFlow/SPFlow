from __future__ import annotations

import torch

from spflow.modules.leaves import Categorical, Normal
from spflow.modules.products.product import Product


def test_sampling_apis_return_tuple_when_leaf_params_requested() -> None:
    leaf = Normal(scope=[0, 1], out_channels=1, num_repetitions=1)

    sample_only = leaf.sample(num_samples=3)
    assert isinstance(sample_only, torch.Tensor)
    assert sample_only.shape == (3, 2)

    samples, records = leaf.sample(num_samples=4, return_leaf_params=True)
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (4, 2)
    assert len(records) == 1

    evidence = torch.full((5, 2), float("nan"))
    samples_ev, records_ev = leaf.sample_with_evidence(evidence=evidence, return_leaf_params=True)
    assert isinstance(samples_ev, torch.Tensor)
    assert samples_ev.shape == (5, 2)
    assert len(records_ev) == 1

    mpe_samples, mpe_records = leaf.mpe(num_samples=6, return_leaf_params=True)
    assert isinstance(mpe_samples, torch.Tensor)
    assert mpe_samples.shape == (6, 2)
    assert len(mpe_records) == 1


def test_sampling_collects_heterogeneous_leaf_parameter_records() -> None:
    torch.manual_seed(11)
    categorical = Categorical(
        scope=[0, 1, 2],
        out_channels=2,
        num_repetitions=1,
        K=4,
    )
    normal = Normal(
        scope=[3, 4, 5],
        out_channels=2,
        num_repetitions=1,
    )
    model = Product(inputs=[categorical, normal])

    samples, records = model.sample(num_samples=7, return_leaf_params=True)

    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (7, 6)
    assert len(records) == 2

    by_type = {record.leaf_type: record for record in records}
    assert {"Categorical", "Normal"} == set(by_type.keys())

    cat_record = by_type["Categorical"]
    assert cat_record.scope_cols == (0, 1, 2)
    assert cat_record.active_mask.shape == (7, 3)
    assert bool(cat_record.active_mask.all())
    assert set(cat_record.params.keys()) == {"logits"}
    assert cat_record.params["logits"].shape == (7, 3, 4)

    normal_record = by_type["Normal"]
    assert normal_record.scope_cols == (3, 4, 5)
    assert normal_record.active_mask.shape == (7, 3)
    assert bool(normal_record.active_mask.all())
    assert {"loc", "scale"}.issubset(set(normal_record.params.keys()))
    assert normal_record.params["loc"].shape == (7, 3)
    assert normal_record.params["scale"].shape == (7, 3)
    if "log_scale" in normal_record.params:
        assert normal_record.params["log_scale"].shape == (7, 3)
