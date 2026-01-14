import torch

from spflow.meta.data.scope import Scope
from spflow.modules.leaves.normal import Normal
from spflow.modules.sums.sum import Sum
from spflow.utils.cache import Cache
from spflow.zoo.sos import inner_product_matrix


def test_inner_product_matrix_cache_reuses_results_for_shared_submodules():
    # Build a tiny DAG: the same child leaf is re-used twice under Sum inputs.
    leaf = Normal(scope=Scope([0]), out_channels=1, num_repetitions=1)
    root = Sum(
        inputs=[leaf, leaf], out_channels=1, num_repetitions=1, weights=torch.tensor([[[[0.3]], [[0.7]]]])
    )

    cache = Cache()
    k1 = inner_product_matrix(root, root, cache=cache)
    memo = cache.extras.get("_sos_inner_product_memo")
    assert isinstance(memo, dict)
    size_after_first = len(memo)

    # Second call should hit memoization and not grow it.
    k2 = inner_product_matrix(root, root, cache=cache)
    size_after_second = len(memo)

    torch.testing.assert_close(k1, k2)
    assert size_after_second == size_after_first
