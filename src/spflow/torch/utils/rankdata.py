"""Algorithm to compute the ranks of specified data.

Typical usage example:

    ranks = rankdata(data, method)
"""
import torch


def rankdata(data: torch.Tensor, method: str = "average") -> torch.Tensor:
    """Computes the ranks for given input samples according to occurrences.

    Returns the ranks for given input samples according to their unique occurrences.
    Implemented in close reference ot SciPy's 'stats.rankdata', but only for two-dimensional inputs.

    Args:
        data:
            Two-dimensional PyTorch tensor containing the input data to be ranked. Each row is regarded as a sample and each column as an independent feature.
        method:
            String specifiying how to handle ties.
            In case of 'average', ties are assigned the average rank of all instances with the same value.
            In case of 'min', ties are assigned the smallest rank of all instances with the same value.
            In case of 'max', ties are assigned the largest rank of all instances with the same value.
            In case of 'ordinal', ties are assigned unique ranks in order of their occurrence.
            In case of 'dense', ties are handled as in the case of 'min', but ranks are reindexed so that there are no skipped rank indices, due to the tie-breaking.
            Defaults to 'average'.

    Returns:
        Two-dimensional PyTorch tensor containing the ranks of the input data.

    Raises:
        ValueError: Invalid method.
    """
    # implemented in reference to SciPy's 'stats.rankdata': https://github.com/scipy/scipy/blob/v1.9.1/scipy/stats/_stats_py.py#L9047-L9153
    # NOTE: method 'ordinal' was not ported since PyTorch's sort/argsort has random tie-breaks (allows for 'stable' option for since PyTorch 1.9.0, but ignored here for backward-compatibility)

    if method not in ["dense", "min", "max", "average"]:
        raise ValueError(f"Unkown method {method} for 'rankdata'.")

    # ids of sorted observations
    sort_ids = torch.argsort(data, dim=0)

    # ids of original observations (to go back from sorted observations to original ones)
    inv_sort_ids = torch.argsort(sort_ids, dim=0)

    # sort observations (avoid recomputation)
    sorted_obs = torch.vstack([feature[order] for feature, order in zip(data.T, sort_ids.T)]).T

    # check if any element is identical to the element before it (for tie-breaking)
    first_occurrence_mask = torch.vstack(
        [
            torch.ones(1, data.shape[1], dtype=bool),
            sorted_obs[1:] != sorted_obs[:-1],
        ]
    )

    # assign each observation its unique id in sorted order
    dense = torch.vstack(
        [s[order] for (s, order) in zip(torch.cumsum(first_occurrence_mask, dim=0).T, inv_sort_ids.T)]
    ).T

    if method == "dense":
        return dense

    # for the empirical cdf we are looking for number of values <= a point, that means all observations that have the same ids count each other, too
    # for each observation id get its id of first occurrence in order (tells us how many observations came before that)
    counts = [
        torch.concat([torch.nonzero(mask).squeeze(-1), torch.tensor([len(mask)])]) for mask in first_occurrence_mask.T
    ]

    if method == "max":
        # return first occurrence id for each assigned observation id
        return torch.vstack([c[d] for c, d in zip(counts, dense.T)]).T

    if method == "min":
        # return first occurrence id for each assigned observation id
        return torch.vstack([c[d - 1] + 1 for c, d in zip(counts, dense.T)]).T

    # else use 'average' method
    return torch.vstack([c[d] + c[d - 1] + 1 for c, d in zip(counts, dense.T)]).T / 2.0