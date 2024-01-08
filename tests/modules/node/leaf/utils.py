from spflow import log_likelihood
import numpy as np
from spflow import tensor as T


def compare_spflow_with_scipy_dist(spflow_leaf, scipy_log_prob_fn, data):
    """Compare spflow leaf with scipy distribution.

    Args:
        spflow_leaf:
            Spflow leaf.
        scipy_dist:
            Scipy distribution.
        data:
            Data to compare.
    """
    # Compute log likelihoods with scipy
    scipy_ll = scipy_log_prob_fn(data)

    # Compute log likelihoods with spflow
    spflow_ll = log_likelihood(spflow_leaf, data)

    # Compare
    assert np.allclose(T.tolist(spflow_ll), T.tolist(scipy_ll))
