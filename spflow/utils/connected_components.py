"""Algorithm to find connected components in an undirected graph.

Typical usage example:

    cc_list = connected_components(adjacency_matrix)
"""

import scipy
from spflow.tensor import Tensor


def connected_components(adjacency_matrix: Tensor) -> list[set[int]]:
    """Computes all connected components in an undirected graph.

    See also https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html
    """
    result = scipy.sparse.csgraph.connected_components(adjacency_matrix, directed=False)
    component_indices = result[1]
    # TODO: This is the format of how the result is current used in other places. Maybe change this?
    result = [{r} for r in component_indices]
    return result
