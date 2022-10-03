"""
Created on September 26, 2022

@authors: Philipp Deibert
"""
import torch


def connected_components(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """TODO."""
    
    if not torch.all(adjacency_matrix == adjacency_matrix.T):
        raise ValueError("Connected components expects input to be an undirected graph, but specified adjacency matrix was not symmetrical.")

    # convert adjacency matrix to boolean (any non-zero entries are treated as an edge)
    adjacency_matrix = (adjacency_matrix != 0)

    # set of all vertices
    vertices = set(range(adjacency_matrix.shape[0]))

    # list of connected components
    ccs = []

    while vertices:

        # perform breadth-first search
        visited = set()
        active_set = set([vertices.pop()])

        # while there are previously unvisited vertices
        while active_set:

            source = active_set.pop()
            visited.update([source])

            # get neighbors of node
            neighbors = torch.where(adjacency_matrix[source])[0]

            # add all unvisited vertices to active set
            active_set.update(set(neighbors.tolist()).difference(visited))

        # add visited vertices to list of connected components
        ccs.append(visited)

        # remove connected component from set of vertices to explore
        vertices = vertices.difference(visited)

    return ccs