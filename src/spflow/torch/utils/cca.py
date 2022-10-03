"""
Created on September 26, 2022

@authors: Philipp Deibert
"""
from typing import Tuple
import torch

import numpy as np

def cca(x: torch.Tensor, y: torch.Tensor, n_components: int=2, center: bool=True, scale: bool=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """TODO."""
    # ported from Scikit-learn's 'cross_decomposition.CCA': https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/cross_decomposition/_pls.py#L793
    # all credits belong to the original authors

    # assumed to be different samples
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if x.ndim != 2:
        raise ValueError(f"Observations for 'x' must be either one- or two-dimensional, but was {x.ndim}-dimensional.")
    if y.ndim != 2:
        raise ValueError(f"Observations for 'y' must be either one- or two-dimensional, but was {y.ndim}-dimensional.")
    
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"'x' and 'y' must have same number of observations, but have {x.shape[0]} and {y.shape[0]} observations, respectively.")

    if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
        raise ValueError("Observations for 'x' contain invalid values.")
    if torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
        raise ValueError("Observations for 'y' contain invalid values.")
    
    n_samples = x.shape[0]
    n_features_x = x.shape[1]
    n_features_y = y.shape[1]

    rank_upper_bound = min(n_samples, n_features_x, n_features_y)
    
    if n_components < 1 or n_components > rank_upper_bound:
        raise ValueError(f"Invalid number of components {n_components} for 'cca'.")

    # standardize data
    if center:
        x_mean = x.mean(dim=0)
        x -= x_mean
    
        y_mean = y.mean(dim=0)    
        y -= y_mean

    if scale:
        x_std = x.std(dim=0, unbiased=True)
        x /= x_std

        y_std = y.std(dim=0, unbiased=True)
        y /= y_std

    # allocate data
    x_weights = torch.zeros(n_features_x, n_components)
    y_weights = torch.zeros(n_features_y, n_components)

    x_scores = torch.zeros(n_samples, n_components)
    y_scores = torch.zeros(n_samples, n_components)

    x_loadings = torch.zeros(n_features_x, n_components)
    y_loadings = torch.zeros(n_features_y, n_components)

    # for each component
    for k in range(n_components):

        # compute singular value decomposition
        U, _, Vt = torch.linalg.svd(torch.matmul(x.T, y), full_matrices=False)

        x_w = U[:, 0]
        y_w = Vt[0, :]

        # align signs
        sign = torch.sign(x_w[torch.argmax(x_w.abs())])
        x_w *= sign
        y_w *= sign

        # TODO
        x_weights[:, k] = x_w
        y_weights[:, k] = y_w

        # TODO
        x_scores[:, k] = x_s = torch.matmul(x, x_w)
        y_scores[:, k] = y_s = torch.matmul(y, y_w)

        # TODO
        x_loadings[:, k] = x_l = torch.matmul(x_s, x) / torch.dot(x_s, x_s)
        y_loadings[:, k] = y_l = torch.matmul(y_s, y) / torch.dot(y_s, y_s)

        # update x, y
        x -= torch.outer(x_s, x_l)
        y -= torch.outer(y_s, y_l)

    # compute transformation matrices
    x_rotations = torch.matmul(x_weights, torch.matmul(x_loadings.T, x_weights).pinverse())
    y_rotations = torch.matmul(y_weights, torch.matmul(y_loadings.T, y_weights).pinverse())

    return x_rotations, y_rotations, x_weights, y_weights, x_scores, y_scores, x_loadings, y_loadings