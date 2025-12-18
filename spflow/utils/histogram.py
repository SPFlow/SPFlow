"""Histogram bin edge computation utilities for PyTorch.

Translates NumPy's histogram "auto" bin size estimation to PyTorch tensors.
"""

import torch
from torch import Tensor


def _ptp_torch(x: Tensor) -> Tensor:
    """Compute peak-to-peak (max - min) of a tensor.

    Args:
        x: Input tensor.

    Returns:
        Peak-to-peak value.
    """
    return x.max() - x.min()


def _hist_bin_fd_torch(x: Tensor) -> Tensor:
    """Freedman-Diaconis bin width estimator.

    The Freedman-Diaconis rule uses the interquartile range (IQR) and
    the number of observations to compute an optimal bin width.

    Args:
        x: Input data tensor.

    Returns:
        Estimated bin width.
    """
    iqr = torch.quantile(x.float(), 0.75) - torch.quantile(x.float(), 0.25)
    return 2.0 * iqr * x.size(0) ** (-1.0 / 3.0)


def _hist_bin_sturges_torch(x: Tensor) -> Tensor:
    """Sturges bin width estimator.

    Sturges' formula determines the number of bins based on the
    logarithm of the sample size.

    Args:
        x: Input data tensor.

    Returns:
        Estimated bin width.
    """
    return _ptp_torch(x) / (torch.log2(torch.tensor(x.size(0), dtype=torch.float32)) + 1.0)


def _hist_bin_auto_torch(x: Tensor) -> Tensor:
    """Automatic bin width selector.

    Selects the minimum of Freedman-Diaconis and Sturges estimates,
    falling back to Sturges if FD returns zero.

    Args:
        x: Input data tensor.

    Returns:
        Estimated bin width.
    """
    fd_bw = _hist_bin_fd_torch(x)
    sturges_bw = _hist_bin_sturges_torch(x)
    return min(fd_bw, sturges_bw) if fd_bw > 0 else sturges_bw


def _get_outer_edges_torch(
    a: Tensor, range_bounds: tuple[float, float] | None = None
) -> tuple[float, float]:
    """Determine the outer bin edges from data or given range.

    Args:
        a: Input data tensor.
        range_bounds: Optional (min, max) tuple for bin edges.

    Returns:
        Tuple of (first_edge, last_edge).

    Raises:
        ValueError: If range is invalid or non-finite.
    """
    if range_bounds is not None:
        first_edge, last_edge = range_bounds
        if first_edge > last_edge:
            raise ValueError("max must be larger than min in range parameter.")
        if not (
            torch.isfinite(torch.tensor(first_edge))
            and torch.isfinite(torch.tensor(last_edge))
        ):
            raise ValueError(f"Supplied range [{first_edge}, {last_edge}] is not finite.")
    elif a.numel() == 0:
        # Handle empty tensor case
        first_edge, last_edge = 0.0, 1.0
    else:
        first_edge, last_edge = a.min().item(), a.max().item()
        if not (
            torch.isfinite(torch.tensor(first_edge))
            and torch.isfinite(torch.tensor(last_edge))
        ):
            raise ValueError(
                f"Autodetected range [{first_edge}, {last_edge}] is not finite."
            )

    # Expand if the range is empty to avoid divide-by-zero errors
    if first_edge == last_edge:
        first_edge -= 0.5
        last_edge += 0.5

    return first_edge, last_edge


def get_bin_edges_torch(
    a: Tensor, range_bounds: tuple[float, float] | None = None
) -> tuple[Tensor, tuple[float, float, int]]:
    """Compute histogram bin edges using automatic bin width estimation.

    Uses the same logic as NumPy's histogram with bins='auto'.

    Args:
        a: 1D input data tensor.
        range_bounds: Optional (min, max) tuple for bin edges.

    Returns:
        Tuple of:
            - bin_edges: Tensor of bin edge values
            - (first_edge, last_edge, n_bins): Bin range info

    Raises:
        TypeError: If weights are provided (not supported).
        ValueError: If range values are invalid.
    """
    first_edge, last_edge = _get_outer_edges_torch(a, range_bounds)

    # Filter the array based on the range if necessary
    if range_bounds is not None:
        a = a[(a >= first_edge) & (a <= last_edge)]

    # If the input tensor is empty after filtering, use 1 bin
    if a.numel() == 0:
        n_equal_bins = 1
    else:
        # Calculate the bin width using the automatic estimator
        width = _hist_bin_auto_torch(a)
        if width > 0:
            n_equal_bins = int(torch.ceil((last_edge - first_edge) / width).item())
        else:
            # If width is zero, fall back to 1 bin
            n_equal_bins = 1

    # Compute bin edges
    bin_edges = torch.linspace(
        first_edge, last_edge, n_equal_bins + 1, dtype=torch.float32, device=a.device
    )

    return bin_edges, (first_edge, last_edge, n_equal_bins)
