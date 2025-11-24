import numpy as np
import torch
from scipy.stats import rankdata


def rdc(x, y, f=torch.sin, k=20, s=1 / 6.0, n=1):
    """Computes the Randomized Dependence Coefficient.

    Source: https://github.com/garydoranjr/rdc/blob/master/rdc/rdc.py

    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.

    Args:
        x: Input tensor of shape (samples,) or (samples, variables).
        y: Input tensor of shape (samples,) or (samples, variables).
        f: Function to use for random projection. Defaults to torch.sin.
        k: Number of random projections to use. Defaults to 20.
        s: Scale parameter. Defaults to 1/6.0.
        n: Number of times to compute the RDC and return the median for stability.
            Defaults to 1.

    Returns:
        Tensor: The Randomized Dependence Coefficient value.
    """
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except torch.linalg.LinAlgError:
                pass
        return torch.median(torch.tensor(values, device=x.device, dtype=x.dtype))

    if len(x.shape) == 1:
        x = x.reshape((-1, 1))
    if len(y.shape) == 1:
        y = y.reshape((-1, 1))

    # Copula Transformation
    cx = torch.stack([rankdata_ordinal(xc) for xc in x.T], dim=1) / torch.prod(torch.tensor(x.shape))
    cy = torch.stack([rankdata_ordinal(yc) for yc in y.T], dim=1) / torch.prod(torch.tensor(y.shape))

    # Add a vector of ones so that w.x + b is just a dot product
    O = torch.ones(cx.shape[0], 1)
    X = torch.stack([cx, O], dim=1)
    Y = torch.stack([cy, O], dim=1)

    # Random linear projections
    Rx = (s / X.shape[1]) * torch.randn(X.shape[1], k)
    Ry = (s / Y.shape[1]) * torch.randn(Y.shape[1], k)
    X = torch.mm(X.squeeze(-1), Rx)
    Y = torch.mm(Y.squeeze(-1), Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = torch.cov(torch.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:
        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0 : k0 + k, k0 : k0 + k]
        Cxy = C[:k, k0 : k0 + k]
        Cyx = C[k0 : k0 + k, :k]

        eigs = torch.linalg.eigvals(
            torch.mm(torch.mm(torch.linalg.pinv(Cxx), Cxy), torch.mm(torch.linalg.pinv(Cyy), Cyx))
        )

        # Handle complex eigenvalues properly
        if torch.is_complex(eigs):
            # Check if imaginary parts are negligible (numerical precision issues)
            max_imag = torch.max(torch.abs(torch.imag(eigs)))
            if max_imag > 1e-8:
                # Significant imaginary parts indicate numerical issues
                ub -= 1
                k = (ub + lb) // 2
                continue

            # Use real parts for the range check
            eigs_real = torch.real(eigs)
            eigenvals_valid = 0 <= torch.min(eigs_real) and torch.max(eigs_real) <= 1
        else:
            # Real eigenvalues - use original logic
            eigenvals_valid = 0 <= torch.min(eigs) and torch.max(eigs) <= 1

        # Binary search if k is too large
        if not eigenvals_valid:
            ub -= 1
            k = (ub + lb) // 2
            continue

        if lb == ub:
            break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    # Return the square root of the maximum eigenvalue (using real part if complex)
    if torch.is_complex(eigs):
        return torch.sqrt(torch.max(torch.real(eigs)))
    else:
        return torch.sqrt(torch.max(eigs))


def cca_loop(k, C):
    """Computes canonical correlations using binary search for numerical stability.

    Args:
        k: Number of random projections to use.
        C: Covariance matrix of shape (2*k, 2*k).

    Returns:
        Tensor: Square root of the maximum eigenvalue (canonical correlation).
    """
    k0 = k
    lb = 1
    ub = k
    while True:
        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0 : k0 + k, k0 : k0 + k]
        Cxy = C[:k, k0 : k0 + k]
        Cyx = C[k0 : k0 + k, :k]

        eigs = torch.linalg.eigvals(
            torch.mm(torch.mm(torch.linalg.pinv(Cxx), Cxy), torch.mm(torch.linalg.pinv(Cyy), Cyx))
        )

        # Handle complex eigenvalues properly
        if torch.is_complex(eigs):
            # Check if imaginary parts are negligible (numerical precision issues)
            max_imag = torch.max(torch.abs(torch.imag(eigs)))
            if max_imag > 1e-8:
                # Significant imaginary parts indicate numerical issues
                ub -= 1
                k = (ub + lb) // 2
                continue

            # Use real parts for the range check
            eigs_real = torch.real(eigs)
            eigenvals_valid = 0 <= torch.min(eigs_real) and torch.max(eigs_real) <= 1
        else:
            # Real eigenvalues - use original logic
            eigenvals_valid = 0 <= torch.min(eigs) and torch.max(eigs) <= 1

        # Binary search if k is too large
        if not eigenvals_valid:
            ub -= 1
            k = (ub + lb) // 2
            continue

        if lb == ub:
            break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    # Return the square root of the maximum eigenvalue (using real part if complex)
    if torch.is_complex(eigs):
        return torch.sqrt(torch.max(torch.real(eigs)))
    else:
        return torch.sqrt(torch.max(eigs))


def rankdata_ordinal(x):
    """PyTorch equivalent of scipy.stats.rankdata(method='ordinal').

    Args:
        x: Input tensor to rank.

    Returns:
        Tensor: Ordinal ranks of the input values.
    """
    # Get the indices that would sort the array
    sorted_indices = torch.argsort(x)

    # Create ranks array
    ranks = torch.empty_like(sorted_indices, dtype=torch.float)
    ranks[sorted_indices] = torch.arange(1, len(x) + 1, dtype=torch.float)

    return ranks


def rdc_np(x, y, f=np.sin, k=20, s=1 / 6.0, n=1):
    """Computes the Randomized Dependence Coefficient using NumPy.

    Source: https://github.com/garydoranjr/rdc/blob/master/rdc/rdc.py

    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.

    Args:
        x: Input array of shape (samples,) or (samples, variables).
        y: Input array of shape (samples,) or (samples, variables).
        f: Function to use for random projection. Defaults to np.sin.
        k: Number of random projections to use. Defaults to 20.
        s: Scale parameter. Defaults to 1/6.0.
        n: Number of times to compute the RDC and return the median for stability.
            Defaults to 1.

    Returns:
        numpy.ndarray: The Randomized Dependence Coefficient value.
    """
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError:
                pass
        return np.median(values)

    if len(x.shape) == 1:
        x = x.reshape((-1, 1))
    if len(y.shape) == 1:
        y = y.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method="ordinal") for xc in x.T]) / float(x.size)
    cy = np.column_stack([rankdata(yc, method="ordinal") for yc in y.T]) / float(y.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s / X.shape[1]) * torch.randn(X.shape[1], k).cpu().numpy()
    Ry = (s / Y.shape[1]) * torch.randn(X.shape[1], k).cpu().numpy()
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)
    # C = C.astype(np.float32)
    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:
        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0 : k0 + k, k0 : k0 + k]
        Cxy = C[:k, k0 : k0 + k]
        Cyx = C[k0 : k0 + k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy), np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and 0 <= np.min(eigs) and np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub:
            break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))


def cca_loop_np(k, C):
    """Computes canonical correlations using binary search for numerical stability (NumPy version).

    Args:
        k: Number of random projections to use.
        C: Covariance matrix of shape (2*k, 2*k).

    Returns:
        numpy.ndarray: Square root of the maximum eigenvalue (canonical correlation).
    """
    k0 = k
    lb = 1
    ub = k
    while True:
        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0 : k0 + k, k0 : k0 + k]
        Cxy = C[:k, k0 : k0 + k]
        Cyx = C[k0 : k0 + k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy), np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and 0 <= np.min(eigs) and np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub:
            break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))
