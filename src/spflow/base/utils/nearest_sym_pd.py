"""Algorithm to compute the closest positive definite matrix to a given matrix.

Typical usage example:

    pd_matrix = nearest_sym_pd(matrix)
"""
import numpy as np


def nearest_sym_pd(A: np.ndarray) -> np.ndarray:
    """Algorithm to compute the closest positive definite matrix to a given matrix.

    Returns the closest positive definite matrix to a given matrix in the Frobenius norm,
    as described in (Higham, 1988): "Computing a nearest symmetric positive semidefinite matrix".

    Args:
        A:
            Numpy array to compute closest positive definite matrix to.

    Returns:
        Closest positive definite matrix to input in the Frobenius norm.
    """
    # compute closest positive definite matrix as described in (Higham, 1988) https://www.sciencedirect.com/science/article/pii/0024379588902236
    # based on MATLAB implementation (found here https://mathworks.com/matlabcentral/fileexchange/42885-nearestspd?s_tid=mwa_osa_a) and this Python port: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194

    def is_pd(A: np.ndarray) -> np.ndarray:
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    # make sure matrix is symmetric
    B = (A + A) / 2

    # compute symmetric polar factor of B from SVD (which is symmetric positive definite)
    U, s, _ = np.linalg.svd(B)
    H = np.dot(U, np.dot(np.diag(s), U.T))

    # compute closest symmetric positive semi-definite matrix to A in Frobenius norm (see paper linked above)
    A_hat = (B + H) / 2
    # again, make sure matrix is symmetric
    A_hat = (A_hat + A_hat.T) / 2

    # check if matrix is actually symmetric positive-definite
    if is_pd(A_hat):
        return A_hat

    # else fix it
    spacing = np.spacing(np.linalg.norm(A_hat))
    I = np.eye(A.shape[0])
    k = 1

    while not is_pd(A_hat):
        # compute smallest real part eigenvalue
        min_eigval = np.min(np.real(np.linalg.eigvalsh(A_hat)))
        # adjust matrix
        A_hat += I * (-min_eigval * (k ** 2) + spacing)
        k += 1

    return A_hat
