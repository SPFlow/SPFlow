"""
Created on October 14, 2022

@authors: Philipp Deibert
"""
import torch
from packaging import version


def torch_spacing(A: torch.Tensor) -> torch.Tensor:
    return torch.min(
        torch.nextafter(A,  torch.tensor(float("inf")))-A, 
        torch.nextafter(A, -torch.tensor(float("inf")))-A
    )


def nearest_sym_pd(A: torch.Tensor) -> torch.Tensor:
    # compute closest positive definite matrix as described in (Higham, 1988) https://www.sciencedirect.com/science/article/pii/0024379588902236
    # based on MATLAB implementation found here: https://mathworks.com/matlabcentral/fileexchange/42885-nearestspd?s_tid=mwa_osa_a and this Python port: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194

    if version.parse(torch.__version__) < version.parse("1.11.0"):
        exception = RuntimeError
    else:
        exception = torch.linalg.LinAlgError

    def is_pd(A: torch.Tensor) -> torch.Tensor:
        try:
            torch.linalg.cholesky(A)
            return True
        except exception:
            return False

    # make sure matrix is symmetric
    B = (A + A)/2

    # compute symmetric polar factor of B from SVD (which is symmetric positive definite)
    U, s, _ = torch.linalg.svd(B)
    H = torch.matmul(U, torch.matmul(torch.diag(s), U.T))
    
    # compute closest symmetric positive semi-definite matrix to A in Frobenius norm (see paper linked above)
    A_hat = (B+H)/2
    # again, make sure matrix is symmetric
    A_hat = (A_hat + A_hat.T)/2

    # check if matrix is actually symmetric positive-definite
    if is_pd(A_hat):
        return A_hat

    # else fix it
    spacing = torch_spacing(torch.linalg.norm(A_hat))
    I = torch.eye(A.shape[0])
    k = 1

    while not is_pd(A_hat):
        # compute smallest real part eigenvalue
        eigvals = torch.linalg.eigvalsh(A_hat)

        if torch.is_complex(eigvals):
            eigval = torch.real(eigvals)
        
        min_eigval = torch.min(eigvals)
        
        # adjust matrix
        A_hat += I*(-min_eigval*(k**2) + spacing)
        k += 1

    return A_hat