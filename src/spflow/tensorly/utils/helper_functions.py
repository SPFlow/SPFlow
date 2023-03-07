import numpy as np
import tensorly as tl
import torch



def tl_ravel(tensor: tl.tensor) -> tl.tensor:
    return tl.reshape(tensor, (-1))


def tl_vstack(stackList):  # all elements have the same shape
    return tl.concatenate(stackList, axis=0).reshape((len(stackList), stackList[0].shape[0]))


def tl_isclose(a: tl.tensor, b: tl.tensor, rtol: float, atol: float) -> tl.tensor:
    return tl.abs(a - b) <= (atol + rtol * tl.abs(b))

def tl_stack(arrays, axis=0): # TODO: Does not work for axis=-1
    # check that all arrays have the same shape
    shapes = [tl.shape(arr) for arr in arrays]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError("All input arrays must have the same shape.")

    # insert a new axis along the specified axis
    stacked = tl.concatenate([tl_unsqueeze(arr, axis=axis) for arr in arrays], axis=axis)
    return stacked

def tl_squeeze(arr, axis=None):
    # get the shape of the input array
    shape = tl.shape(arr)

    # if axis is not specified, remove all dimensions of size 1
    if axis is None:
        squeezed_shape = tuple(d for d in shape if d != 1)
        if squeezed_shape == ():
            return tl.reshape(arr, (1,))
        else:
            return tl.reshape(arr, squeezed_shape)

    # if axis is an integer, remove the dimension at that axis if it has size 1
    elif isinstance(axis, int):
        if shape[axis] != 1:
            return arr
        else:
            return tl.reshape(arr, (*shape[:axis], *shape[axis+1:]))

    # if axis is a tuple of integers, remove the dimensions at each specified axis if they have size 1
    elif isinstance(axis, tuple):
        squeezed_shape = tuple(d for i, d in enumerate(shape) if i not in axis or d != 1)
        return tl.reshape(arr, squeezed_shape)

    # raise an error if axis is not an integer or tuple of integers
    else:
        raise TypeError("Axis must be an integer or tuple of integers.")

def tl_unsqueeze(arr, axis):
    # get the shape of the input array
    shape = tl.shape(arr)

    # if axis is a tuple of integers, add new dimensions at each specified axis
    if isinstance(axis, tuple):
        new_shape = list(shape)
        for ax in axis:
            new_shape.insert(ax, 1)
        return tl.reshape(arr, new_shape)

    # if axis is an integer, add a new dimension at that axis
    elif isinstance(axis, int):
        new_shape = list(shape)
        new_shape.insert(axis, 1)
        return tl.reshape(arr, new_shape)

    # raise an error if axis is not an integer or tuple of integers
    else:
        raise TypeError("Axis must be an integer or tuple of integers.")

def tl_tolist(tensor:tl.tensor):
    backend = tl.get_backend()
    if backend == "numpy" or backend == "pytorch":
        return tensor.tolist()
    else:
        raise NotImplementedError("tl_tolist is not implemented for this backend")

def tl_unique(tensor, axis=None):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.unique(tensor, axis=axis))
    elif backend == "pytorch":
        return tl.tensor(torch.unique(tensor, axis=axis))
    else:
        raise NotImplementedError("tl_unique is not implemented for this backend")

def tl_isnan(tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.isnan(tensor))
    elif backend == "pytorch":
        return tl.tensor(torch.isnan(tensor))
    else:
        raise NotImplementedError("tl_isnan is not implemented for this backend")

def tl_isinf(tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.isinf(tensor))
    elif backend == "pytorch":
        return tl.tensor(torch.isinf(tensor))
    else:
        raise NotImplementedError("tl_isinf is not implemented for this backend")

def tl_isfinite(tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.isfinite(tensor))
    elif backend == "pytorch":
        return tl.tensor(torch.isfinite(tensor))
    else:
        raise NotImplementedError("tl_isfinite is not implemented for this backend")

def tl_full(shape, fill_value, dtype=float):
    return tl.ones(shape,dtype=dtype) * fill_value

def tl_eigvalsh(tensor: tl.tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.linalg.eigvalsh(tensor))
    elif backend == "pytorch":
        return tl.tensor(torch.linalg.eigvalsh(tensor))
    else:
        raise NotImplementedError("tl_eigvalsh is not implemented for this backend")

def tl_inv(tensor: tl.tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.linalg.inv(tensor))
    elif backend == "pytorch":
        return tl.tensor(torch.linalg.inv(tensor))
    else:
        raise NotImplementedError("tl_inv is not implemented for this backend")

def tl_cholesky(tensor: tl.tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.linalg.cholesky(tensor))
    elif backend == "pytorch":
        return tl.tensor(torch.linalg.cholesky(tensor))
    else:
        raise NotImplementedError("tl_cholesky is not implemented for this backend")

def tl_svd(tensor: tl.tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        u, s, vh = np.linalg.svd(tensor)
        return tl.tensor(u), tl.tensor(s), tl.tensor(vh)
    elif backend == "pytorch":
        u, s, vh = torch.linalg.svd(tensor)
        return tl.tensor(u), tl.tensor(s), tl.tensor(vh)
    else:
        raise NotImplementedError("tl_svd is not implemented for this backend")

def tl_real(tensor: tl.tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.real(tensor))
    elif backend == "pytorch":
        return tl.tensor(torch.real(tensor))
    else:
        raise NotImplementedError("tl_cholesky is not implemented for this backend")

def ix_(*args):
    """
    Construct an open mesh from multiple sequences.

    Parameters
    ----------
    *args : sequences
        Sequences to be used as axes of the meshgrid.

    Returns
    -------
    out : tuple of tensors
        Tuple of tensors representing the Cartesian product of the input sequences.
    """
    out = []
    for i, arg in enumerate(args):
        if i == len(args)-1:
            out.append(arg)
        else:
            out.append(tl.transpose(tl.reshape(arg,(1,-1))))

    return tuple(out)

def tensor_nan_to_num(tensor : tl.tensor, copy=True):
    """
    Replace NaN and infinity values in a tensor with zero and finite values, respectively.

    Parameters
    ----------
    tensor : array_like
        Input tensor.
    copy : bool, optional
        Whether to return a copy of `tensor` (True) or modify `tensor` in-place (False). Default is True.

    Returns
    -------
    array_like
        Tensor with NaN and infinity values replaced.

    Notes
    -----
    This function operates element-wise and preserves the shape and dtype of the input tensor.
    """

    if copy:
        tensor = tl.copy(tensor)

    tensor[tl.isnan(tensor)] = 0
    tensor[tl.isinf(tensor)] = tl.sign(tensor[tl.isinf(tensor)]) * tl.finfo(tensor.dtype).max

    return tensor

def tl_cov(tensor: tl.tensor, aweights=None, ddof=None):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.cov(tensor,aweights=aweights,ddof=ddof))
    elif backend == "pytorch":
        return tl.tensor(torch.cov(tensor,aweights=aweights,correction=ddof))
    else:
        raise NotImplementedError("tl_cov is not implemented for this backend")
