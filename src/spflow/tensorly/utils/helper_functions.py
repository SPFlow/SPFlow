import numpy as np
import tensorly as tl
import torch
from typing import Union, Optional
from scipy.special import logsumexp, softmax

T = Union[np.ndarray, torch.Tensor]

#def tl_ravel(tensor: tl.tensor) -> tl.tensor:
#    return tl.reshape(tensor, (-1))

def tl_ravel(tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.ravel(tensor))
    elif backend == "pytorch":
        if not (torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.ravel(tensor))
    else:
        raise NotImplementedError("tl_ravel is not implemented for this backend")

#def tl_vstack(stackList):  # all elements have the same shape # TODO: Does not work (example test learing layer bernoulli.py
#    return tl.concatenate(stackList, axis=0).reshape((len(stackList), stackList[0].shape[0]))

def tl_vstack(tensor):
    backend = tl.get_backend()

    if isinstance(tensor[0], np.ndarray):#backend == "numpy":
        return tl.tensor(np.vstack(tensor))
    elif torch.is_tensor(tensor[0]):#backend == "pytorch":
        #if not (torch.is_tensor(tensor)):
        #    tensor = torch.tensor(tensor)
        return torch.vstack(tensor)

    else:
        raise NotImplementedError("tl_vstack is not implemented for this backend")

#def tl_isclose(a: tl.tensor, b: tl.tensor, rtol=1e-05, atol=1e-08) -> tl.tensor:
#    return tl.abs(a - b) <= (atol + rtol * tl.abs(b))

def tl_isclose(a, b, rtol=1e-05, atol=1e-08):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.isclose(a=a, b=b, rtol=rtol, atol=atol))
    elif backend == "pytorch":
        #if not (torch.is_tensor(a)):
        a = torch.tensor(a, dtype=torch.float32)
        #if not (torch.is_tensor(b)):
        b = torch.tensor(b, dtype=torch.float32)
        return tl.tensor(torch.isclose(input=a, other=b, rtol=np.float32(rtol), atol=np.float32(atol)))
    else:
        raise NotImplementedError("tl_isclose is not implemented for this backend")

#def tl_allclose(a: tl.tensor, b: tl.tensor, rtol=1e-05, atol=1e-08):
#    a = tl.tensor(a, dtype=tl.float64)
#    b = tl.tensor(b, dtype=tl.float64)
#    return tl.all(tl.abs(a - b) <= (atol + rtol * tl.abs(b)))

def tl_allclose(a, b, rtol=1e-05, atol=1e-08):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.allclose(a=a, b=b, rtol=rtol, atol=atol))
    elif backend == "pytorch":
        #if not (torch.is_tensor(a)):
        a = torch.tensor(a, dtype=torch.float32)
        #if not (torch.is_tensor(b)):
        b = torch.tensor(b, dtype=torch.float32)
        return tl.tensor(torch.allclose(input=a, other=b, rtol=rtol, atol=atol))
    else:
        raise NotImplementedError("tl_isclose is not implemented for this backend")


def tl_stack(arrays, axis=0): # TODO: Does not work for axis=-1; tl.stack verwenden!
    # check that all arrays have the same shape
    shapes = [tl.shape(arr) for arr in arrays]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError("All input arrays must have the same shape.")

    # insert a new axis along the specified axis
    stacked = tl.concatenate([tl_unsqueeze(arr, axis=axis) for arr in arrays], axis=axis)
    return stacked


"""
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
"""

def tl_squeeze(tensor, axis=None):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.squeeze(tensor, axis=axis))
    elif backend == "pytorch":
        if not (torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.squeeze(tensor, dim=axis))
    else:
        raise NotImplementedError("tl_squeeze is not implemented for this backend")

"""
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
"""

def tl_unsqueeze(tensor, axis=None):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.expand_dims(tensor, axis=axis))
    elif backend == "pytorch":
        if not (torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.unsqueeze(tensor, dim=axis))
    else:
        raise NotImplementedError("tl_squeeze is not implemented for this backend")

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
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.unique(tensor, dim=axis))
    else:
        raise NotImplementedError("tl_unique is not implemented for this backend")

def tl_isnan(tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.isnan(tensor))
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.isnan(tensor), dtype=bool)
    else:
        raise NotImplementedError("tl_isnan is not implemented for this backend")

def tl_isinf(tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.isinf(tensor))
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.isinf(tensor), dtype=bool)
    else:
        raise NotImplementedError("tl_isinf is not implemented for this backend")

def tl_isfinite(tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.isfinite(tensor))
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.isfinite(tensor), dtype=bool)
    else:
        raise NotImplementedError("tl_isfinite is not implemented for this backend")

#def tl_full(shape, fill_value, dtype=float):
#    return tl.ones(shape,dtype=dtype) * fill_value

def tl_full(shape, fill_value, dtype=None):
    backend = tl.get_backend()
    if dtype == None:
        dtype = tl.float64
    if backend == "numpy":
        return tl.tensor(np.full(shape, fill_value), dtype=dtype)
    elif backend == "pytorch":
        return tl.tensor(torch.full(shape, fill_value), dtype=dtype)
    else:
        raise NotImplementedError("tl_full is not implemented for this backend")

def tl_eigvalsh(tensor: tl.tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.linalg.eigvalsh(tensor))
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.linalg.eigvalsh(tensor))
    else:
        raise NotImplementedError("tl_eigvalsh is not implemented for this backend")

def tl_inv(tensor: tl.tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.linalg.inv(tensor))
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.linalg.inv(tensor))
    else:
        raise NotImplementedError("tl_inv is not implemented for this backend")

def tl_cholesky(tensor: tl.tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.linalg.cholesky(tensor))
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.linalg.cholesky(tensor))
    else:
        raise NotImplementedError("tl_cholesky is not implemented for this backend")

def tl_svd(tensor: tl.tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        u, s, vh = np.linalg.svd(tensor)
        return tl.tensor(u), tl.tensor(s), tl.tensor(vh)
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        u, s, vh = torch.linalg.svd(tensor)
        return tl.tensor(u), tl.tensor(s), tl.tensor(vh)
    else:
        raise NotImplementedError("tl_svd is not implemented for this backend")

def tl_real(tensor: tl.tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.real(tensor))
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.real(tensor))
    else:
        raise NotImplementedError("tl_cholesky is not implemented for this backend")
"""
def tl_ix_(*args):
   out = []
   for i, arg in enumerate(args):
          if i == len(args)-1:
              out.append(arg)
          else:
              out.append(tl.transpose(tl.reshape(arg,(1,-1))))

   return tuple(out)
"""
def tl_ix_(*args, indexing: Optional): # TODO: test in detail if correct
    backend = tl.get_backend()
    if backend == "numpy":
        arr = np.ix_(*args)
        return tuple([tl.tensor(arr[i]) for i in range(len(arr))])
    elif backend == "pytorch":
        arr = torch.meshgrid(*args, indexing=indexing)
        return tuple([tl.tensor(arr[i]) for i in range(len(arr))])
    else:
        raise NotImplementedError("tl_ix_ is not implemented for this backend")


#def tl_nan_to_num(tensor : tl.tensor, copy=True):
#    """
#    Replace NaN and infinity values in a tensor with zero and finite values, respectively.
#
#    Parameters
#    ----------
#    tensor : array_like
#        Input tensor.
#    copy : bool, optional
#        Whether to return a copy of `tensor` (True) or modify `tensor` in-place (False). Default is True.
#
#    Returns
#    -------
#    array_like
#        Tensor with NaN and infinity values replaced.
#
#    Notes
#    -----
#    This function operates element-wise and preserves the shape and dtype of the input tensor.
#    """
#
#    if copy:
#        tensor = tl.copy(tensor)
#
#    tensor[tl.isnan(tensor)] = 0
#    tensor[tl.isinf(tensor)] = tl.sign(tensor[tl.isinf(tensor)]) * tl.finfo(tensor.dtype).max
#
#    return tensor

def tl_nan_to_num(tensor, copy=True):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.nan_to_num(tensor, copy=copy))
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.nan_to_num(tensor))
    else:
        raise NotImplementedError("tl_nan_to_num is not implemented for this backend")

def tl_cov(tensor: tl.tensor, aweights=None, ddof=None):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.cov(tensor,aweights=aweights,ddof=ddof))
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.cov(tensor,aweights=aweights,correction=ddof))
    else:
        raise NotImplementedError("tl_cov is not implemented for this backend")

def tl_repeat(tensor, repeats, axis=None):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.repeat(tensor,repeats=repeats,axis=axis))
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.repeat_interleave(tensor, repeats=repeats, dim=axis))
    else:
        raise NotImplementedError("tl_cov is not implemented for this backend")

def tl_tile(tensor, repeats):
    backend = tl.get_backend()
    if backend == "numpy":
        return np.tile(tensor,reps=repeats)
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tensor.repeat((1, repeats))
    else:
        raise NotImplementedError("tl_tile is not implemented for this backend")

def tl_spacing(tensor: tl.tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.spacing(tensor))
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.min(
            torch.nextafter(tensor, torch.tensor(float("inf"))) - tensor,
            torch.nextafter(tensor, -torch.tensor(float("inf"))) - tensor,
        ))
    else:
        raise NotImplementedError("tl_cov is not implemented for this backend")

def tl_split(tensor: tl.tensor, indices_or_sections, axis=0):
    backend = tl.get_backend()
    if backend == "numpy":
        arr = np.split(tensor,indices_or_sections=indices_or_sections,axis=axis)
        tensor_list = [tl.tensor(arr[i]) for i in range(len(arr))]
        return tensor_list
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        if not(isinstance(indices_or_sections,list)):
            indices_or_sections = indices_or_sections.tolist()
        arr = torch.tensor_split(tensor, indices_or_sections, axis)
        tensor_list = [el for el in arr]
        return tensor_list
    else:
        raise NotImplementedError("tl_cov is not implemented for this backend")

def tl_array_split(tensor: tl.tensor, indices_or_sections, axis=0):
    backend = tl.get_backend()
    if backend == "numpy":
        arr = np.array_split(tensor, indices_or_sections=indices_or_sections, axis=axis)
        tensor_list = [tl.tensor(arr[i]) for i in range(len(arr))]
        return tensor_list
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        arr = torch.tensor_split(tensor, indices_or_sections, dim=axis)
        tensor_list = [tl.tensor(arr[i]) for i in range(len(arr))]
        return tensor_list
    else:
        raise NotImplementedError("tl_cov is not implemented for this backend")

def tl_pad_edge(tensor, pad_width):
    backend = tl.get_backend()
    if backend == "numpy":
        # pad along axis=1
        return tl.tensor(np.pad(tensor,pad_width=((0,0),pad_width),mode="edge"))
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tl.tensor(torch.nn.functional.pad(tensor, pad=pad_width, mode="replicate"))
    else:
        raise NotImplementedError("tl_cov is not implemented for this backend")

def tl_isinstance(tensor: tl.tensor):
    backend = tl.get_backend()
    if backend == "numpy":
        return isinstance(tensor, np.ndarray)
    elif backend == "pytorch":
        return isinstance(tensor, torch.Tensor)
    else:
        raise NotImplementedError("tl_isinstance is not implemented for this backend")

def tl_nextafter(input, other):
    backend = tl.get_backend()
    if backend == "numpy":
        return tl.tensor(np.nextafter(input,other))
    elif backend == "pytorch":
        if not(torch.is_tensor(input)):
            input = torch.tensor(input)
        if not(torch.is_tensor(other)):
            other = torch.tensor(other)
        return tl.tensor(torch.nextafter(input,other))
    else:
        raise NotImplementedError("tl_nextafter is not implemented for this backend")

def tl_logsumexp(tensor, axis=None, keepdims=False):
    backend = tl.get_backend()
    if backend == "numpy":
        return logsumexp(tensor,axis=axis,keepdims=keepdims)
    elif backend == "pytorch":
        if not(torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        if keepdims==None:
            keepdims=False
        return torch.logsumexp(input=tensor, dim=axis, keepdim=keepdims)
    else:
        raise NotImplementedError("tl_logsumexp is not implemented for this backend")

def tl_softmax(input, axis):
    backend = tl.get_backend()
    if isinstance(input, np.ndarray):#backend == "numpy":
        return softmax(input,axis=axis)
    elif torch.is_tensor(input):#backend == "pytorch":
        if not (torch.is_tensor(input)):
            input = torch.tensor(input)
        return torch.nn.functional.softmax(input=input, dim=axis)
    else:
        raise NotImplementedError("tl_squeeze is not implemented for this backend")

def tl_cartesian_product(*input):
    backend = tl.get_backend()
    if backend == "numpy":
        mesh = np.meshgrid(*input, indexing='ij')
        stacked = np.stack(mesh, axis=-1)
        flattened = stacked.reshape(-1, len(input))
        return flattened
    elif backend == "pytorch":
        return torch.cartesian_prod(*input)
    else:
        raise NotImplementedError("tl_squeeze is not implemented for this backend")

def tl_multinomial(input, num_samples):
    backend = tl.get_backend()
    if backend == "numpy":
        return np.array([np.random.choice(len(i),num_samples, p=i) for i in input])
    elif backend == "pytorch":
        return torch.multinomial(input, num_samples)
    else:
        raise NotImplementedError("tl_multinomial is not implemented for this backend")

def tl_multivariate_normal(loc, cov_matrix, size):
    backend = tl.get_backend()
    if backend == "numpy":
        return np.random.multivariate_normal(loc, cov_matrix, size)
    elif backend == "pytorch":
        return torch.distributions.MultivariateNormal(loc, cov_matrix).sample(size)
    else:
        raise NotImplementedError("tl_multivariate_normal is not implemented for this backend")

def tl_hstack(input):
    backend = tl.get_backend()
    if backend == "numpy":
        return np.hstack(input)
    elif backend == "pytorch":
        return torch.hstack(input)
    else:
        raise NotImplementedError("tl_hstack is not implemented for this backend")

def tl_toNumpy(input):
    if isinstance(input,np.ndarray):
        return input
    elif torch.is_tensor(input):
        return input.detach().cpu().numpy()
    elif isinstance(input, list):
        return np.array(input)
    elif isinstance(input, float):
        return np.array(input)
    elif isinstance(input, np.int32):
        return np.array(input)
    elif isinstance(input, int):
        return np.array(input)
    else:
        raise NotImplementedError("tl_toNumpy is not implemented for this type")
