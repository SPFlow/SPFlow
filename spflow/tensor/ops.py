from typing import Optional
import random

import numpy as np
import scipy

import logging

from spflow.tensor.backend import (
    Tensor,
    get_backend,
    Backend,
    MethodNotImplementedError,
    _TENSOR_TYPES,
    is_jax_available,
)
from spflow.tensor.dtype import (
    get_default_dtype,
    get_default_float_dtype,
    get_default_int_dtype,
    isint,
    int32,
    int64,
)

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    pass

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


def tensor(data: Tensor, dtype=None, device=None, requires_grad=False, copy=False) -> Tensor:
    """
    General tensor constructor for all data inputs.

    The main idea is that this function should be able to convert any data input to a tensor of the data dtype or a
    specified dtype (and device if applicable). If the input data is already a tensor, it is returned as is if
    dtype and device match. Otherwise, a copy is returned with the specified dtype and device.
    """
    backend = get_backend()

    if dtype is None:
        if istensor(data):
            dtype = data.dtype
        else:
            dtype = get_default_dtype(data)
    if backend == Backend.NUMPY:
        if not isinstance(data, np.ndarray) or copy:
            data = np.array(data, dtype=dtype)
    elif backend == Backend.PYTORCH:
        if isinstance(data, torch.Tensor):
            # If source is a tensor, use clone-detach as suggested by PyTorch
            if copy:
                data = data.clone().detach()
        elif is_jax_available() and isinstance(data, jnp.ndarray):
            # If source is a JAX array, convert to numpy and then to PyTorch
            data = torch.tensor(np.asarray(data), dtype=dtype, device=device, requires_grad=requires_grad)
        else:
            # Else, use PyTorch's tensor constructor
            data = torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
    elif backend == Backend.JAX:
        if not isinstance(data, jnp.ndarray) or copy:
            data = jnp.array(data, dtype=dtype)
    else:
        raise MethodNotImplementedError(backend)
    return to(data, dtype=dtype, device=device)


def get_device(data: Tensor):
    # data is numpy return dummy cpu device
    if isinstance(data, np.ndarray):
        return "cpu"
    elif isinstance(data, torch.Tensor):
        return data.device
    elif isinstance(data, jnp.ndarray):
        return jax.devices()[0]
    else:
        raise MethodNotImplementedError(get_backend())


def to(data: Tensor, dtype=None, device=None) -> Tensor:
    """Move tensor to specified device and cast to specified dtype.
    Note:
    - If device is None, the tensor stays on the same device.
    - If dtype is None, the tensor stays with the same dtype.
    """
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.asarray(data, dtype=dtype)
    elif backend == Backend.JAX:
        return jnp.asarray(data, dtype=dtype)
    elif backend == Backend.PYTORCH:
        return data.to(dtype=dtype, device=device)
    else:
        raise MethodNotImplementedError(backend)


def ravel(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.ravel(data)
    elif backend == Backend.PYTORCH:
        return torch.ravel(data)
    elif backend == Backend.JAX:
        return jnp.ravel(data)
    else:
        raise MethodNotImplementedError(backend)


def vstack(tensors: Tensor) -> Tensor:
    backend = get_backend()
    tensors = [tensor(t) for t in tensors]

    if backend == Backend.NUMPY:
        return np.vstack(tensors)
    elif backend == Backend.PYTORCH:
        return torch.vstack(tensors)
    elif backend == Backend.JAX:
        return jnp.vstack(tensors)
    else:
        raise MethodNotImplementedError(backend)


def hstack(tensors) -> Tensor:
    backend = get_backend()
    tensors = [tensor(t) for t in tensors]
    if backend == Backend.NUMPY:
        return np.hstack(tensors)
    elif backend == Backend.PYTORCH:
        return torch.hstack(tensors)
    elif backend == Backend.JAX:
        return jnp.hstack(tensors)
    else:
        raise MethodNotImplementedError(backend)


def isclose(a, b, rtol=1e-05, atol=1e-08) -> Tensor:
    backend = get_backend()
    a, b = tensor(a), tensor(b)
    if backend == Backend.NUMPY:
        return np.isclose(a=a, b=b, rtol=rtol, atol=atol)
    elif backend == Backend.JAX:
        return jnp.isclose(a, b, rtol=rtol, atol=atol)
    elif backend == Backend.PYTORCH:
        a = a.clone().detach()
        b = b.clone().detach()
        return torch.isclose(a, b, rtol=np.float32(rtol), atol=np.float32(atol))
    else:
        raise MethodNotImplementedError(backend)


def allclose(a, b, rtol=1e-05, atol=1e-08) -> Tensor:
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.allclose(a=a, b=b, rtol=rtol, atol=atol)
    elif backend == Backend.PYTORCH:
        a = torch.tensor(a, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        return torch.allclose(input=a, other=b, rtol=rtol, atol=atol)
    elif backend == Backend.JAX:
        return jnp.allclose(a, b, rtol=rtol, atol=atol)
    else:
        raise MethodNotImplementedError(backend)


def tolist(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    data = tensor(data)
    if backend == Backend.NUMPY or backend == Backend.PYTORCH or backend == Backend.JAX:
        return data.tolist()
    else:
        raise MethodNotImplementedError(backend)


def unique(data: Tensor, axis=None) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.unique(data, axis=axis)
    elif backend == Backend.PYTORCH:
        return torch.unique(data, dim=axis)
    elif backend == Backend.JAX:
        return jnp.unique(data, axis=axis)
    else:
        raise MethodNotImplementedError(backend)


def isnan(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.isnan(data)
    elif backend == Backend.JAX:
        return jnp.isnan(data)
    elif backend == Backend.PYTORCH:
        return torch.isnan(data)
    else:
        raise MethodNotImplementedError(backend)


def isinf(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.isinf(data)
    elif backend == Backend.JAX:
        return jnp.isinf(data)
    elif backend == Backend.PYTORCH:
        return torch.isinf(data)
    else:
        raise MethodNotImplementedError(backend)


def isfinite(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.isfinite(data)
    elif backend == Backend.JAX:
        return jnp.isfinite(data)
    elif backend == Backend.PYTORCH:
        return torch.isfinite(data)
    else:
        raise MethodNotImplementedError(backend)


def full(shape, fill_value, dtype=None, device=None) -> Tensor:
    backend = get_backend()
    if dtype is None:
        dtype = get_default_dtype(fill_value)
    if backend == Backend.NUMPY:
        return np.full(shape, fill_value, dtype=dtype)
    elif backend == Backend.JAX:
        return jnp.full(shape, fill_value, dtype=dtype)
    elif backend == Backend.PYTORCH:
        return torch.full(shape, fill_value, dtype=dtype, device=device)
    else:
        raise MethodNotImplementedError(backend)


def eigvalsh(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        result = np.linalg.eigvalsh(data)
    elif backend == Backend.PYTORCH:
        result = torch.linalg.eigvalsh(data)
    elif backend == Backend.JAX:
        result = jnp.linalg.eigvalsh(data)
    else:
        raise MethodNotImplementedError(backend)
    return to(result, dtype=data.dtype)


def inv(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.linalg.inv(data)
    elif backend == Backend.PYTORCH:
        return torch.linalg.inv(data)
    elif backend == Backend.JAX:
        return jnp.linalg.inv(data)
    else:
        raise MethodNotImplementedError(backend)


def cholesky(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.linalg.cholesky(data)
    elif backend == Backend.PYTORCH:
        return torch.linalg.cholesky(data)
    elif backend == Backend.JAX:
        return jnp.linalg.cholesky(data)
    else:
        raise MethodNotImplementedError(backend)


def svd(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        u, s, vh = np.linalg.svd(data)
        return u, s, vh
    elif backend == Backend.PYTORCH:
        u, s, vh = torch.linalg.svd(data)
        return u, s, vh
    elif backend == Backend.JAX:
        u, s, vh = jnp.linalg.svd(data)
        return u, s, vh
    else:
        raise MethodNotImplementedError(backend)


def real(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.real(data)
    elif backend == Backend.PYTORCH:
        return torch.real(data)
    elif backend == Backend.JAX:
        return jnp.real(data)
    else:
        raise MethodNotImplementedError(backend)


def ix_(*args, indexing: Optional = None) -> Tensor:
    backend = get_backend()
    args = [tensor(arg) for arg in args]
    if backend == Backend.NUMPY:
        arr = np.ix_(*args)
        return tuple(tensor(arr[i]) for i in range(len(arr)))
    elif backend == Backend.PYTORCH:
        tensor_list = []
        for i, t in enumerate(args):
            shape = [1] * len(args)
            shape[i] = -1
            reshaped_tensor = t.view(shape)
            tensor_list.append(reshaped_tensor)

        return tuple(tensor_list)
    elif backend == Backend.JAX:
        arr = jnp.ix_(*args)
        return tuple(tensor(arr[i]) for i in range(len(arr)))
    else:
        raise MethodNotImplementedError(backend)


def nan_to_num(data: Tensor, copy=True) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.nan_to_num(data, copy=copy)
    elif backend == Backend.PYTORCH:
        return torch.nan_to_num(data)
    elif backend == Backend.JAX:
        return jnp.nan_to_num(data)
    else:
        raise MethodNotImplementedError(backend)


def cov(data: Tensor, aweights=None, ddof=None) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.cov(data, aweights=aweights, ddof=ddof, dtype=data.dtype)
    elif backend == Backend.PYTORCH:
        return torch.cov(data, aweights=aweights, correction=ddof)
    elif backend == Backend.JAX:
        return jnp.cov(data, aweights=aweights, ddof=ddof)
    else:
        raise MethodNotImplementedError(backend)


def repeat(data: Tensor, repeats, axis=None) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.repeat(data, repeats=repeats, axis=axis)
    elif backend == Backend.PYTORCH:
        return torch.repeat_interleave(data, repeats=repeats, dim=axis)
    elif backend == Backend.JAX:
        return jnp.repeat(data, repeats=repeats, axis=axis)
    else:
        raise MethodNotImplementedError(backend)


def tile(data: Tensor, repeats) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.tile(data, reps=repeats)
    elif backend == Backend.PYTORCH:
        return data.repeat((1, repeats))
    elif backend == Backend.JAX:
        return jnp.tile(data, reps=repeats)
    else:
        raise MethodNotImplementedError(backend)


def spacing(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.spacing(data)
    elif backend == Backend.PYTORCH:
        return torch.nextafter(data, torch.tensor(float("inf"))) - data
    elif backend == Backend.JAX:
        return jnp.nextafter(data, jnp.array(float("inf"))) - data
    else:
        raise MethodNotImplementedError(backend)


def split(data: Tensor, indices_or_sections, axis=0) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        arr = np.split(data, indices_or_sections=indices_or_sections, axis=axis)
        return [a for a in arr]
    elif backend == Backend.PYTORCH:
        if isinstance(indices_or_sections, int):
            res = torch.chunk(data, indices_or_sections, dim=axis)
        elif isinstance(indices_or_sections, list):
            res = torch.tensor_split(data, indices_or_sections, dim=axis)
        else:
            raise ValueError("indices_or_sections must be either int or list of ints")

        return [r for r in res]
    elif backend == Backend.JAX:
        arr = jax.numpy.split(data, indices_or_sections, axis)
        tensor_list = [el for el in arr]
        return tensor_list
    else:
        raise MethodNotImplementedError(backend)


def array_split(data: Tensor, indices_or_sections, axis=0) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        arr = np.array_split(data, indices_or_sections=indices_or_sections, axis=axis)
        tensor_list = [tensor(arr[i]) for i in range(len(arr))]
        return tensor_list
    elif backend == Backend.PYTORCH:
        arr = torch.tensor_split(data, indices_or_sections, dim=axis)
        tensor_list = [tensor(arr[i]) for i in range(len(arr))]
        return tensor_list
    elif backend == Backend.JAX:
        arr = jax.numpy.array_split(data, indices_or_sections, axis)
        tensor_list = [el for el in arr]
        return tensor_list
    else:
        raise MethodNotImplementedError(backend)


def pad_edge(data: Tensor, pad_width) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    data = tensor(data)
    if backend == Backend.NUMPY:
        # pad along axis=1
        return np.pad(data, pad_width=((0, 0), pad_width), mode="edge")
    elif backend == Backend.PYTORCH:
        return torch.nn.functional.pad(data, pad=pad_width, mode="replicate")
    elif backend == Backend.JAX:
        return jnp.pad(data, pad_width=((0, 0), pad_width), mode="edge")
    else:
        raise MethodNotImplementedError(backend)


def istensor(data: Tensor) -> Tensor:
    backend = get_backend()
    if backend == Backend.NUMPY:
        return isinstance(data, np.ndarray)
    elif backend == Backend.PYTORCH:
        return isinstance(data, torch.Tensor)
    elif backend == Backend.JAX:
        return isinstance(data, jnp.ndarray)
    else:
        raise MethodNotImplementedError(backend)


def nextafter(data: Tensor, other: Tensor) -> Tensor:
    backend = get_backend()
    data = tensor(data)
    other = tensor(other)
    if backend == Backend.NUMPY:
        return np.nextafter(data, other)
    elif backend == Backend.PYTORCH:
        return torch.nextafter(data, other)
    elif backend == Backend.JAX:
        return jnp.nextafter(data, other)
    else:
        raise MethodNotImplementedError(backend)


def logsumexp(data: Tensor, axis, keepdims=False) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return scipy.special.logsumexp(data, axis=axis, keepdims=keepdims)
    elif backend == Backend.PYTORCH:
        return torch.logsumexp(input=data, dim=axis, keepdim=keepdims)
    elif backend == Backend.JAX:
        return jax.scipy.special.logsumexp(data, axis=axis, keepdims=keepdims)
    else:
        raise MethodNotImplementedError(backend)


def softmax(data: Tensor, axis) -> Tensor:
    data = tensor(data)
    backend = get_backend()

    if backend == Backend.NUMPY:
        return scipy.special.softmax(data, axis=axis)
    elif backend == Backend.PYTORCH:
        return torch.nn.functional.softmax(input=data, dim=axis)
    elif backend == Backend.JAX:
        return jax.nn.softmax(data, axis=axis)
    else:
        raise MethodNotImplementedError(backend)


def log_softmax(data: Tensor, axis) -> Tensor:
    data = tensor(data)
    backend = get_backend()

    if backend == Backend.NUMPY:
        return scipy.special.log_softmax(data, axis=axis)
    elif backend == Backend.PYTORCH:
        return torch.nn.functional.log_softmax(input=data, dim=axis)
    elif backend == Backend.JAX:
        return jax.nn.log_softmax(data, axis=axis)
    else:
        raise MethodNotImplementedError(backend)


def cartesian_product(*input) -> Tensor:
    backend = get_backend()
    input = [tensor(i) for i in input]
    if backend == Backend.NUMPY:
        mesh = np.meshgrid(*input, indexing="ij")
        stacked = np.stack(mesh, axis=-1)
        flattened = stacked.reshape(-1, len(input))
        return flattened
    elif backend == Backend.PYTORCH:
        return torch.cartesian_prod(*input)
    elif backend == Backend.JAX:
        return jnp.stack(jnp.meshgrid(*input, indexing="ij"), axis=-1).reshape(-1, len(input))
    else:
        raise MethodNotImplementedError(backend)


def multinomial(data: Tensor, num_samples) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.array([np.random.choice(len(i), num_samples, p=i) for i in data])
    elif backend == Backend.PYTORCH:
        return torch.multinomial(data, num_samples)
    else:
        raise MethodNotImplementedError(backend)


def multivariate_normal(loc, cov_matrix, size) -> Tensor:
    loc, cov_matrix = tensor(loc), tensor(cov_matrix)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.random.multivariate_normal(loc, cov_matrix, size)
    elif backend == Backend.PYTORCH:
        return torch.distributions.MultivariateNormal(loc, cov_matrix).sample(size)
    else:
        raise MethodNotImplementedError(backend)


def toNumpy(data: Tensor) -> Tensor:
    if isinstance(data, np.ndarray):
        return data
    elif torch.is_tensor(data):
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, float):
        return np.array(data)
    elif isinstance(data, tl.float32):
        return np.array(data)
    elif isinstance(data, tl.int32):
        return np.array(data)
    elif isinstance(data, int):
        return np.array(data)
    else:
        raise MethodNotImplementedError(get_backend())


def _random_jax_prngkey() -> Tensor:
    return jax.random.PRNGKey(random.randint(0, 2**32 - 1))


def randn(*size, dtype=None, device=None, seed=None) -> Tensor:
    backend = get_backend()
    if dtype is None:
        dtype = get_default_float_dtype()
    if backend == Backend.NUMPY:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(size).astype(dtype)
    elif backend == Backend.JAX:
        key = jax.random.PRNGKey(seed) if seed is not None else _random_jax_prngkey()
        return jax.random.normal(key, shape=size, dtype=dtype)
    elif backend == Backend.PYTORCH:
        with torch.random.fork_rng():
            if seed is not None:
                torch.manual_seed(seed)
            result = torch.randn(*size, dtype=dtype, device=device)
        return result
    else:
        raise MethodNotImplementedError(backend)


def rand(*size, dtype=None, device=None, seed=None) -> Tensor:
    backend = get_backend()
    if dtype is None:
        dtype = get_default_float_dtype()
    if backend == Backend.NUMPY:
        rng = np.random.default_rng(seed)
        return rng.random(size).astype(dtype)
    elif backend == Backend.JAX:
        key = jax.random.PRNGKey(seed) if seed is not None else _random_jax_prngkey()
        return jax.random.uniform(key, shape=size, dtype=dtype)
    elif backend == Backend.PYTORCH:
        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                return torch.rand(*size, dtype=dtype, device=device)
        else:
            return torch.rand(*size, dtype=dtype, device=device)
    else:
        raise MethodNotImplementedError(backend)


def randint(low, high=None, size=None, dtype=None, device=None, seed=None) -> Tensor:
    backend = get_backend()
    if dtype is None:
        dtype = get_default_int_dtype()
    if backend == Backend.NUMPY:
        rng = np.random.default_rng(seed)
        return rng.integers(low, high, size, dtype)
    elif backend == Backend.JAX:
        key = jax.random.PRNGKey(seed) if seed is not None else _random_jax_prngkey()
        return jax.random.randint(key, low, high, shape=size, dtype=dtype)
    elif backend == Backend.PYTORCH:
        with torch.random.fork_rng():
            if seed is not None:
                torch.manual_seed(seed)
            result = torch.randint(low, high, size, dtype=dtype, device=device)
        return result
    else:
        raise MethodNotImplementedError(backend)


def sigmoid(data: Tensor) -> Tensor:
    backend = get_backend()
    data = tensor(data)
    if backend == Backend.NUMPY:
        return 1 / (1 + np.exp(-data))
    elif backend == Backend.JAX:
        return jax.nn.sigmoid(data)
    elif backend == Backend.PYTORCH:
        return torch.sigmoid(data)
    else:
        raise MethodNotImplementedError(backend)


def ones(*size, dtype=None, device=None) -> Tensor:
    backend = get_backend()
    if dtype is None:
        dtype = get_default_float_dtype()
    if backend == Backend.NUMPY:
        return np.ones(size, dtype=dtype)
    elif backend == Backend.JAX:
        return jnp.ones(size, dtype=dtype)
    elif backend == Backend.PYTORCH:
        return torch.ones(*size, dtype=dtype, device=device)
    else:
        raise MethodNotImplementedError(backend)


def zeros(*size, dtype=None, device=None) -> Tensor:
    backend = get_backend()
    if dtype is None:
        dtype = get_default_float_dtype()
    if backend == Backend.NUMPY:
        return np.zeros(size, dtype=dtype)
    elif backend == Backend.JAX:
        return jnp.zeros(size, dtype=dtype)
    elif backend == Backend.PYTORCH:
        return torch.zeros(*size, dtype=dtype, device=device)
    else:
        raise MethodNotImplementedError(backend)


def logical_xor(data_a: Tensor, data_b: Tensor) -> Tensor:
    data_a, data_b = tensor(data_a), tensor(data_b)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.logical_xor(data_a, data_b)
    elif backend == Backend.JAX:
        return jnp.logical_xor(data_a, data_b)
    elif backend == Backend.PYTORCH:
        return torch.logical_xor(data_a, data_b)
    else:
        raise MethodNotImplementedError(backend)


def logical_and(data_a: Tensor, data_b: Tensor) -> Tensor:
    data_a, data_b = tensor(data_a), tensor(data_b)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.logical_and(data_a, data_b)
    elif backend == Backend.JAX:
        return jnp.logical_and(data_a, data_b)
    elif backend == Backend.PYTORCH:
        return torch.logical_and(data_a, data_b)
    else:
        raise MethodNotImplementedError(backend)


def pow(data: Tensor, exponent) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.power(data, exponent)
    elif backend == Backend.JAX:
        return jnp.power(data, exponent)
    elif backend == Backend.PYTORCH:
        return torch.pow(data, exponent)
    else:
        raise MethodNotImplementedError(backend)


def requires_grad_(data: Tensor, flag=True) -> Tensor:
    backend = get_backend()
    if backend == Backend.PYTORCH:
        return data.requires_grad_(flag)
    else:
        return data


def set_tensor_data(destination: Tensor, data: Tensor) -> Tensor:
    """Overwrite the data of the given tensor with the given data.

    This is necesseray for e.g. torch where, if tensor is wrapped in a nn.Parameter, assigning `tensor = data` would
    overwrite the parameter wrapper.
    """
    destination = tensor(destination)
    data = tensor(data, dtype=destination.dtype, device=get_device(destination))
    backend = get_backend()
    if backend == Backend.NUMPY:
        destination[:] = data
        return destination
    elif backend == Backend.PYTORCH:
        destination.data = data
        return destination
    elif backend == Backend.JAX:
        return data
    else:
        raise MethodNotImplementedError(backend)


def arange(start, stop, step=1, dtype=None, device=None) -> Tensor:
    backend = get_backend()
    if dtype is None:
        dtype = get_default_int_dtype()
    if backend == Backend.NUMPY:
        return np.arange(start=start, stop=stop, step=step, dtype=dtype)
    elif backend == Backend.JAX:
        if dtype is None:
            dtype = get_default_float_dtype()
        return jnp.arange(start, stop, step, dtype=dtype)
    elif backend == Backend.PYTORCH:
        return torch.arange(start=start, end=stop, step=step, dtype=dtype, device=device)
    else:
        raise MethodNotImplementedError(backend)


def assign_at_index_2(destination: Tensor, index_1: Tensor, index_2: Tensor, values) -> Tensor:
    destination, index_1, index_2, values = (
        tensor(destination),
        tensor(index_1),
        tensor(index_2),
        tensor(values),
    )
    destination = tensor(destination)
    values = tensor(values)

    # If we have int indices, ensure int64 dtype
    if isint(index_1):
        index_1 = tensor(index_1, dtype=int64())
    else:
        index_1 = tensor(index_1)
    if isint(index_2):
        index_2 = tensor(index_2, dtype=int64())
    else:
        index_2 = tensor(index_2)

    backend = get_backend()
    if backend == Backend.NUMPY:
        destination[index_1, index_2] = values
        return destination
    elif backend == Backend.PYTORCH:
        destination[index_1, index_2] = values
        return destination
    elif backend == Backend.JAX:
        return destination.at[index_1, index_2].set(values)
    else:
        raise MethodNotImplementedError(backend)


def squeeze(data: Tensor, axis=None) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.squeeze(data, axis=axis)
    elif backend == Backend.JAX:
        return jnp.squeeze(data, axis=axis)
    elif backend == Backend.PYTORCH:
        return torch.squeeze(data, dim=axis)
    else:
        raise MethodNotImplementedError(backend)


def unsqueeze(data: Tensor, axis) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.expand_dims(data, axis=axis)
    elif backend == Backend.JAX:
        return jnp.expand_dims(data, axis=axis)
    elif backend == Backend.PYTORCH:
        return torch.unsqueeze(data, dim=axis)
    else:
        raise MethodNotImplementedError(backend)


def lgamma(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return scipy.special.gammaln(data)
    elif backend == Backend.JAX:
        return jax.lax.lgamma(data)
    elif backend == Backend.PYTORCH:
        return torch.lgamma(data)
    else:
        raise MethodNotImplementedError(backend)


def all(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.all(data)
    elif backend == Backend.JAX:
        return jnp.all(data)
    elif backend == Backend.PYTORCH:
        return torch.all(data)
    else:
        raise MethodNotImplementedError(backend)


def any(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.any(data)
    elif backend == Backend.JAX:
        return jnp.any(data)
    elif backend == Backend.PYTORCH:
        return torch.any(data)
    else:
        raise MethodNotImplementedError(backend)


def cumsum(data: Tensor, axis=None) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.cumsum(data, axis=axis)
    elif backend == Backend.JAX:
        return jnp.cumsum(data, axis=axis)
    elif backend == Backend.PYTORCH:
        return torch.cumsum(data, dim=axis)
    else:
        raise MethodNotImplementedError(backend)


def concatenate(arrays: Tensor, axis=0) -> Tensor:
    backend = get_backend()
    arrays = [tensor(a) for a in arrays]
    if backend == Backend.NUMPY:
        return np.concatenate(arrays, axis=axis)
    elif backend == Backend.JAX:
        return jnp.concatenate(arrays, axis=axis)
    elif backend == Backend.PYTORCH:
        return torch.cat(arrays, dim=axis)
    else:
        raise MethodNotImplementedError(backend)


def copy(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.copy(data)
    elif backend == Backend.JAX:
        return jnp.copy(data)
    elif backend == Backend.PYTORCH:
        return data.clone()
    else:
        raise MethodNotImplementedError(backend)


def exp(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.exp(data)
    elif backend == Backend.JAX:
        return jnp.exp(data)
    elif backend == Backend.PYTORCH:
        return torch.exp(data)
    else:
        raise MethodNotImplementedError(backend)


def index_update(data: Tensor, indices, values) -> Tensor:
    data = tensor(data)
    if isint(indices):
        indices = tensor(indices, dtype=int64())
    else:
        indices = tensor(indices)
    backend = get_backend()
    if backend == Backend.NUMPY:
        data[indices] = values
        return data
    elif backend == Backend.JAX:
        return data.at[indices].set(values)
    elif backend == Backend.PYTORCH:
        data[indices] = values
        return data
    else:
        raise MethodNotImplementedError(backend)


def log(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.log(data)
    elif backend == Backend.JAX:
        return jnp.log(data)
    elif backend == Backend.PYTORCH:
        return torch.log(data)
    else:
        raise MethodNotImplementedError(backend)


def min(data: Tensor, axis=None) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.min(data, axis=axis)
    elif backend == Backend.JAX:
        return jnp.min(data, axis=axis)
    elif backend == Backend.PYTORCH:
        if axis is not None:
            return torch.min(data, dim=axis)[0]
        else:
            return torch.min(data)
    else:
        raise MethodNotImplementedError(backend)


def max(data: Tensor, axis=None) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.max(data, axis=axis)
    elif backend == Backend.JAX:
        return jnp.max(data, axis=axis)
    elif backend == Backend.PYTORCH:
        if axis is not None:
            return torch.max(data, dim=axis)[0]
        else:
            return torch.max(data)
    else:
        raise MethodNotImplementedError(backend)


def abs(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.abs(data)
    elif backend == Backend.JAX:
        return jnp.abs(data)
    elif backend == Backend.PYTORCH:
        return torch.abs(data)
    else:
        raise MethodNotImplementedError(backend)


def ndim(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.ndim(data)
    elif backend == Backend.JAX:
        return jnp.ndim(data)
    elif backend == Backend.PYTORCH:
        return data.ndim
    else:
        raise MethodNotImplementedError(backend)


def prod(data: Tensor, axis=None) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.prod(data, axis=axis)
    elif backend == Backend.JAX:
        return jnp.prod(data, axis=axis)
    elif backend == Backend.PYTORCH:
        return torch.prod(data, dim=axis)
    else:
        raise MethodNotImplementedError(backend)


def shape(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.shape(data)
    elif backend == Backend.JAX:
        return jnp.shape(data)
    elif backend == Backend.PYTORCH:
        return data.shape
    else:
        raise MethodNotImplementedError(backend)


def sqrt(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.sqrt(data)
    elif backend == Backend.JAX:
        return jnp.sqrt(data)
    elif backend == Backend.PYTORCH:
        return torch.sqrt(data)
    else:
        raise MethodNotImplementedError(backend)


def stack(arrays: Tensor, axis=0) -> Tensor:
    backend = get_backend()
    arrays = [tensor(a) for a in arrays]
    if backend == Backend.NUMPY:
        return np.stack(arrays, axis=axis)
    elif backend == Backend.JAX:
        return jnp.stack(arrays, axis=axis)
    elif backend == Backend.PYTORCH:
        return torch.stack(arrays, dim=axis)
    else:
        raise MethodNotImplementedError(backend)


def sum(data: Tensor, axis=None, keepdims=False) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.sum(data, axis=axis, keepdims=keepdims)
    elif backend == Backend.JAX:
        return jnp.sum(data, axis=axis, keepdims=keepdims)
    elif backend == Backend.PYTORCH:
        if axis is None:
            axis = ()  # Necessary for PyTorch<=2.0.0 (doesn't accept None)
        return torch.sum(data, dim=axis, keepdim=keepdims)
    else:
        raise MethodNotImplementedError(backend)


def diag(data: Tensor) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.diag(data)
    elif backend == Backend.JAX:
        return jnp.diag(data)
    elif backend == Backend.PYTORCH:
        return torch.diag(data)
    else:
        raise MethodNotImplementedError(backend)


def transpose(data: Tensor, axes=None) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.transpose(data, axes=axes)
    elif backend == Backend.JAX:
        return jnp.transpose(data, axes=axes)
    elif backend == Backend.PYTORCH:
        axes = axes or list(range(ndim(data)))[::-1]
        return data.permute(*axes)
    else:
        raise MethodNotImplementedError(backend)


def dot(data_a: Tensor, data_b: Tensor) -> Tensor:
    data_a, data_b = tensor(data_a), tensor(data_b)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.dot(data_a, data_b)
    elif backend == Backend.JAX:
        return jnp.dot(data_a, data_b)
    elif backend == Backend.PYTORCH:
        return torch.dot(data_a, data_b)
    else:
        raise MethodNotImplementedError(backend)


def norm(data: Tensor, ord=None, axis=None) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.linalg.norm(data, ord=ord, axis=axis)
    elif backend == Backend.JAX:
        return jnp.linalg.norm(data, ord=ord, axis=axis)
    elif backend == Backend.PYTORCH:
        return torch.linalg.norm(data, ord=ord, dim=axis)
    else:
        raise MethodNotImplementedError(backend)


def eye(N, dtype=None, device=None) -> Tensor:
    backend = get_backend()
    if dtype is None:
        dtype = get_default_float_dtype()
    if backend == Backend.NUMPY:
        return np.eye(N, dtype=dtype)
    elif backend == Backend.JAX:
        return jnp.eye(N, dtype=dtype)
    elif backend == Backend.PYTORCH:
        return torch.eye(N, dtype=dtype, device=device)
    else:
        raise MethodNotImplementedError(backend)


def sort(data: Tensor, axis=-1) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.sort(data, axis=axis)
    elif backend == Backend.JAX:
        return jnp.sort(data, axis=axis)
    elif backend == Backend.PYTORCH:
        return torch.sort(data, dim=axis)[0]
    else:
        raise MethodNotImplementedError(backend)


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    condition, x, y = tensor(condition), tensor(x), tensor(y)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.where(condition, x, y)
    elif backend == Backend.JAX:
        return jnp.where(condition, x, y)
    elif backend == Backend.PYTORCH:
        return torch.where(condition, x, y)
    else:
        raise MethodNotImplementedError(backend)


def reshape(data: Tensor, shape) -> Tensor:
    data = tensor(data)
    backend = get_backend()
    if backend == Backend.NUMPY:
        return np.reshape(data, shape)
    elif backend == Backend.JAX:
        return jnp.reshape(data, shape)
    elif backend == Backend.PYTORCH:
        return torch.reshape(data, shape)
    else:
        raise MethodNotImplementedError(backend)

def searchsorted(sorted_sequence, values, side="left") -> Tensor:
    sorted_sequence, values = tensor(sorted_sequence), tensor(values)
    backend = get_backend()
    if backend == Backend.NUMPY:
        result = np.searchsorted(sorted_sequence, values, side=side)
    elif backend == Backend.JAX:
        result = jnp.searchsorted(sorted_sequence, values, side=side)
    elif backend == Backend.PYTORCH:
        result = torch.searchsorted(sorted_sequence, values, side=side)
    else:
        raise MethodNotImplementedError(backend)
    return to(result, dtype=int32())


def bincount(data: Tensor, weights=None, minlength=0) -> Tensor:
    if weights is None:
        weights = ones(len(data), device=get_device(data))
    data, weights = tensor(data), tensor(weights)
    backend = get_backend()
    if backend == Backend.NUMPY:
        result = np.bincount(data, weights=weights, minlength=minlength)
    elif backend == Backend.JAX:
        result = jnp.bincount(data, weights=weights, minlength=minlength)
    elif backend == Backend.PYTORCH:
        result = torch.bincount(data, weights=weights, minlength=minlength)
    else:
        raise MethodNotImplementedError(backend)

    return to(result, dtype=get_default_int_dtype())
