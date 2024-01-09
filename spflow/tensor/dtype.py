import numpy as np

try:
    import torch
except ImportError:
    pass

try:
    import jax.numpy as jnp
except ImportError:
    pass

from spflow.tensor.backend import (
    Tensor,
    _TENSOR_TYPES,  # We need this for isinstance checks against Tensor in Python<=3.9
    is_torch_available,
    is_jax_available,
    is_numpy,
    is_pytorch,
    is_jax,
    get_backend,
    Backend,
    MethodNotImplementedError,
)


def isfloat(data: Tensor):
    if isinstance(data, list) and len(data) > 0:
        return isfloat(data[0])
    elif isinstance(data, tuple) and len(data) > 0:
        return isfloat(data[0])
    elif isinstance(data, (float, np.floating)):
        return True
    elif isinstance(data, _TENSOR_TYPES):
        if isinstance(data, np.ndarray):
            return np.issubdtype(data.dtype, np.floating)
        elif is_torch_available() and isinstance(data, torch.Tensor):
            return torch.is_floating_point(data)
        elif is_jax_available() and isinstance(data, jnp.ndarray):
            return jnp.issubdtype(data.dtype, jnp.floating)
    return False


def isint(data: Tensor):
    if isinstance(data, list) and len(data) > 0:
        return isint(data[0])
    elif isinstance(data, tuple) and len(data) > 0:
        return isint(data[0])
    elif isinstance(data, (int, np.integer)):
        return True
    elif is_jax_available() and isinstance(data, jnp.integer):
        return True
    elif isinstance(data, _TENSOR_TYPES):
        if isinstance(data, np.ndarray):
            return np.issubdtype(data.dtype, np.integer)
        elif is_torch_available() and isinstance(data, torch.Tensor):
            # Torch is no is_integer -> we need to check for not float and not complex
            return (
                not torch.is_floating_point(data) and not torch.is_complex(data) and data.dtype != torch.bool
            )
        elif is_jax_available() and isinstance(data, jnp.ndarray):
            return jnp.issubdtype(data.dtype, jnp.integer)
    return False


def isbool(data: Tensor):
    if isinstance(data, list) and len(data) > 0:
        return isbool(data[0])
    elif isinstance(data, tuple) and len(data) > 0:
        return isbool(data[0])
    elif isinstance(data, bool):
        return True
    elif isinstance(data, _TENSOR_TYPES):
        if isinstance(data, np.ndarray):
            return data.dtype == np.dtype(bool)
        elif is_torch_available() and isinstance(data, torch.Tensor):
            return data.dtype == torch.bool
        elif is_jax_available() and isinstance(data, jnp.ndarray):
            return data.dtype == jnp.dtype(jnp.bool_)
    return False


def get_default_dtype(data):
    if isfloat(data):
        return get_default_float_dtype()
    elif isbool(data):  # Note: bool test has to happen before int since isinstance(True, int) == True
        return bool
    elif isint(data):
        return get_default_int_dtype()
    else:
        raise ValueError(f"Cannot infer default dtype for data of type {type(data)}")


def int32():
    backend = get_backend()
    if is_numpy():
        return np.int32
    elif is_pytorch():
        return torch.int32
    elif is_jax():
        return jnp.int32
    else:
        raise MethodNotImplementedError(backend)


def int64():
    backend = get_backend()
    if is_numpy():
        return np.int64
    elif is_pytorch():
        return torch.int64
    elif is_jax():
        return jnp.int64
    else:
        raise MethodNotImplementedError(backend)


def float16():
    backend = get_backend()
    if is_numpy():
        return np.float16
    elif is_pytorch():
        return torch.float16
    elif is_jax():
        return jnp.float16
    else:
        raise MethodNotImplementedError(backend)


def float32():
    backend = get_backend()
    if is_numpy():
        return np.float32
    elif is_pytorch():
        return torch.float32
    elif is_jax():
        return jnp.float32
    else:
        raise MethodNotImplementedError(backend)


def float64():
    backend = get_backend()
    if is_numpy():
        return np.float64
    elif is_pytorch():
        return torch.float64
    elif is_jax():
        return jnp.float64
    else:
        raise MethodNotImplementedError(backend)


def boolean():
    backend = get_backend()
    if is_numpy():
        return np.bool_
    elif is_pytorch():
        return torch.bool
    elif is_jax():
        return jnp.bool_
    else:
        raise MethodNotImplementedError(backend)


def get_default_float_dtype():
    return float32()


def get_default_int_dtype():
    return int32()
