import torch

def complex_min(tensor):
    """
    Calculates the min for complex numbers by comparing the real part first and then the imaginary part.

    Args:
    tensor (torch.Tensor): A tensor containing complex numbers.

    Returns:
    complex: The minimum complex number.
    """
    if tensor.numel() == 0:
        raise ValueError("The input tensor is empty")

    # Initialize min with the first element of the tensor
    min_val = tensor[0]

    for value in tensor:
        if (value.real < min_val.real) or (value.real == min_val.real and value.imag < min_val.imag):
            min_val = value

    return min_val

def complex_max(tensor):
    """
    Calculates the max for complex numbers by comparing the real part first and then the imaginary part.

    Args:
    tensor (torch.Tensor): A tensor containing complex numbers.

    Returns:
    complex: The maximum complex number.
    """
    if tensor.numel() == 0:
        raise ValueError("The input tensor is empty")

    # Initialize max with the first element of the tensor
    max_val = tensor[0]

    for value in tensor:
        if (value.real > max_val.real) or (value.real == max_val.real and value.imag > max_val.imag):
            max_val = value

    return max_val

def complex_ge(tensor, value):
    """
    Compares a complex tensor with a complex value and returns a boolean tensor.

    Args:
    tensor (torch.Tensor): A tensor containing complex numbers.
    value (complex): A complex number to compare with.

    Returns:
    torch.Tensor: A boolean tensor indicating whether each element in the tensor is greater or equal to the value.
    """
    return (tensor.real > value.real) | ((tensor.real == value.real) & (tensor.imag >= value.imag))

def complex_le(tensor, value):
    """
    Compares a complex tensor with a complex value and returns a boolean tensor.

    Args:
    tensor (torch.Tensor): A tensor containing complex numbers.
    value (complex): A complex number to compare with.

    Returns:
    torch.Tensor: A boolean tensor indicating whether each element in the tensor is less or equal to the value.
    """
    return (tensor.real < value.real) | ((tensor.real == value.real) & (tensor.imag <= value.imag))


