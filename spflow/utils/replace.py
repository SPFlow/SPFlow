"""Context manager for temporarily replacing class methods.

Provides a clean API for method substitution with automatic handling of
decorators like @cached. Useful for testing, debugging, and experimentation.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Callable, TypeVar

T = TypeVar("T")


@contextmanager
def replace(method_ref: Callable, replacement_func: Callable):
    """Temporarily replace a class method with a custom implementation.

    Automatically detects and preserves decorators (like @cached) by re-applying
    them to the replacement function. Works at the class level, affecting all
    instances of the class.

    Args:
        method_ref: Reference to the method to replace (e.g., Sum.log_likelihood).
            This should be an unbound method (accessed via the class, not instance).
        replacement_func: The function to use as replacement. Must have a compatible
            signature with the original method (including 'self' as first parameter).

    Yields:
        None.

    Example:
        ```python
        def my_custom_ll(self, data, cache=None):
            # Custom implementation
            return torch.ones(len(data))

        model = Product(Sum(Product(Normal(...))))

        # Normal inference
        model.log_likelihood(data)

        # Use custom implementation for Sum modules
        with replace(Sum.log_likelihood, my_custom_ll):
            model.log_likelihood(data)  # Sum instances now use my_custom_ll
        ```

    Raises:
        ValueError: If the class cannot be inferred from the method reference.
        TypeError: If method_ref is not a valid method reference.
    """
    # Extract class and method name from method reference
    target_class, method_name = _extract_class_and_name(method_ref)

    # Get the original method
    original_method = getattr(target_class, method_name)

    # Detect if the original method is decorated (e.g., with @cached)
    # Methods decorated with @functools.wraps have __wrapped__ attribute
    is_decorated = hasattr(original_method, "__wrapped__")

    # Prepare the replacement method
    if is_decorated:
        # Re-apply the decorator to the replacement function
        # Import here to avoid circular imports
        from spflow.utils.cache import cached

        new_method = cached(replacement_func)
    else:
        new_method = replacement_func

    # Replace the method on the class
    setattr(target_class, method_name, new_method)

    try:
        yield
    finally:
        # Restore the original method
        setattr(target_class, method_name, original_method)


def _extract_class_and_name(method_ref: Callable) -> tuple[type, str]:
    """Extract the owner class and method name from a method reference.

    Args:
        method_ref: An unbound method reference (e.g., Sum.log_likelihood).

    Returns:
        A tuple of (owner_class, method_name).

    Raises:
        ValueError: If the class cannot be extracted from the method reference.
        TypeError: If method_ref is not a valid method reference.
    """
    # Verify it's a callable
    if not callable(method_ref):
        raise TypeError(f"Expected a callable method reference, got {type(method_ref)}")

    # Get the method name
    method_name = getattr(method_ref, "__name__", None)
    if method_name is None:
        raise TypeError("Method reference must have a __name__ attribute")

    # Get the qualified name to extract the class
    # __qualname__ looks like "ClassName.method_name"
    qualname = getattr(method_ref, "__qualname__", None)
    if qualname is None:
        raise ValueError("Cannot determine class from method reference: missing __qualname__")

    # Parse the qualified name to get the class name
    if "." not in qualname:
        raise ValueError(
            f"Cannot determine class from method reference: "
            f"__qualname__='{qualname}' has no '.' separator"
        )

    class_name = qualname.rsplit(".", 1)[0]

    # Get the module containing the method
    module = inspect.getmodule(method_ref)
    if module is None:
        raise ValueError("Cannot determine module for method reference")

    # Try to get the class from the module's globals
    target_class = getattr(module, class_name, None)
    if target_class is None:
        raise ValueError(
            f"Cannot find class '{class_name}' in module '{module.__name__}'. "
            f"Method reference qualname: {qualname}"
        )

    return target_class, method_name
