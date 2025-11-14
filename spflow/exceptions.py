"""Exception classes for SPFlow probabilistic circuits.

Defines custom exceptions for parameter validation, scope errors, tensor
shape mismatches, structure errors, and external dependency issues.
"""


class InvalidParameterCombinationError(Exception):
    """Raised when incompatible parameters are provided together."""

    pass


class ScopeError(Exception):
    """Raised when variable scopes are invalid or incompatible."""

    pass


class ShapeError(ValueError):
    """Raised when tensor shapes don't meet expected requirements."""

    pass


class StructureError(ValueError):
    """Raised when circuit structure or configuration is invalid."""

    pass


class GraphvizError(Exception):
    """Raised when Graphviz is not installed or fails to execute.

    Includes detailed installation instructions in error message.
    """

    INSTALL_INSTRUCTIONS = (
        "To fix this issue:\n"
        "  1. Install the Graphviz system dependency:\n"
        "     - On macOS: brew install graphviz\n"
        "     - On Ubuntu/Debian: sudo apt-get install graphviz\n"
        "     - On Windows: Download from https://graphviz.org/download/\n"
        "  2. Verify installation by running: dot -V\n"
        "  3. Check that Graphviz binaries are in your system PATH\n\n"
        "For more details, see the README.md file in the SPFlow repository."
    )

    def __str__(self) -> str:
        """Return error message with installation instructions."""
        message = super().__str__()
        return f"{message}\n\n{self.INSTALL_INSTRUCTIONS}"
