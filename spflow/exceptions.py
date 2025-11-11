class InvalidParameterCombinationError(Exception):
    """Raised when an invalid combination of parameters is provided."""

    pass


class ScopeError(Exception):
    """Raised when an invalid scope is provided."""

    pass


class ShapeError(ValueError):
    """Raised when tensor shape or dimensions don't meet expected requirements."""

    pass


class StructureError(ValueError):
    """Raised when module structure or configuration is invalid."""

    pass


class GraphvizError(Exception):
    """Raised when Graphviz is not installed or fails to execute."""

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
        """Return the error message with installation instructions appended."""
        message = super().__str__()
        return f"{message}\n\n{self.INSTALL_INSTRUCTIONS}"
