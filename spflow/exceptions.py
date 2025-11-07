class InvalidParameterCombinationError(Exception):
    """Raised when an invalid combination of parameters is provided."""

    pass


class ScopeError(Exception):
    """Raised when an invalid scope is provided."""

    pass


class GraphvizError(Exception):
    """Raised when Graphviz is not installed or fails to execute."""

    pass
