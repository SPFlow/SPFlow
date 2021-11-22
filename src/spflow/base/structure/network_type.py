"""
Created on Okt 12, 2021

@authors: Kevin Huy Nguyen

This file provides the abstract NetworkType class for dispatching methods on specific networks.
"""
from contextlib import contextmanager
from abc import ABC


class NetworkType(ABC):
    """Abstract network type class for dispatching methods."""


class UnspecifiedNetworkType(NetworkType):
    """Dummy networktype acting as a placeholder when no networktype is specified."""

    def __repr__(self) -> str:
        return "UnspecifiedNetworkType"


class SPN(NetworkType):
    """Class for the network type 'Sum-Product Network'."""

    def __repr__(self) -> str:
        return "SumProductNetwork"


class BN(NetworkType):
    """Class for the network type 'Bayesian Networks'."""

    def __repr__(self) -> str:
        return "BayesianNetwork"


# Global networktype
NETWORK_TYPE_CURRENT: NetworkType = UnspecifiedNetworkType()


@contextmanager
def set_network_type(network_type: NetworkType):
    """
    ContextManager to set the current NetworkType. When the context exits, the networktype is reset to
    its previous value.

    Args:
      network_type (NetworkType): New global networktype.
    """
    # Get original networktype
    global NETWORK_TYPE_CURRENT
    networktype_previous = NETWORK_TYPE_CURRENT

    # Set networktype
    NETWORK_TYPE_CURRENT = network_type

    # Enter with statement
    yield

    # Reset networktype
    NETWORK_TYPE_CURRENT = networktype_previous


def get_network_type() -> NetworkType:
    """
    Get the current global networktype.

    Returns:
        NetworkType: The current global NetworkType.
    """
    return NETWORK_TYPE_CURRENT
