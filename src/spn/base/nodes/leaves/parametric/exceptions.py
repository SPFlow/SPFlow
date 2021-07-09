"""
Created on June 26, 2021

@authors: Bennet Wittelsbach
"""


class InvalidParametersError(Exception):
    """Exception used in parametric leafs and thrown if the parameters of a distribution are not available or
    invalid."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
