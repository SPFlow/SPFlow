import os
from typing import TypeAlias, Union
import pickle

from spflow.modules.module import Module


PathLike: TypeAlias = Union[str, bytes, os.PathLike]


def save_model(model: Module, path: PathLike) -> None:
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path: PathLike) -> Module:
    with open(path, "rb") as file:
        return pickle.load(file)
