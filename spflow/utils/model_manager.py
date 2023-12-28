import pickle
from spflow.modules.module import Module


def save_model(model: Module, file: str):
    pickle.dump(model, open(file, "wb"))


def load_model(file: str) -> Module:
    return pickle.load(open(file, "rb"))
