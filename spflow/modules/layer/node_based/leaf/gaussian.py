from typing import List, Union

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope
from spflow.tensor.ops import Tensor


class GaussianLayer:  # ToDo: backend über T.getBackend() abfragen
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        mean: Union[int, float, list[float], Tensor] = 0.0,
        std: Union[int, float, list[float], Tensor] = 1.0,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.gaussian import GaussianLayer as TensorlyGaussian
        from spflow.torch.structure.general.layer.leaf.gaussian import GaussianLayer as TorchGaussian

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGaussian(scope=scope, mean=mean, std=std, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchGaussian(scope=scope, mean=mean, std=std, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.gaussian import GaussianLayer as TensorlyGaussian
        from spflow.torch.structure.general.layer.leaf.gaussian import GaussianLayer as TorchGaussian

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGaussian.accepts(signatures)
        elif backend == "pytorch":
            return TorchGaussian.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.gaussian import GaussianLayer as TensorlyGaussian
        from spflow.torch.structure.general.layer.leaf.gaussian import GaussianLayer as TorchGaussian

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGaussian.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchGaussian.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")


"""
    @property
    def dist(self):
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGaussian.dist
        elif backend == "pytorch":
            return TorchGaussian.dist
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    def set_params(self, mean: float, std: float) -> None:
        backend = T.get_backend()
        if backend == "numpy":
            TensorlyGaussian.set_params(mean=mean, std=std)
        elif backend == "pytorch":
            return TorchGaussian.set_params(mean=mean, std=std)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    def get_params(self) -> tuple[float, float]:
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGaussian.get_params()
        elif backend == "pytorch":
            return TorchGaussian.get_params()
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    def check_support(self, data: Tensor, is_scope_data: bool = False) -> Tensor:
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyGaussian.check_support(data=data, is_scope_data=is_scope_data)
        elif backend == "pytorch":
            return TorchGaussian.check_support(data=data, is_scope_data=is_scope_data)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")
"""
