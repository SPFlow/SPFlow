from typing import List, Union

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope
from spflow.tensor.ops import Tensor


class LogNormalLayer:  # ToDo: backend über T.getBackend() abfragen
    def __new__(
        cls,
        scope: Union[Scope, list[Scope]],
        mean: Union[int, float, list[float], Tensor] = 0.0,
        std: Union[int, float, list[float], Tensor] = 1.0,
        n_nodes: int = 1,
        **kwargs,
    ):
        from spflow.base.structure.general.layer.leaf.log_normal import LogNormalLayer as TensorlyLogNormal
        from spflow.torch.structure.general.layer.leaf.log_normal import LogNormalLayer as TorchLogNormal

        """TODO"""
        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyLogNormal(scope=scope, mean=mean, std=std, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchLogNormal(scope=scope, mean=mean, std=std, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("LogNormal is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: list[FeatureContext]) -> bool:
        from spflow.base.structure.general.layer.leaf.log_normal import LogNormalLayer as TensorlyLogNormal
        from spflow.torch.structure.general.layer.leaf.log_normal import LogNormalLayer as TorchLogNormal

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyLogNormal.accepts(signatures)
        elif backend == "pytorch":
            return TorchLogNormal.accepts(signatures)
        else:
            raise NotImplementedError("LogNormal is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: list[FeatureContext]):
        from spflow.base.structure.general.layer.leaf.log_normal import LogNormalLayer as TensorlyLogNormal
        from spflow.torch.structure.general.layer.leaf.log_normal import LogNormalLayer as TorchLogNormal

        backend = T.get_backend()
        if backend == "numpy":
            return TensorlyLogNormal.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchLogNormal.from_signatures(signatures)
        else:
            raise NotImplementedError("LogNormal is not implemented for this backend")
